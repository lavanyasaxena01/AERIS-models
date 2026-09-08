
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from models.optical_sar_multimodal import OpticalSARChangeNet
from models.temporal_optical import SiameseTemporalCD
from training.losses import build_loss
from training.metrics import ChangeMetrics
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import overlay_mask, save_panel, to_rgb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W_SMALL = (32, 64, 128, 256)  # smoke-test widths (same architecture, lighter for CPU)
OUT = Path(__file__).resolve().parents[1] / "outputs"


def make_optical_batch(b=2, c=3, h=128, w=128, seed=0):
    rng = np.random.RandomState(seed)
    t1 = torch.from_numpy(rng.randn(b, c, h, w).astype(np.float32))
    t2 = t1 + 0.05 * torch.from_numpy(rng.randn(b, c, h, w).astype(np.float32))
    mask = torch.zeros(b, 1, h, w)
    for i in range(b):
        for _ in range(3):
            y = rng.randint(0, h - 24); x = rng.randint(0, w - 24)
            hh = rng.randint(8, 24); ww = rng.randint(8, 24)
            t2[i, :, y:y + hh, x:x + ww] += 2.0
            mask[i, 0, y:y + hh, x:x + ww] = 1.0
    return t1, t2, mask


def make_mm_batch(b=2, co=3, cs=1, h=128, w=128, seed=0):
    rng = np.random.RandomState(seed)
    ot1 = torch.from_numpy(rng.randn(b, co, h, w).astype(np.float32))
    st1 = torch.from_numpy(rng.randn(b, cs, h, w).astype(np.float32))
    ot2 = ot1 + 0.05 * torch.from_numpy(rng.randn(b, co, h, w).astype(np.float32))
    st2 = st1 + 0.05 * torch.from_numpy(rng.randn(b, cs, h, w).astype(np.float32))
    mask = torch.zeros(b, 1, h, w)
    for i in range(b):
        for _ in range(3):
            y = rng.randint(0, h - 24); x = rng.randint(0, w - 24)
            hh = rng.randint(8, 24); ww = rng.randint(8, 24)
            ot2[i, :, y:y + hh, x:x + ww] += 2.0
            st2[i, :, y:y + hh, x:x + ww] += 1.5
            mask[i, 0, y:y + hh, x:x + ww] = 1.0
    return ot1, ot2, st1, st2, mask


def check_forward_backward(name, model, inputs, target):
    model = model.to(DEVICE)
    inputs = [x.to(DEVICE) for x in inputs]
    target = target.to(DEVICE)
    logits = model(*inputs)
    assert logits.shape == (target.shape[0], 1, *target.shape[-2:]), f"{name}: bad output {logits.shape}"
    loss = build_loss({"type": "bce_dice", "pos_weight": 5.0})(logits, target)
    loss.backward()
    nz = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    tot = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[{name}] forward OK logits={tuple(logits.shape)} loss={loss.item():.4f} non-zero grads {nz}/{tot}")
    assert nz > 0, f"{name}: no gradients!"
    return model


def check_arbitrary_size(name, model, batch_fn):
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        batch = batch_fn(h=96, w=168)  # divisible by 8, NOT by 32
        out = model(*[x.to(DEVICE) for x in batch[:-1]])
    assert out.shape[-2:] == (96, 168), f"{name}: {out.shape}"
    print(f"[{name}] arbitrary size 96x168 (out_strides=(4,8,16)) OK -> logits {tuple(out.shape)}")


def overfit(name, model, inputs, target, steps=150, lr=3e-3):
    model = model.to(DEVICE); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = build_loss({"type": "bce_dice", "pos_weight": 5.0})
    inputs = [x.to(DEVICE) for x in inputs]
    target = target.to(DEVICE)
    losses = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(*inputs), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (i + 1) % 25 == 0:
            print(f"[{name}] overfit step {i + 1}/{steps} loss={np.mean(losses[-25:]):.4f}")
    first, last = float(np.mean(losses[:25])), float(np.mean(losses[-25:]))
    print(f"[{name}] overfit loss {first:.4f} -> {last:.4f}")
    assert last < first, f"{name}: loss did not decrease"
    return model


def check_metrics(name, model, inputs, target):
    model.eval()
    with torch.no_grad():
        logits = model(*[x.to(DEVICE) for x in inputs])
    cm = ChangeMetrics(threshold=0.5)
    cm.update(logits, target.to(DEVICE))
    m = {k: round(v, 4) for k, v in cm.compute().items()}
    print(f"[{name}] train-set metrics after overfit: {m}")
    assert 0.0 <= m["f1"] <= 1.0


def checkpoint_roundtrip(name, model, model_fn, inputs):
    model.eval()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint({"epoch": 1, "model": model.state_dict(), "best": 0.5,
                         "config": {"model": {}}}, str(p))
        fresh = model_fn().to(DEVICE)
        load_checkpoint(str(p), fresh)
        fresh.eval()
        with torch.no_grad():
            a = model(*[x.to(DEVICE) for x in inputs])
            b = fresh(*[x.to(DEVICE) for x in inputs])
    assert torch.allclose(a, b, atol=1e-6), f"{name}: checkpoint round-trip mismatch"
    print(f"[{name}] checkpoint save/load OK (reloaded outputs identical)")


def inference_artifacts(name, model, np_inputs, gt):
    model.eval()
    with torch.no_grad():
        logits = model(*[torch.from_numpy(x).to(DEVICE) for x in np_inputs])
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob >= 0.5).astype(np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    save_panel(str(OUT / f"{name}_panel.png"),
               t1=np_inputs[0][0], t2=np_inputs[1][0], gt=gt[0, 0].numpy(),
               prob=prob, pred=pred, title=name)
    overlay = overlay_mask(to_rgb(np_inputs[0][0]), pred)
    save_panel(str(OUT / f"{name}_overlay.png"), t1=overlay)
    print(f"[{name}] inference OK prob=[{prob.min():.3f},{prob.max():.3f}] "
          f"changed px={int(pred.sum())} -> {OUT}/{name}_panel.png")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"device: {DEVICE} | torch {torch.__version__}")

    # ============ MODEL 1 ============
    m1 = SiameseTemporalCD(in_channels=3, encoder_widths=W_SMALL, out_strides=(4, 8, 16, 32))
    print(f"[model1] params: {sum(p.numel() for p in m1.parameters()) / 1e6:.2f}M")
    t1, t2, mask = make_optical_batch()
    m1 = check_forward_backward("model1", m1, [t1, t2], mask)
    check_arbitrary_size("model1",
                         SiameseTemporalCD(in_channels=3, encoder_widths=W_SMALL[:3], encoder_blocks=(2, 2, 2),
                                           out_strides=(4, 8, 16)),
                         make_optical_batch)
    m1 = overfit("model1", m1, [t1, t2], mask)
    check_metrics("model1", m1, [t1, t2], mask)
    checkpoint_roundtrip("model1", m1,
                         lambda: SiameseTemporalCD(in_channels=3, encoder_widths=W_SMALL, out_strides=(4, 8, 16, 32)),
                         [t1, t2])
    inference_artifacts("model1", m1, [t1.numpy(), t2.numpy()], mask)

    # ============ MODEL 2 ============
    m2 = OpticalSARChangeNet(optical_channels=3, sar_channels=1, encoder_widths=W_SMALL,
                             out_strides=(4, 8, 16, 32), fusion_mode="cross_attention")
    print(f"[model2] params: {sum(p.numel() for p in m2.parameters()) / 1e6:.2f}M")
    ot1, ot2, st1, st2, mask2 = make_mm_batch()
    m2 = check_forward_backward("model2", m2, [ot1, ot2, st1, st2], mask2)
    check_arbitrary_size("model2",
                         OpticalSARChangeNet(optical_channels=3, sar_channels=1, encoder_widths=W_SMALL[:3],
                                             encoder_blocks=(2, 2, 2), out_strides=(4, 8, 16), fusion_mode="gated"),
                         make_mm_batch)
    m2 = overfit("model2", m2, [ot1, ot2, st1, st2], mask2)
    check_metrics("model2", m2, [ot1, ot2, st1, st2], mask2)
    checkpoint_roundtrip("model2", m2,
                         lambda: OpticalSARChangeNet(optical_channels=3, sar_channels=1, encoder_widths=W_SMALL,
                                                     out_strides=(4, 8, 16, 32), fusion_mode="cross_attention"),
                         [ot1, ot2, st1, st2])
    inference_artifacts("model2", m2, [ot1.numpy(), ot2.numpy(), st1.numpy(), st2.numpy()], mask2)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
