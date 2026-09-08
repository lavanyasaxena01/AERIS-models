# AERIS Models

This repository contains two independent change-detection models and their inference visualizations.

## Model 1

Model 1 uses `SiameseTemporalCD` for optical temporal change detection. It takes an optical T1/T2 image pair and produces:

- the real dataset ground-truth annotation when `--gt` is supplied;
- `prediction_mask.png`, the binary model prediction at the selected model threshold;
- `probability.npy` and `probability_map.png`, the continuous pixel-wise model probability;
- `changed_regions.png`, a visualization-only grouping of probability-based regions over T2;
- `region_mask.png`, the binary mask used only for that region visualization;
- comparison and image outputs, plus IoU, Dice/F1, precision, recall, and accuracy when ground truth is supplied.

The region labels (`R1: detected change`, `R2: detected change`, and so on) are descriptive visualization labels only. Model 1 does not perform semantic class prediction.

Run the tested Model 1 example from the repository root:

```powershell
python inference/predict.py --model optical --checkpoint runs/optical/best.pt --t1 "datasets/LEVIR CD/test/A/test_1.png" --t2 "datasets/LEVIR CD/test/B/test_1.png" --gt "datasets/LEVIR CD/test/label/test_1.png" --out outputs/model1_test
```

The actual prediction remains `pred = (prob >= threshold).astype(np.uint8)`. The separate probability-based region threshold affects only `changed_regions.png` and `region_mask.png`.

## Model 2

Model 2 uses `OpticalSARChangeNet` with optical T1/T2 and SAR T1/T2 inputs. Its existing semantic change pipeline preserves the model's binary change prediction and adds region grouping/classification-style visualization from detected regions. The semantic interpretation is a lightweight post-processing layer, not a replacement model.

Run the tested Model 2 example with the valid checkpoint:

```powershell
python inference/semantic_change.py --query "general land cover change" --opt-t1 "datasets/Opt-sar-model_dataset/test/optical_t1/scene_0000.png" --opt-t2 "datasets/Opt-sar-model_dataset/test/optical_t2/scene_0000.png" --sar-t1 "datasets/Opt-sar-model_dataset/test/sar_t1/scene_0000.png" --sar-t2 "datasets/Opt-sar-model_dataset/test/sar_t2/scene_0000.png" --checkpoint runs/optical_sar/last.pt --gt "datasets/Opt-sar-model_dataset/test/labels/scene_0000.png" --threshold 0.3 --min-region 20 --out outputs/model2_test
```

The tested `runs/optical_sar/last.pt` checkpoint loads successfully. The local `runs/optical_sar/best.pt` was corrupted and must not be renamed or substituted; provide a valid checkpoint when running Model 2.

## Output terminology

- **Ground Truth**: the actual annotation loaded from the dataset.
- **Prediction Mask**: the binary output produced by the model threshold and existing post-processing.
- **Probability Map**: the continuous sigmoid probability output from the model.
- **Changed Regions**: a visualization of detected regions over the current/T2 image.
- **Semantic Region Visualization**: Model 2's region grouping and lightweight interpretation layer; it is not ground truth and does not replace the model prediction.

Generated outputs, datasets, training runs, checkpoints, virtual environments, and caches are excluded by `.gitignore`.
