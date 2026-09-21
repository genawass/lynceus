# Offline foundation: measured feasibility

Date: 2026-09-19. This report separates checks performed in this workspace from deployment and accuracy claims that require additional resources.

## Environment observed

| Resource | Observation |
|---|---|
| CPU | Intel Core i7-11700, 16 logical CPUs exposed |
| RAM | Approximately 31.2 GiB total; available memory is a time-dependent observation |
| Storage | Approximately 343 GiB free when inventoried |
| Python | 3.12.9 |
| PyTorch | Runtime 2.6.0+cu124; CUDA unavailable in this execution environment |
| GPU access | **Superseded.** The first inventory found no `/dev/nvidia*` nodes and `nvidia-smi` exiting 9. On re-measurement the nodes are present, `nvidia-smi` succeeds, and `torch.cuda.is_available()` is true: one RTX 3060, 12288 MiB, driver 580.178.04, compute capability 8.6. Treat the first observation as a property of that process, not of the host |
| Transformers | Runtime reports 5.3.0; installed distribution metadata reports 4.49.0 |
| NumPy | Runtime imports 1.26.4; metadata reports 2.2.6. Two dist-info directories (`numpy-1.26.4.dist-info`, `numpy-2.2.6.dist-info`) are present in the same site-packages |
| Model cache | No model directories found in the default Hugging Face hub cache |
| Independent benchmark images | None supplied or staged in this project |

The original GPU observation did not prove the host had no GPU, only that that process could not use one — and a later process could. A negative capability observation is valid for the process that made it and must be re-measured before it constrains a deployment profile. The Transformers and NumPy metadata mismatches also mean this ambient environment is not a clean, reproducible deployment bundle. Record both versions, then stage a consistent isolated environment before production model-feasibility claims.

`importlib.metadata` alone is not a trustworthy environment report here: it disagrees with the imported module for two of the four pinned runtime packages. `lynceus doctor` therefore reports `versions` (metadata) and `runtime_versions` (imported) separately and lists `version_conflicts`, and bundle preflight rejects a declared dependency whose imported version differs from the manifest. Local build suffixes such as PyTorch's `+cu124` are compared on the public version only.

Machine-readable observations are in [hardware-inventory.json](hardware-inventory.json). Resource observations are a snapshot, not a capacity guarantee.

## Checks actually performed

- Imported `Owlv2Processor` and `Owlv2ForObjectDetection` with offline environment flags.
- Executed a tiny, randomly initialized OWLv2 on CPU while socket connection methods were denied.
- Observed finite objectness values and box coordinates, with objectness shape `[1, 4]`, box shape `[1, 4, 4]`, and class logits shape `[1, 4, 2]`.
- Inspected the installed OWLv2 processor's coordinate scaling. `Owlv2ImageProcessor.pad` squares the image with bottom/right padding of side `max(height, width)`, so predicted boxes are normalized to that square and not to the source. The adapter unscales with `target_sizes = [[side, side]]` and then clips the padding area away; unscaling with the source height/width would shrink every box on a non-square image. This correction is derived from the installed processor's source and remains unverified against trained weights.

The [API smoke result](owlv2-api-smoke.json) records the random-weight test. It establishes local API compatibility only. It does not establish trained-model inference, accuracy, usable class labels, GPU memory fit, or performance of the proposed complete pipeline.

The [official Transformers OWLv2 documentation](https://huggingface.co/docs/transformers/model_doc/owlv2) describes the objectness and detection interfaces. The [Google checkpoint page](https://huggingface.co/google/owlv2-base-patch16-ensemble) identifies a candidate pretrained artifact. Neither source establishes the quality of a Lynceus annotation.

## Current delivery boundary

The foundation supports building and validating the offline contract and measuring recorded predictions. Its baseline adapter is designed for pre-staged local OWLv2 weights. Baseline proposals remain uncertain because source-boundary refinement, semantic verification, and the independent omission-search routes have not been qualified.

`profile_complete` means the baseline schedule ran. It is not exhaustive saturation, complete labeling, or reference qualification. There is no model-quality result yet.

## Resources needed for the next measured step

1. A local, integrity-manifested OWLv2 checkpoint and processor/tokenizer assets. CPU execution may be used for an initial correctness smoke test; latency and memory still need measurement.
2. A clean supported runtime or an environment that exposes the intended GPU devices. Measure resource use with the actual model and image sizes before selecting larger candidates.
3. Independent reference images and annotations with compatible visible-extent boxes, ontology mapping, scope, and source/split provenance. Synthetic fixtures are not a substitute for these data.
4. For the full specification, local semantic-inventory and concept/segmentation model bundles and an empirically tested qualification domain.

Asset staging happens before offline inference. Runtime must never silently retrieve missing weights or send image content to a service. No user image or benchmark data has been uploaded by this work.

## Measured with staged weights

Date: 2026-09-19. `google/owlv2-large-patch14-ensemble`, 1.7 GiB across 8 hash-manifested assets, run through `lynceus annotate` on VisDrone frames of 1360x765.

| Configuration | Views per image | Wall seconds per image | Peak RSS |
|---|---|---|---|
| `--tile-levels 0`, CPU | 1 | 85 | 5.8 GiB |
| `--tile-levels 2`, CUDA | 14 | 58 | 2.6 GiB |

GPU utilisation between views is low: a separate process per image reloads the 1.7 GiB checkpoint each time, so load dominates a single-image invocation. Batch throughput would improve by reusing one adapter across images, which the single-image CLI contract (assumption A1) does not currently do.

CPU and CUDA runs of the same image and configuration produced slightly different retained-object counts (78 versus 71 on `0000069_01878_d_0000005`), from float differences near the score threshold. WP-14 requires a preregistered reproducibility tolerance; device is part of the configuration a qualification identity must bind, and these runs are not evidence that any tolerance is met.

### OWLv2 padding correction, measured

On `0000069_01878_d_0000005` (1360x765, padded side 1360), scoring retained boxes against the 36 eligible VisDrone references at IoU 0.5:

| Unscaling | References matched | Mean IoU of matches |
|---|---|---|
| `target_sizes=[[side, side]]` (correct) | 31 / 36 | 0.791 |
| `target_sizes=[[height, width]]` (previous) | 2 / 36 | 0.510 |

The previous form compressed every y coordinate by 765/1360. Every VisDrone frame is non-square, so the defect was catastrophic there and would have been invisible on square test images.
