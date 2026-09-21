# WeDetect and OWLv2 model-selection note

## Decision

Use **WeDetect-Uni or WeDetect-Anything as the next class-independent proposal candidate**, retain **OWLv2 as a prompted and alternate objectness source**, and evaluate their union before choosing a production discovery route. Neither model may accept an annotation from its own score. Naming, boundary refinement, independent verification, and uncertainty assessment remain separate stages.

This is a candidate-selection decision, not a quality claim. The published WeDetect and OWLv2 numbers use different datasets, training mixtures, checkpoints, and evaluation protocols, so they do not establish which model is better for Lynceus.

## Why WeDetect fits discovery better

WeDetect-Uni is explicitly trained to retrieve generic object proposals through an objectness prompt while the detector is frozen. WeDetect-Anything combines prompt-free foreground proposals with region embeddings and vocabulary retrieval. This maps directly to Lynceus's need to find objects before deciding what they are. It also offers a deployment path that avoids MMDetection for inference.

The current OWLv2 adapter searches a finite ontology with text prompts. That is useful for known concepts, but it makes proposal recall depend on the supplied vocabulary. OWLv2 also exposes a query-independent objectness head; a fair comparison must implement that route directly rather than treating prompted OWLv2 as its only mode.

## Comparison

| Dimension | WeDetect-Uni / Anything | OWLv2 | Delivery implication |
|---|---|---|---|
| Prompt-free proposals | First-class purpose | Objectness exists, but the current adapter does not expose it | Trial WeDetect first; add direct OWLv2 objectness |
| Naming | Region embeddings matched to a large vocabulary; the base WeDetect prompts are Chinese | Flexible text queries through a CLIP-style detector | Keep naming behind a common resolver and test multilingual vocabulary effects |
| Small objects | 640 and 1280 input variants; still requires tiling and cap measurement | Large checkpoints and tiled inference are already integrated | Run identical source-resolution tiles and proposal budgets |
| Proposal limits | Example pipeline starts from 8,400 locations and defaults to top 100 after filtering | Query and post-processing caps depend on implementation | Expose every cap and report cap hits; never silently truncate |
| Tight boundaries | Box detector; optional mask-refine training is described | Box detector | Retain a separate source-resolution segmentation/refinement stage |
| Deployment | PyTorch and ONNX examples; self-contained inference example avoids MMDetection | Existing local Transformers adapter; official JAX implementation | Prefer ONNX for an isolated WeDetect worker if parity is verified |
| Maturity | Very recent CVPR 2026 project | Older, widely reproduced baseline | Keep OWLv2 as the stable baseline until same-data evidence is complete |
| License | Repository states GPL-v3 for models and code | Official Scenic implementation is Apache-2.0; checkpoint terms still need recording | Resolve distribution and derivative-work obligations before bundling WeDetect |
| Evidence diversity | YOLO-World-derived convolutional/FPN retrieval detector | ViT/CLIP-style open-vocabulary detector | Their union can improve coverage, but agreement is not a probability of correctness |

## Same-data experiment

Freeze the images, policy, ontology, tile schedule, proposal caps, and evaluator before running:

1. Current prompted OWLv2 baseline.
2. OWLv2 query-independent objectness proposals.
3. WeDetect-Uni proposals without vocabulary filtering.
4. WeDetect-Anything proposals plus its frozen vocabulary match.
5. Candidate union, followed by the same deduplication, boundary, naming, and verification stages.

Report class-neutral recall at IoU 0.5, retained precision, duplicate/merge errors, small-object and low-contrast strata, proposal-cap hit rate, wall time, peak VRAM/RAM, and failures. Keep every raw model proposal `uncertain` until an independently qualified verification rule supports acceptance.

## Implementation order

1. Generalize the current detector boundary into a model-neutral `ProposalAdapter` that returns boxes, objectness, embeddings when available, view coordinates, cap metadata, and provenance.
2. Add direct OWLv2 objectness extraction so the baseline comparison is fair.
3. Add an isolated WeDetect ONNX adapter with pinned preprocessing, hashes, coordinate tests, and offline network-denial tests.
4. Run the frozen comparison and choose the default discovery route from Lynceus metrics, not published AP.
5. Add candidate union only if its incremental recall justifies the precision and compute cost.

## Primary sources

- [WeDetect repository and model family](https://github.com/WeChatCV/WeDetect)
- [WeDetect paper](https://arxiv.org/abs/2512.12309)
- [WeDetect-Anything instructions](https://github.com/WeChatCV/WeDetect/blob/main/wedetect_anything/README.md)
- [WeDetect deployment instructions](https://github.com/WeChatCV/WeDetect/blob/main/deploy/README.md)
- [OWLv2 paper](https://arxiv.org/abs/2306.09683)
- [Official OWL-ViT/OWLv2 implementation](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
