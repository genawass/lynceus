# Lynceus

Offline automatic object annotation from a single image, with explicit uncertainty and measured qualification for downstream use.

This repository contains a delivery specification and a runnable offline foundation slice. **It is not a qualified annotator.** No pretrained weights are staged here, no image has been annotated by a trained model, and no quality benchmark has run. Every output is `unqualified`.

## Contract

Start with the [BMAD specification](_bmad-output/specs/spec-lynceus/SPEC.md), then read its six companions. The [delivery plan](_bmad-output/specs/spec-lynceus/delivery-plan.md) defines ordered work packages, dependencies, acceptance criteria, and release gates. The [specification review](_bmad-output/reviews/lynceus-spec-review.md) records readiness and source preservation.

The specification retains the goal of complete reference annotations while making an essential distinction: search saturation and model agreement cannot certify that an arbitrary image has no missed objects. Reference use requires independent, domain-specific qualification; unresolved cases remain explicit.

## Implemented slice

[Runnable offline annotation foundation](_bmad-output/implementation-artifacts/spec-offline-foundation.md): policy/ontology, native schema and cross-field invariants, image normalization, local bundle preflight, a conservative OWLv2 baseline adapter, immutable artifacts with safe resume, and independent-prediction evaluation. Scope and remaining work are tracked in [delivery status](docs/delivery-status.md); measured environment facts are in [feasibility](docs/feasibility.md). The [autolabel quality strategy](docs/autolabel-strategy.md) compares zero-shot, domain-adapted, decoupled, semi-supervised, verifier-cascade, and agentic directions; the [WeDetect versus OWLv2 decision](docs/model-selection.md) defines the next discovery-model experiment within that strategy.

## Install and run

```sh
pip install -e '.[test]'          # add '.[model]' only when staging real OWLv2 weights
python -m pytest
```

| Command | Purpose |
|---|---|
| `lynceus doctor --offline [--bundle PATH]` | Report environment, dependency version conflicts, and staged-bundle preflight; loads no model |
| `lynceus policy` | Emit the versioned annotation policy and ontology |
| `lynceus schema` | Emit the `lynceus.annotation/1.0` JSON Schema |
| `lynceus validate ANNOTATION [--artifact-root DIR]` | Validate a result and its artifact hashes |
| `lynceus annotate IMAGE --bundle PATH --output DIR` | Run discovery into a new run directory (see tiling below) |
| `lynceus annotate ... --refine-bundle PATH` | Refine boundaries against pixels with a second model; off unless named |
| `lynceus resume RUN [--output DIR]` | Re-validate a run; continue an interrupted one into a new directory |
| `lynceus evaluate --benchmark FILE` | Score recorded independent predictions; never grants qualification |

### Tiled discovery

`--tile-levels k` adds overlapping source-resolution tiles to the full-frame pass. Tiles are cropped before the processor resizes, so a smaller tile reaches the model at higher magnification — which is what small objects need when a 1360x765 frame would otherwise be downscaled to the model's 1008px input.

Each level's grid **follows the image aspect ratio** rather than being square. The processor pads every view to a square before resizing, so a square grid on a 16:9 image produces 16:9 tiles that spend ~44% of each forward pass on padding. Distributing the same tile budget by aspect keeps tiles near square:

| image | level 1 grid | tiles | magnification | padding waste |
|---|---|---|---|---|
| 1360x765 (16:9) | 3x2 | 6 | 1.85x | 19.5% |
| 1024x1024 | 2x2 | 4 | 1.72x | 0% |
| 765x1360 (portrait) | 2x3 | 6 | 1.85x | 19.5% |

`--tile-overlap` is the linear fraction two adjacent tiles share, so a border object is whole in at least one view. Coverage is always exactly 1.0 at every level.

#### Merging repeated views

Tiling and dense scenes pull the geometry in opposite directions:

- A detection **truncated at a tile border** overlaps its whole-object counterpart with *low IoU but high intersection-over-smaller*. IoU alone under-merges exactly the duplicates slicing creates.
- IoS is *also* high whenever any small box sits inside a larger one — ordinary crowd geometry. IoS alone collapses distinct neighbours, which is the suppression of a valid separate instance the annotation policy forbids.

[SAHI](https://arxiv.org/abs/2202.06934) resolves this by choosing one metric globally. Here the recorded truncation flag separates the two cases instead: **IoS is trusted only when one observation reaches its view border; otherwise the two must agree on IoU.** Measured on 11,553 observations from a 14-view run, IoS-only merging destroyed 13% of correct detections by collapsing crowd neighbours; the conditional rule recovers most of them.

Only same-label observations merge, so a backpack inside a person's box survives. The winning box is kept rather than a union of the merged boxes, because the annotation policy forbids asserting extent that no single view supported — losing boxes become `alternative_boxes` and set `boundary: unresolved`. Every discarded view stays in `evidence/candidates.jsonl` with its disposition and merge reason.

`--tile-levels 0` (the default) is the single full-image pass, which is the baseline the G7 iterative-benefit gate measures against. Tiling never licenses `search: saturated`; the profile becomes `owlv2-tiled-Lk-v1` and still reports `profile_complete`. A failed view is recorded as an `unsearched` region and forbids a complete scan status.

### Ontologies

Labels resolve against a versioned ontology, never free text. `lynceus policy` lists what this build ships; `--ontology <id>` selects one for a run. An annotation records the ontology it used, and validation resolves that name against the shipped registry byte for byte, so a result cannot claim an ontology this build does not contain.

| id | useful classes | for |
|---|---|---|
| `lynceus-broad-v1` (default) | 24 | general photographs |
| `aerial-traffic-v1` | 11 | overhead and oblique street scenes |

The prompt list is the ontology's useful classes, so ontology choice is also prompt-set choice: the detector scores all queries competitively, and irrelevant classes both waste compute and steal probability mass from valid detections. A domain ontology must be **declared before measurement** under assumption A8 — selecting classes after seeing results is the dataset-level prompt optimization the contract forbids. Changing ontology changes the qualification identity and invalidates any qualification tied to the previous one.

### Acceptance

Without `--accept` every object stays `uncertain` with all five uncertainty dimensions `not_assessed`: the discovery routes alone produce unverified proposals, and the contract forbids presenting those as supported. `--accept` runs a frozen rule that marks a dimension `supported` only where a check ran and passed:

| dimension | supported when |
|---|---|
| `existence` | score meets `--accept-score` and the instance was seen in `--accept-min-views` views, capped by the number of views the schedule actually planned |
| `class` | the label is a useful ontology class and the score threshold is met |
| `boundary` | not truncated at a view border, and any alternative boxes agree within `--accept-boundary-iou` |
| `count` | the object is an `instance`, where the count is 1 by construction rather than inferred from density |
| `granularity` | no larger retained object contains it beyond `--accept-granularity-ios`; otherwise recorded as `possible_part_of:<id>` |

An object is `accepted` only when all five are supported and the class is useful, which is what `output-contract.md` requires. Model scores are raw detector outputs: a threshold on one is a declared rule, never a probability of correctness, and no probability field is emitted anywhere. Agreement across views is one checkpoint looking twice — evidence that a detection reproduces, not independent confirmation — and the reason codes distinguish `reproduced_in_N_views` from `single_view_schedule`.

[`examples/visdrone/risk_coverage.py`](examples/visdrone/risk_coverage.py) sweeps the acceptance threshold and reports error against retained useful coverage at every operating point, including the all-abstention point where precision is undefined rather than perfect. It re-applies rules to evidence already recorded, so it costs no inference.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Completed; result written to stdout |
| 2 | Invalid input, invalid annotation, or unsafe artifact reference |
| 3 | Result produced but the search budget was exhausted (partial) |
| 4 | Execution failed or was interrupted; partial artifacts retained |
| 5 | Bundle preflight failed (missing, altered, or incompatible local assets) |

### Staging a model bundle

[`configs/bundle.example.json`](configs/bundle.example.json) is a template, not a usable bundle: its `assets` list is empty, so preflight rejects it with `missing_model_assets`. A usable bundle lists every file under `model_dir` with its SHA-256, and declares the `torch` and `transformers` versions the adapter will import. Preflight compares those declarations against both the installed distribution metadata and the module that actually imports, because the two can disagree. Nothing is downloaded at runtime.

## VisDrone example

[`examples/visdrone/`](examples/visdrone/) evaluates runs against VisDrone2019-DET val under a frozen mapping declared before predictions are read. `mapping.json` fixes the image selection, the category mapping, the excluded categories, and the **declared category scope** — the ontology classes the reference exhaustively annotates. VisDrone labels no buildings, trees or street furniture, so a correct prediction of such a class is unscoreable rather than a false positive; predictions outside the scope leave precision denominators and are counted separately. `unknown_object` and `entity` never leave a denominator, so abstention cannot buy precision.

```sh
python examples/visdrone/run_panel.py --bundle BUNDLE --out RUNS_DIR --seed 1
python examples/visdrone/build_benchmark.py --runs RUNS_DIR --output benchmark.json
lynceus evaluate --benchmark benchmark.json
python examples/visdrone/ablation.py --runs baseline=DIR_A --runs tiled=DIR_B
python examples/visdrone/render_comparison.py --runs baseline=DIR_A --runs tiled=DIR_B --out viz/
```

Aerial drone imagery is outside assumption A4, and ten images cannot support any gate. These numbers are descriptive only.

### Building an exhaustive reference panel

VisDrone annotates only road users, which bounds three things at once: precision is a lower bound of unknown tightness, calibration cannot be fitted because an unmatched box is not a scoreable error, and granularity conformance has nothing to score against. More finite-category data relieves none of them; one small exhaustively annotated panel relieves all three.

[`lynceus.reference`](src/lynceus/reference.py) and [`examples/visdrone/prepare_reference.py`](examples/visdrone/prepare_reference.py) make that panel's protocol checkable rather than promised. `prepare` writes one source-resolution review crop per tile and a skeleton to fill; `ingest` refuses anything that would make the panel quietly worthless.

```sh
python examples/visdrone/prepare_reference.py prepare --out panel/ --seed 1
open panel/index.html          # annotate each image, save reference.json into its folder
python examples/visdrone/prepare_reference.py ingest --out panel/ --benchmark exhaustive.json
lynceus evaluate --benchmark exhaustive.json
```

`prepare` writes a local review page per image beside a copy of that image. It opens over `file://` with no server, no network and no external dependency, so the panel is built under the same offline conditions the annotator requires. It walks the planned tiles, zooming each to fill the viewport so small entities are visible at source resolution, records which tiles were reviewed, and saves the `reference.json` that `ingest` checks.

The page shows no prediction and cannot load a run directory. A reference built by reviewing predictions measures agreement rather than independence, and the anchoring does not show up in the finished file.

Five obligations are enforced on ingest, because each fails invisibly otherwise: every planned review tile must be recorded as visited (an unvisited tile looks like empty space), granularity and scene layer must be tagged (an untagged part looks like an instance), ambiguity must be marked `unresolved` rather than decided (those instances stay in recall denominators and leave precision ones), leakage identity must be carried, and the panel must declare it was annotated blind. The default selection is the held-out panel already measured, so annotating it converts the existing precision bound into a number instead of producing an incomparable second result.

An exhaustive panel declares no category scope — there is nothing outside it to exclude — so precision becomes a measurement rather than a bracket.

### Two diagnostics that survive a finite-category reference

VisDrone annotates only road users, so an unmatched box is not evidence of a false box: precision is a lower bound of unknown tightness, and any change that adds boxes cannot be judged against it. Two questions escape that, and [`lynceus.diagnostics`](src/lynceus/diagnostics.py) answers them.

**Miss attribution** partitions the misses, which the reference fully determines — a resolvable reference instance either was localized or was not, and no assumption about unannotated objects enters. Each miss is assigned to the mechanism that lost it, and only one bucket is a case for another detector:

| Bucket | What it means | Repair |
|---|---|---|
| `merged_with_neighbour` | a retained box covers it, but the assignment gave that box to another reference | instance separation |
| `localized_below_match` | something reached it; no box cleared the criterion | boundary refinement |
| `below_threshold` | proposed, but every proposal scored under the shipped threshold | ranking; free |
| `merged_away` | a scoring proposal was absorbed into a cluster that does not cover it | reconciliation |
| `never_proposed` | nothing reached it at any score in any view | **another route** |

The probe run is the same schedule at a lower threshold. The adapter applies the score threshold inside `predict`, so sub-threshold proposals never reach the candidate store and running again lower is the only way to see them.

**Boundary headroom** bounds what reconciliation can return before a reconciliation rule is written: for each missed reference, the best IoU any recorded observation reached, at any score, in any view. A miss whose cluster already holds a clearing box is a selection failure; a miss where no view produced one needs geometry, and no selection rule returns it.

**Selection separability** then asks whether anything the pipeline actually has — detector score, view magnification, truncation flag — would pick that clearing box. A ceiling is stated in terms of the reference, which inference cannot see. A signal separating no case cannot be used, and a rule written anyway encodes the reference rather than deriving it.

**Pruner selectivity** applies to anything that removes retained boxes. What it removes among matched boxes is exactly measurable; what it removes among unmatched boxes is not. That asymmetry suffices, because a random pruner removes both at the same rate, so the gap is the signal. The detector's own score gives the baseline curve an external pruner has to beat.

```sh
python examples/visdrone/run_panel.py --bundle BUNDLE --out runs/shipped --seed 1 --threshold 0.1
python examples/visdrone/run_panel.py --bundle BUNDLE --out runs/probe   --seed 1 --threshold 0.005
python examples/visdrone/miss_attribution.py --shipped runs/shipped --probe runs/probe --out attribution.json
```

`--scores FILE` swaps the detector score for an external pruner's, keyed `"<image stem>/<candidate id>"`, so a candidate pruner is measured against the same curve rather than on its own terms. Boxes the external scorer cannot score leave both curves, so the pruner and the baseline it must beat are compared on exactly the same boxes.

[`examples/visdrone/ttn_scores.py`](examples/visdrone/ttn_scores.py) is one such scorer, and shows what the interface expects. It needs a local clone of the [TTN repository](https://github.com/voxel51/ttn) and its released weights — imported from a path rather than vendored, because that repository states no licence, which blocks redistribution though not measurement. TTN is an in-context comparator, so it also needs a per-class bank of ground-truth reference patches; that bank comes from the VisDrone train split under A3, with the 24 sequences shared between train and val excluded, one of which is in the held-out panel. Measured result: it does not beat the detector score, in any size bin, with either released variant.

Both are descriptive. Selectivity shows that a scorer discriminates, never that the surviving set became correct.

### Refining boxes against pixels

[`examples/visdrone/sam_refine.py`](examples/visdrone/sam_refine.py) prompts a box-prompted SAM with each retained box and replaces it with the box of the mask SAM rates highest. No box is added and none removed, so recall can only move through geometry — which is the claim under test. [`examples/visdrone/refine_effect.py`](examples/visdrone/refine_effect.py) then reports what moved, counting references recovered and lost rather than mean IoU, because that is what the recall gate counts.

```sh
python examples/visdrone/sam_refine.py --runs runs/shipped --crop-margin 1.0 --out refined.json
python examples/visdrone/refine_effect.py --runs runs/shipped --refined refined.json
```

`--crop-margin` refines each box inside its own source-resolution crop, which is what the architecture asks for; `--whole-image` refines against one pass over the full view instead. The difference matters on small objects: a full-view pass resizes the image to 1024 on its longest side, so a twelve-pixel object arrives at about six pixels. Both modes are kept so the comparison is a measurement rather than an assertion.

## Maintaining the contract

BMAD's local [bmad-spec skill](_bmad/skills/bmad-spec/SKILL.md) and shared helpers are included so the contract can be maintained. This is a selected workflow bundle from an existing BMAD 6.11.0 installation, not a full BMAD installation. See [workflow provenance](_bmad/provenance.json).

Specification updates go through `bmad-spec`: append decisions to the spec's `.memlog.md`, rederive the contract, and repeat coherence and preservation checks.
