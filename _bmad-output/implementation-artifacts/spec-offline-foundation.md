---
title: Runnable offline annotation foundation
type: feature
created: 2026-09-19
status: in-review
baseline_commit: NO_VCS
review_loop_iteration: 2
context:
  - '{project-root}/_bmad-output/specs/spec-lynceus/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-lynceus/annotation-policy.md'
  - '{project-root}/_bmad-output/specs/spec-lynceus/output-contract.md'
  - '{project-root}/_bmad-output/specs/spec-lynceus/evaluation.md'
---

<frozen-after-approval reason="Derived implementation slice authorized by the user's lets go after the delivery specification">

## Intent

**Problem:** Lynceus has a delivery specification but no runnable foundation. There are no cached weights or independent reference images here, and CUDA is unavailable, so real-model feasibility and reference qualification cannot be claimed yet.

**Approach:** Build a tested local CLI/library that enforces the annotation policy, validates outputs, preflights a staged model bundle, normalizes images, runs a conservative OWLv2 baseline when weights exist, and evaluates independent recorded predictions. Preserve unfinished model/data qualification explicitly.

## Boundaries & Constraints

**Always:** Read the product contract; local-only runtime; no target learning; original-coordinate boxes; uncertainty and missing runs remain visible; qualify nothing without independent evidence. Use Python 3.11+, installed Pydantic 2, Pillow, NumPy/SciPy and pytest. Pin installed versions used by this slice.

**Ask First:** Only missing deployment resources or decisions that actually prevent further work; the existing user authorization covers implementation and local tests.

**Never:** Fabricate detections, silently download weights, treat mock outputs as model validation, equate baseline completion with saturation, or modify the canonical product SPEC outside bmad-spec.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Offline preflight | Local manifest and assets | Hash/capability/environment report | Typed missing/incompatible assets; no download |
| Annotation | JPEG/PNG and valid OWLv2 bundle | Unqualified baseline result with conservative uncertain proposals and evidence | No accepted accuracy claim from scores |
| Geometry | Rotations, crop edges, invalid boxes | Source-coordinate valid extents | Reject NaN, inversion, bounds violations |
| Policy | Parts, clothing, reflections, unknowns | Declared kinds/layers and exclusions | Preserve unresolved interpretations |
| Evaluation | Frozen independent references and predictions | One-to-one metrics with explicit denominators | Missing/failed images remain misses; duplicate predictions penalized |
| Artifact validation | Tampering, escaping refs, impossible state | Explicit validation failure | No silent repair or unsafe reuse |

</frozen-after-approval>

## Code Map

- `_bmad-output/specs/spec-lynceus/` — product contract; application code is derived from it and never edits it.
- `src/lynceus/` — new package; CLI, policy, contracts, images, artifacts, model adapters, and evaluator.
- `tests/` — synthetic contract/algorithm tests, not quality qualification.
- `configs/`, `examples/`, `docs/` — staged-bundle example, benchmark fixture, measured feasibility and usage.
- `_bmad/` — workflow tooling; source copies retain hashes and MIT attribution.
- No usable Git repository is available. Preserve a file/hash inventory for review instead of inventing a revision.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml`, `src/lynceus/__init__.py`, `__main__.py`, `cli.py` — package and commands: doctor, policy, schema, validate, annotate, resume, evaluate.
- [x] `src/lynceus/policy.py`, `data/*.json`, `contracts.py`, `tests/test_policy.py`, `tests/test_contracts.py` — versioned policy/ontology, native schema and cross-field invariants.
- [x] `src/lynceus/evaluation.py`, `tests/test_evaluation.py`, `examples/benchmark.json` — evaluator, split leakage checks, image-cluster uncertainty and inconclusive qualification.
- [x] `src/lynceus/images.py`, `geometry.py`, `artifacts.py`, `bundle.py`, `adapters/owlv2.py`, `pipeline.py` — source transforms, local-only adapter, immutable artifacts and safe resume.
- [x] `tests/test_images.py`, `test_bundle.py`, `test_pipeline.py`, `test_cli.py` — integration and matrix cases with deterministic fake adapters explicitly confined to tests.
- [x] `README.md`, `docs/feasibility.md`, `docs/delivery-status.md` — runnable commands, actual hardware/resources, remaining work and verified scope.

**Acceptance Criteria:**
- Given a network-denied process, when doctor/fixture evaluation/validation execute, then none attempts a network connection and errors have stable nonzero exit codes.
- Given known duplicates, unknown labels, groups and missing runs, when evaluated, then correct metric numerators/denominators match manually determined outcomes.
- Given invalid uncertainty, relations, artifacts or search claims, when validated, then the result is rejected.
- Given no staged model weights, when real annotation is requested, then it fails clearly without fake boxes; fake adapters demonstrate orchestration only in tests.
- Given a completed or interrupted run, when resumed, then dependency/artifact mismatches cannot be silently reused.

## Spec Change Log

Review iteration 1, 2026-09-19. Six contract deviations found by reviewing the implementation against `output-contract.md`, `annotation-policy.md` and memlog decision D15, all fixed in place with regression tests. None changes the product contract, so no `bmad-spec` rederivation is required.

1. **Ingestion (D15, WP-04).** `normalize_image` accepted multi-frame PNG and silently used the first frame, and applied no source-size cap. It now raises typed `multi_frame_image`, `excessive_image_size` and `malformed_image` errors. `MAX_SOURCE_PIXELS` is set below Pillow's decompression-bomb warning threshold so the typed rejection fires first.
2. **Dependency verification (WP-02 acceptance).** `preflight` and `doctor` read versions from `importlib.metadata` only. On this host that metadata disagrees with the module that actually imports for `transformers` (4.49.0 vs 5.3.0) and `numpy` (2.2.6 vs 1.26.4), so preflight passed a bundle whose runtime was incompatible and the ad-hoc guard in the adapter fired only after the run directory and manifest had been written. `check_dependencies` now compares the manifest against both metadata and the imported module, `doctor` reports `runtime_versions` and `version_conflicts`, and `Owlv2Adapter` requires the bundle to declare every package it imports. Local build suffixes such as `+cu124` are compared on the public version.
3. **OWLv2 coordinates (CAP-4, annotation-policy geometry).** `Owlv2ImageProcessor.pad` squares the image with bottom/right padding of side `max(height, width)`, so predicted boxes are normalized to that square. The adapter unscaled with the source height/width, which shrinks every box on a non-square image; clipping hid the symptom rather than correcting the scale. It now unscales with the padded side and then clips. `docs/feasibility.md` had already recorded the padding behaviour, so this was a known hazard left unhandled in code. Unverified against trained weights.
4. **Overlay artifact (output-contract artifact layout).** `overlay.png` was the normalized source image re-encoded, which the contract forbids ("never a source image"). It now renders boxes, object identities and an unqualified banner.
5. **Summary and events (output-contract required fields, WP-05).** `summary` carried neither coverage metrics nor a resource summary, and `events.jsonl` carried no resource usage, although both are required. `summarize` now takes coverage and resource arguments, `validate_annotation` requires `coverage` and `resources` with their named fields while still recomputing every count it can, and each event records wall time, peak RSS and model calls. `summary_counts` gained `unsearched_regions`.
6. **Exit codes (WP-05 deliverable).** Documented in `README.md` together with the command reference and bundle-staging rules.

### Iteration 2, 2026-09-19 — tiled discovery (partial WP-06)

Requested during the VisDrone run: a single full-image pass downscales 1360x765 to the model's 1008px input, so VisDrone's 12-44px objects lose most of their resolution before inference. Delivered the tile planner, source-resolution tiling, cross-view merging and the route coverage manifest from WP-06. Not delivered: class-independent proposals, semantic inventory, concept/exemplar routes. WP-06 is therefore partial, not complete.

- `geometry.plan_tiles` produces a full frame plus, for each level *j* up to `tile_levels`, a (*j*+1)x(*j*+1) grid enlarged by `tile_overlap`. The plan depends only on image dimensions and frozen parameters, never on image content, so it is not target adaptation. `geometry.covered_fraction` measures the exact pixel union of completed views for the coverage manifest.
- `pipeline.discover` crops each view at source resolution and maps observations back by translation. A failed view is recorded, the scan continues, and the failure becomes an `unsearched` region that forbids `profile_complete`.
- `pipeline.merge_observations` merges repeated views highest-score-first using intersection over the smaller box rather than IoU, since a tile-truncated detection overlaps its whole-object counterpart with low IoU but high IoS. This is SAHI's matching rule; the SAHI library itself was not adopted because the bundle manifest must pin and hash every runtime dependency, its HuggingFace model type does not cover OWLv2's text-prompt path, and it does not emit the per-candidate lineage `output-contract.md` requires. Two deliberate deviations from SAHI: only same-label observations merge, so a backpack inside a person's box is never suppressed by geometry alone (WP-07); and the winning box is kept rather than a union, because the annotation policy forbids asserting extent no single view supported. Merged boxes become `alternative_boxes` and set `boundary: unresolved`.
- `Owlv2Adapter` now loads the checkpoint once per run instead of once per call, selects CUDA when available, and exposes `capabilities()` for the adapter capability record WP-02 asks for.
- Profile ids are `owlv2-single-pass-v1` for `tile_levels=0` and `owlv2-tiled-Lk-v1` otherwise. Tiling never licenses `saturated`; `validate_annotation` still rejects it for every profile.
- `evaluation.evaluate_benchmark` accepts an optional `category_scope` with a mandatory reason. It implements the "preregistered compatible subset/mapping is documented" branch of WP-01 for finite-category references: predictions labelled outside the scope leave precision denominators and are counted, while `unknown_object` and `entity` never leave a denominator so abstention cannot satisfy gate G6. Absent the field the evaluator behaves exactly as before.
- `examples/visdrone/` holds the frozen mapping, the benchmark converter, the ablation comparison and the reference-vs-prediction renderer.

Empirical confirmation of iteration 1 finding 3: on a 1360x765 VisDrone frame the fixed padded-side unscaling matched 31 of 36 eligible references at IoU 0.5, while the previous source-size unscaling matched 2. Every VisDrone frame is non-square, so the bug was catastrophic there and invisible on square input.

Deliberately not changed: `resume` re-runs discovery rather than reusing completed views. The baseline profile has one stage, so there is nothing to reuse; stage-level reuse belongs with the multi-route schedule in WP-06 and WP-10.

## Design Notes

Shared API: `policy.load_policy()`/`load_ontology()` return JSON dictionaries; `contracts.validate_annotation(payload, artifact_root=None)` returns normalized JSON or raises ValueError; `contracts.annotation_schema()` returns JSON Schema. `evaluation.evaluate_benchmark(payload, bootstrap_replicates=10000, seed=0)` returns JSON metrics. Module imports must have no downloads or GPU allocation. CLI performs structured error reporting.

The implementation subagent owns application code, tests, examples and README. The lead handles independent hardware/API research, `docs/feasibility.md`, `docs/delivery-status.md`, and workflow/review records. Extend the starting API only with an explicit integration message. All model predictions remain uncertain until independent verification is implemented; classify this as an object-discovery baseline, not a complete reference annotator.

## Verification

- `python3 -m pytest` — 51 passed: contract, policy, geometry, evaluator and offline integration cases, including the iteration-1 regression cases for ingestion rejection, runtime dependency divergence, overlay content, report content and resource recording.
- `PYTHONPATH=src python3 -m lynceus doctor --offline` — reports actual environment without loading/downloading models.
- `PYTHONPATH=src python3 -m lynceus evaluate --benchmark examples/benchmark.json` — reproducible fixture metrics, explicitly unqualified.
- `PYTHONPATH=src python3 -m lynceus schema` — valid native JSON Schema.
