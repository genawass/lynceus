# Delivery status

This status distinguishes implemented software from model-quality qualification. The canonical delivery requirements remain in the [BMAD specification](../_bmad-output/specs/spec-lynceus/SPEC.md).

## Pinned inference library: transformers 5.17.0

The pin moved from 4.49.0 to 5.17.0 so that boundary refinement runs as a stage in one process
rather than as a second dependency set. The bump changes the detector's geometry -- on one panel
image with an identical normalized-pixel hash, the two versions retained 394 and 382 objects with
no box identical between them -- so every measurement in this report was re-run on the new pin
before being cited. Figures taken on 4.49.0 are preserved under `out/superseded-transformers-4.49.0/`
and are not current. The re-measurement is recorded below under "Re-measured on the new pin".

## Active slice

[Runnable offline annotation foundation](../_bmad-output/implementation-artifacts/spec-offline-foundation.md): policy/ontology, schema and invariants, image normalization, local bundle checks, conservative OWLv2 baseline, artifact persistence, and independent-prediction evaluation. Implemented and reviewed against the contract (review iteration 1). Verification evidence is recorded below.

The next proposal-model experiment is frozen in [WeDetect and OWLv2 model selection](model-selection.md): WeDetect-Uni / WeDetect-Anything is the preferred class-independent proposal candidate, while OWLv2 remains the prompted baseline and alternate objectness source. Selection requires the same images, tile schedule, caps, evaluator, and downstream verification. Published AP values are not used as a cross-model comparison.

## Product work-package scope

| Work package | Scope in this slice | What still requires later evidence/work |
|---|---|---|
| WP-01 policy/reference fixtures | Implementable default policy, versioned ontology, policy test cases | Freeze the actual qualification population/conventions with compatible external data |
| WP-02 local model bundle | Local-only loader/preflight verifying declared dependencies against both metadata and the imported module, environment inventory with conflict reporting, model API smoke | Stage pretrained assets; measure actual model memory, latency, and required adapter capabilities; confirm the OWLv2 padded-coordinate correction against trained weights |
| WP-03 independent evaluation | Matching, denominator, split-leakage and uncertainty machinery with synthetic fixtures | Licensed independent benchmark inventory, fully populated strata, locked ablations and statistical qualification |
| WP-04 input/result/evidence | Validation and source-coordinate normalization foundation, typed rejection of multi-frame/oversized/malformed input, required coverage and resource summary | Expand cases as segmentation, relations, masks, and calibrated events are introduced |
| WP-05 offline baseline | Baseline orchestration and resume contract, executed end to end on staged OWLv2-large weights | Per-view cache keys for resume (memlog D11); deployment-hardware resource profile beyond the single RTX 3060 measured here |
| WP-06 multiscale discovery | **Partial.** Frozen full-frame plus overlapping source-resolution tile planner, cross-view merging on intersection-over-smaller, union evidence store with per-candidate lineage, route coverage manifest | Class-independent proposals, semantic inventory, concept and exemplar routes; locked ablation on an adequate panel |
| WP-08 conservative naming | **Partial.** Per-class score distributions from the adapter, relative runner-up comparison, broader-supported-class fallback, `unknown_object` when no useful ancestor exists, alternatives retained | Crop and context re-interpretation, synonym normalisation, prompt-anchoring controls |
| WP-09 evidence assessment | **Partial.** Per-dimension assessment under frozen rules, explicit acceptance, risk-versus-coverage sweep | Externally calibrated event estimates, ancestry metadata, independent verification |
| WP-07, WP-10 | Not delivered | Instance graph and boundary refinement, omission audits, saturation |
| WP-11–13 | Not delivered in this foundation slice | Training ignore-mask integration, human-label audit, detector pseudo-reference comparison |
| WP-14–15 | Not delivered as a qualified release | Reproducible full model bundle, clean deployment validation, locked independent qualification |

The tiling slice is partial WP-06: the geometry, merging and coverage machinery exist, the complementary discovery routes do not. Nothing in this report marks all of M0 or M1 complete. Synthetic cases verify software behavior; they do not establish detection quality. A completed baseline must remain unqualified and cannot claim an exhaustive annotation.

## External resources staged since the first report

| Resource | State |
|---|---|
| Model weights | `google/owlv2-large-patch14-ensemble` staged at `/home/genadiy/data/models/owlv2-large-patch14-ensemble`, 8 assets hash-manifested in `bundle-owlv2-large.json`, preflight passing |
| GPU | RTX 3060 12GB, CUDA available. The earlier feasibility report recorded no usable GPU; that observation is superseded |
| Runtime environment | `.venv` with `--system-site-packages` plus a real `transformers==4.49.0` and `numpy==2.2.6`, because the ambient conda environment carries stale `.dist-info` directories that make metadata disagree with the imported module. `doctor` reports `version_conflicts: []` inside the venv |
| Reference data | VisDrone2019-DET val at `/home/genadiy/data/VisDroneMerged`, 548 images with original-format annotations. Frozen 10-image subset and mapping in `examples/visdrone/mapping.json` |

VisDrone is aerial imagery, outside assumption A4, and ten images cannot support any gate in `evaluation.md`. It exercises the software and gives a descriptive baseline-versus-tiled comparison; it qualifies nothing.

## External resources staged for the pruning and refinement experiments

| Resource | State |
|---|---|
| CLIP ViT-L-14 | `openai/clip-vit-large-patch14` staged at `/home/genadiy/data/models/clip-vit-large-patch14`; supported by the pinned `transformers` 4.49.0 |
| TTN weights | Both released variants (`ttn`, `ttnd`) downloaded from the repository's Dropbox links. No hashes are published and the repository states no licence, so they are staged for measurement only and are not bundleable under WP-02/WP-14 |
| SAM 1 | `facebook/sam-vit-huge`, ungated, supported by the pinned `transformers` via `SamModel` |
| SAM 3 | **Blocked.** `facebook/sam3` is gated `manual` on Hugging Face and returns 401 without an approved access request; no token is configured on this host. The pinned `transformers` 4.49.0 also has no SAM 2 or SAM 3 support, so adopting it is a dependency bump to be staged as its own change (D32) |

## External resources still pending

No compatible general-photograph benchmark with exhaustive open-world annotation has been supplied. Resource discovery and its limitations are documented in [feasibility.md](feasibility.md). The code can progress without these resources, but pretrained inference and reference qualification cannot be honestly reported as passed.

## Verification evidence

Run from a clean checkout with `PYTHONPATH=src`. Network sockets are denied inside the suite by an autouse fixture.

| Check | Result |
|---|---|
| `python -m pytest` | 116 passed |
| `lynceus doctor --offline` | Exit 0; reports this host's `numpy` metadata/runtime conflict and loads no model |
| `lynceus evaluate --benchmark examples/benchmark.json` | Exit 0; `qualification.status = inconclusive` |
| `lynceus schema` | Exit 0; valid `lynceus.annotation/1.0` JSON Schema |
| `lynceus annotate <image> --bundle configs/bundle.example.json` | Exit 5 `bundle_error`; no run directory created, no fabricated boxes |

## Review findings closed in iteration 1

| Finding | Contract reference | Resolution |
|---|---|---|
| Multi-frame PNG accepted; first frame chosen silently; no source-size cap | memlog D15, WP-04 | `normalize_image` rejects multi-frame, oversized, and malformed input with typed errors |
| `preflight` and `doctor` trusted `importlib.metadata`, which disagrees with the imported module on this host | WP-02 acceptance | Both versions reported and compared; a declared dependency whose runtime version differs is rejected before annotation |
| OWLv2 boxes were unscaled with the source size although the processor pads to a square | annotation-policy geometry, CAP-4 | Unscale with the padded side, then clip; still unverified against trained weights |
| `overlay.png` was the re-encoded source image | output-contract artifact layout | Boxes, identities, and an unqualified banner are drawn |
| `summary` carried no coverage or resource metrics; events carried no resource usage | output-contract required fields, WP-05 | Both recorded and required by the validator |
| Exit codes were undocumented | WP-05 deliverable | Documented in the README |

These are software-contract fixes. None of them is evidence of detection quality.

## Measured: VisDrone2019-DET val, 10 images, 827 eligible references

Frozen mapping in `examples/visdrone/mapping.json`, declared before predictions were read. OWLv2-large-patch14-ensemble on one RTX 3060. Class-neutral one-to-one matching at IoU 0.5.

| | views/img | recall | prec_all | prec_scoped | cls_err | loc_err | predictions | wall |
|---|---|---|---|---|---|---|---|---|
| `--tile-levels 0` (baseline) | 1 | 0.578 | 0.394 | 0.534 | 0.182 | 0.225 | 1214 | 287 s |
| square grid, IoS-only merge | 14 | 0.768 | 0.230 | 0.293 | 0.153 | 0.238 | 2758 | 548 s |
| `--tile-levels 1` aspect grid, edge-conditional merge | **7** | **0.807** | 0.215 | 0.267 | 0.163 | 0.238 | 3098 | 382 s |

The final configuration reaches higher recall than the 14-view square-grid run on **half the views and 70% of the wall time**, recovering 189 of the 349 baseline misses (54%). Two changes account for it: grids follow the image aspect, so a 16:9 frame gets 3x2 tiles at 1.85x magnification with 19.5% padding waste instead of 3x3 at the same magnification with 43.7% waste; and merging trusts intersection-over-smaller only when an observation reaches its view border, which stopped IoS from collapsing crowd neighbours. Measured offline on 11,553 observations, IoS-only merging destroyed 13% of correct detections that way.

Precision fell slightly because the conditional rule retains more boxes. That is the intended direction: WP-07 forbids suppressing a valid separate instance on geometry alone, and controlling precision is the job of the acceptance stage (WP-09), which does not exist yet. `prec_all` charges every retained instance; `prec_scoped` counts only predictions inside the declared category scope. `cls_err` is the share of matched pairs whose label is not permitted for that reference; `loc_err` is mean (1 - IoU) over matched pairs.

Accepted annotation precision is 0/0 and useful-label coverage is 0/827 for both profiles: no verification stage exists (WP-09), so every object is `uncertain` and nothing can be `accepted`. Gates G1 and G4 are unmeasurable at this stage rather than failed.

The dominant naming error is VisDrone `motor` labelled `bicycle`, 45 of 49 matched motorcycles. The ontology has no motorcycle class and the mapping permits only the broader `vehicle`, so the correct output is `vehicle`. Choosing a supported broader class instead of an unsupported specific one is WP-08, which is not implemented.

None of this qualifies anything. Aerial imagery is outside assumption A4, ten images are far below the 500-image and 5000-instance panel every gate in `evaluation.md` requires, and the evaluator reports `inconclusive` for all of them. The precision drop under tiling is exactly the failure mode gate G6 exists to catch.

## Measured: domain ontology and acceptance

Two changes were measured on the frozen 10-image panel (seed 0, already inspected) and on a
held-out panel never looked at before the runs (seed 1, 717 eligible references).

| panel | ontology | recall | prec_all | prec_scoped | cls_err | loc_err |
|---|---|---|---|---|---|---|
| seed 0 | lynceus-broad-v1 | 0.807 | 0.215 | 0.267 | 0.163 | 0.238 |
| seed 0 | aerial-traffic-v1 | 0.815 | 0.223 | 0.241 | **0.215** | 0.237 |
| seed 1 (held out) | lynceus-broad-v1 | 0.884 | 0.224 | 0.274 | 0.205 | 0.230 |
| seed 1 (held out) | aerial-traffic-v1 | 0.893 | 0.225 | 0.240 | **0.195** | 0.229 |

The domain ontology did **not** deliver the predicted drop in classification error. It worked
exactly as intended for the classes the broad ontology could not express, and introduced a new
confusion of comparable size among the classes it added.

| reference category | broad | aerial | |
|---|---|---|---|
| motor (seed 1) | 8/95 | 39/101 | the gap the change targeted |
| tricycle (seed 1) | 11/22 | 18/22 | same |
| car (seed 0) | 282/282 | 251/284 | new car/van confusion |
| van (seed 0) | 39/39 | 21/40 | same |

Under the broad ontology both `car` and `van` references permitted `{car, vehicle}`, so calling
every four-wheeler a car was always correct. Adding `van` and `truck` created a distinction the
detector cannot make reliably at aerial resolution, and the loss there cancels the motorcycle gain.

Adding classes to an ontology is therefore not free: it fills gaps and creates competitive
confusions at the same time. The correct remedy is the conservative naming of WP-08 — emit the
supported broader class `vehicle` when the specific one is not distinguishable — which would
satisfy both the car and van references. WP-08 is now the blocking piece rather than an
improvement, and the ontology change alone should not be treated as a win.

A second correction: `motor` labelled `bicycle` remains the single largest error even with
`motorcycle` available as a prompt (35 of 65 on seed 0, 44 of 101 on seed 1). The earlier reading
of this as purely an ontology gap was wrong; it is substantially a detector limitation at this
resolution.

### Detection threshold

| threshold | recall | predictions | prec_all |
|---|---|---|---|
| 0.10 | 0.815 | 3019 | 0.223 |
| 0.05 | 0.845 | 5852 | 0.119 |

Halving the threshold buys 0.030 recall for 94% more retained instances and halves precision. Not
a good operating point; the 120 references no view proposes are not mostly threshold-limited.

### Risk versus coverage (WP-09, held-out panel)

| acceptance score | accepted | correct | precision | useful coverage |
|---|---|---|---|---|
| 0.10 | 154 | 92 | 0.597 | 0.128 |
| 0.30 | 79 | 65 | 0.823 | 0.091 |
| 0.50 | 11 | 10 | 0.909 | 0.014 |
| 0.60 | 2 | 2 | 1.000 | 0.003 |
| 0.80+ | 0 | 0 | undefined | 0.000 |

Gate G1 proposes an accepted-precision lower bound of 0.99 and G4 a useful-label coverage of 0.80.
These are nowhere near jointly satisfiable here: coverage has already collapsed to 0.003 by the
point precision approaches 1.0, and that point rests on two accepted objects, which `evaluation.md`
explicitly refuses to treat as establishing a certainty bound. The acceptance stage now exists and
is measurable; it is far from passing.

## Measured: conservative naming (WP-08)

Same tiled configuration and aerial ontology, with the broader-class fallback enabled at the
declared ratio of 0.6. The ratio was fixed before these runs; the sweep below is sensitivity
analysis, not selection.

| panel | naming | recall | prec_all | cls_err | loc_err |
|---|---|---|---|---|---|
| seed 1 (held out) | off | 0.893 | 0.225 | 0.195 | 0.229 |
| seed 1 (held out) | **on** | 0.901 | 0.217 | **0.127** | 0.228 |
| seed 0 | off | 0.815 | 0.223 | 0.215 | 0.237 |
| seed 0 | **on** | 0.825 | 0.220 | **0.133** | 0.236 |

Classification error fell 35% on the held-out panel and 38% on seed 0, with recall slightly up and
precision essentially unchanged. Recall rose because naming runs before merging, so two views that
disagree on the specific class but agree on the broader one now describe one instance instead of
splitting into two.

The mechanism is visible per category on the held-out panel:

| reference category | naming off | naming on |
|---|---|---|
| car | 250/265 | 266/266 |
| van | 12/19 | 19/19 |
| truck | 9/17 | 16/16 |
| motor | 39/101 | 75/101 |
| pedestrian | 106/122 | 99/122 |
| people | 51/60 | 36/65 |

Both confusions that the ontology change could not fix are resolved: the car/van/truck distinction
the ontology introduced, and the motorcycle/bicycle confusion that survived it. Both now emit the
supported broader class `vehicle`, which the mapping permits for every vehicle category.

The cost falls on people. `person` has no useful ancestor in this ontology, so when it competes
with `bicycle` — a rider, genuinely ambiguous at this resolution — the fallback emits
`unknown_object` rather than guess. That is the policy's intended behaviour, and it is why the
`unknown_object` rate rose from 0.000 to 0.065.

The larger cost is specificity: at the declared ratio, 47% of retained instances are labelled
`vehicle` rather than a specific vehicle class. The annotation became more correct and less
informative at the same time. `evaluation.md` already separates these, since generic and unknown
labels do not count toward useful-label coverage while `vehicle` does.

### Ratio sensitivity (held-out panel, declared point 0.6)

| ratio | cls_err | unknown rate | `vehicle` share | specific share |
|---|---|---|---|---|
| 1.0 (disabled) | 0.193 | 0.000 | 0.023 | 0.977 |
| 0.8 | 0.135 | 0.029 | 0.199 | 0.772 |
| 0.7 | 0.133 | 0.044 | 0.331 | 0.625 |
| **0.6 (declared)** | **0.127** | 0.065 | 0.474 | 0.460 |
| 0.5 | 0.124 | 0.081 | 0.594 | 0.324 |
| 0.4 | 0.142 | 0.102 | 0.659 | 0.239 |
| 0.3 | 0.166 | 0.131 | 0.707 | 0.161 |

Error is lowest near the declared point and rises again below 0.4, where abstention itself becomes
the dominant error. The declared value landing near the minimum is fortunate, not evidence: it was
chosen before measurement and must not be re-selected from this curve for a qualification claim.

## Measured: restricting the ontology to relevant classes

`aerial-traffic-strict-v1` keeps the nine road-user classes VisDrone annotates and drops `building`
and `tree`, which the previous ontology carried because the annotation policy annotates every
distinguishable entity. For a traffic-monitoring deployment those are outside the declared
convention, and the ontology records that in `excluded_by_convention` so their absence reads as a
convention boundary rather than a measured negative.

| panel | ontology | predictions | recall | prec_all | prec_scoped | cls_err |
|---|---|---|---|---|---|---|
| seed 1 (held out) | aerial (11 classes) | 2981 | 0.901 | 0.217 | 0.230 | 0.127 |
| seed 1 (held out) | strict (9 classes) | 2808 | 0.901 | **0.230** | 0.230 | 0.127 |
| seed 0 | aerial (11 classes) | 3100 | 0.825 | 0.220 | 0.237 | 0.133 |
| seed 0 | strict (9 classes) | 2876 | 0.825 | **0.237** | 0.237 | 0.133 |

Recall, classification error and localization error are unchanged to three decimals: the removed
boxes never matched a reference. Precision improves because the denominator shrinks, and scoped
and unscoped precision are now identical, so precision is a single interpretable number rather
than a bracket.

Removing a class does not silently reassign its detections. On a building patch the score row is
`building 0.392, car 0.009, bus 0.007`; without `building` the maximum falls below the detection
threshold and the box stops existing. Simulated over 5468 recorded score rows before the runs:
408 observations removed, 2 relabelled.

## Measured: deduplication by box fusion — negative result

The precision decomposition showed roughly 800 retained boxes (28.5%) sitting on an instance that
another box already described. Two mechanisms were built to reach them: a score-weighted consensus
box per cluster, and a bounded re-merge to exploit the moved geometry. Both were measured on the
held-out panel and neither helped.

| configuration | boxes | matched | recall | precision | mean IoU of matches |
|---|---|---|---|---|---|
| 1 round, keep winner (shipped) | 2808 | 646 | 0.901 | 0.230 | 0.772 |
| 2 rounds, keep winner | 2808 | 646 | 0.901 | 0.230 | 0.772 |
| 3 rounds, keep winner | 2808 | 646 | 0.901 | 0.230 | 0.772 |
| 2 rounds, fuse boxes | 2745 | 628 | 0.876 | 0.229 | 0.759 |

**Fusion degrades boxes.** When a tile-truncated observation and a whole-object one describe the
same instance, the weighted mean drags the good box toward the bad one. It cost 18 of 646 matches
and moved mean IoU from 0.772 to 0.759, with no precision gain.

**Re-merging is inert.** Greedy merging is idempotent: feeding the same boxes back produces the
same groups, so rounds beyond the first change nothing unless geometry moves. The 1, 2 and 3 round
results are identical to three decimals.

Both are now off by default and retained behind flags with the measurement recorded. The
under-merge is real but is not reachable by moving boxes between observations: the redundant pairs
disagree by more than the merge threshold and are not view-edge truncated, and the looser criterion
that would capture them also collapses crowd neighbours, which was measured earlier at a 13% loss
of correct detections. Closing it requires boundaries refined against pixels, which is WP-07
proper. Score-weighted fusion is not a cheap substitute for that.

## Implemented: calibrated estimates and the applicability guard

`calibrated_estimates` was previously rejected outright as unimplemented. It is now a validated
type carrying an exactly stated event, a probability in [0,1], the calibration artifact that
produced it, the population it was fitted on, and a localization threshold.

`summary.calibration` is a new required block with a status of `applied` or `inapplicable` and a
reason. The validator enforces that no object may carry an estimate unless calibration is applied,
that the estimate references an artifact present and hash-verified in the run, and that
`probability_all_objects_found` and `is_ground_truth` remain rejected: a box may be scored, the
image's completeness may not.

A calibration whose declared domain differs from the run's ontology produces
`calibration_domain_mismatch` and no numbers, implementing the rule in `evaluation.md` that a fit
from another domain does not establish reliability. With no calibration artifact staged — the
current state for every run here — the result records
`inapplicable / no_calibration_artifact_staged` rather than an empty list a consumer could mistake
for a scored-and-zero object.

No calibration has been fitted. Fitting one requires an exhaustively annotated reference, because
on a finite-category reference an unmatched box is not evidence of a false box. That makes open
question Q4 the blocker for any probability this system emits.

## Measured: where the misses come from, and what a second detector would buy

The ensemble question — add DART, a Grounding DINO variant and WeDetect alongside OWLv2 — was
settled by measurement rather than by argument. Two diagnostics stay interpretable on VisDrone
even though it annotates only road users, and both are now implemented in
[`lynceus.diagnostics`](../src/lynceus/diagnostics.py).

Misses are fully determined by the reference: a resolvable reference instance either was localized
or it was not, and no assumption about unannotated objects enters. So each miss can be attributed
to the mechanism that lost it. The shipped run at threshold 0.1 decides what counts as a miss; a
probe run of the same schedule at 0.005 supplies the proposals the shipped threshold removed
inside the adapter. The re-run reproduced the recorded panel exactly — 2808 retained boxes, 646
matched, recall 0.901, precision 0.230 — so the attribution describes the same result reported
above.

| Bucket | Misses | Share | Repair it implies |
|---|---|---|---|
| localized_below_match | 55 | 0.775 | boundary refinement (WP-07) |
| below_threshold | 9 | 0.127 | ranking; free |
| merged_with_neighbour | 4 | 0.056 | instance separation (WP-07) |
| merged_away | 1 | 0.014 | reconciliation |
| **never_proposed** | **2** | **0.028** | **another detection route** |

**Two of 717 eligible references were never proposed at any score at or above 0.005 in any view.**
That is the entire case for adding a detector. A second route could recover at most 0.3 percentage
points of recall, against a doubling of inference cost and an immediate precision loss, since every
added box enters the retained set while only some enter the matched set. Grounding DINO variants
are deferred on this evidence (D31, D32); the recall gap to G2 is not a detector-capacity gap.

Three quarters of the misses are objects OWLv2 reached with a box that did not clear IoU 0.50. All
71 misses fall in the two smallest size bins — 38 below 16 pixels equivalent side, 26 between 16
and 32, 7 between 32 and 96 — and both `never_proposed` cases are below 16. At that scale IoU is
brutally sensitive: a one-pixel offset on a 12-pixel object moves it across the criterion, so part
of the `localized_below_match` bucket reflects the strictness of the measure rather than a
recoverable boundary. That caveat bounds how much WP-07 can return; it does not change the
conclusion, because the objects are being found either way.

`never_proposed` means not proposed at or above the 0.005 probe floor, not at literally any score.
The bucket has no cap-saturation counterpart here: the OWLv2 adapter keeps every patch token above
its threshold rather than decoding a fixed query budget, so it cannot lose an object to a cap. A
route that decodes a fixed number of queries per forward — 900 for Grounding DINO — can, and on
this panel's density that is a structural disadvantage rather than an incidental one.

### Detector score as a pruner, and the bar an external pruner must clear

A pruner removes boxes rather than adding them, so its removals among matched boxes are exactly
measurable while its removals among unmatched boxes are of unknown value. That asymmetry is enough
to tell a discriminating scorer from a random one, which removes both classes at the same rate.

Sweeping the detector's own score over the 2808 retained boxes:

| threshold | kept | matched kept | matched removed | unmatched removed | selectivity | recall | precision |
|---|---|---|---|---|---|---|---|
| 0.100 | 2808 | 646 | 0.000 | 0.000 | 0.000 | 0.901 | 0.230 |
| 0.148 | 1773 | 597 | 0.076 | 0.456 | 0.380 | 0.833 | 0.337 |
| 0.182 | 1330 | 554 | 0.142 | 0.641 | 0.499 | 0.773 | 0.417 |
| **0.195** | **1182** | **525** | **0.187** | **0.696** | **0.509** | **0.732** | **0.444** |
| 0.253 | 739 | 422 | 0.347 | 0.853 | 0.507 | 0.589 | 0.571 |
| 0.328 | 444 | 320 | 0.505 | 0.943 | 0.438 | 0.446 | 0.721 |

The score is already a competent pruner: at its best operating point it removes 70% of unmatched
boxes while losing 19% of matched ones. **Selectivity 0.509 is the bar.** An external pruner such
as TTN earns its place only by beating that curve, not by separating at all, and the comparison
must be reported by size bin because a CLIP-patch scorer is expected to degrade exactly where this
panel lives (D33).

Neither diagnostic is a gate outcome. Selectivity shows that a scorer discriminates; it cannot show
that the surviving set became correct, because that is the question this reference cannot answer.

## Measured: boundary refinement by reconciliation — negative result

Miss attribution put 55 of 71 misses in `localized_below_match`, which made boundary refinement the
highest-value recall work. Before writing a refinement rule, its ceiling was measured.

**Boundary headroom** asks, for each missed reference, the best IoU any recorded observation
reached — any score, any view, retained or merged. It separates misses a better choice among
existing boxes could return from misses needing geometry no view produced.

| verdict | misses | share |
|---|---|---|
| recoverable_by_selection | 39 | 0.549 |
| needs_new_geometry | 30 | 0.423 |
| not_localized | 2 | 0.028 |

A box clearing IoU 0.50 already exists for 39 of the 71 misses. Taken at face value that is recall
0.901 to 0.955, which would clear G2. It is a ceiling stated in terms of the reference, and
inference cannot see the reference, so the next question is whether anything the pipeline actually
has would pick that box.

**It does not.** Of the 18 cases where a merged member cleared the criterion and its cluster's
representative did not:

| signal | cases it separates | of 18 |
|---|---|---|
| higher score | 0 | 18 |
| more magnified view | 1 | 18 |
| untruncated over truncated | 0 | 18 |
| **no signal separates** | **17** | **18** |

The clearing box scores *lower* than the representative in every case, by 0.024 to 0.242. It comes
from a more magnified view in one case, the same level in six, and a less magnified view in six —
the full frame beat its own tiles as often as the reverse. Neither box is truncated in any case,
so the truncation flag that makes the merge rule work carries no information here.

A representative-selection rule reaching this ceiling would have to prefer lower-scoring boxes
under conditions only the reference reveals. That is fitting to target labels, which the
constraints forbid, and it is the same mistake score-weighted fusion made in the opposite
direction — a plausible rule written before its ceiling was measured, costing 18 of 646 matches.

So WP-07's recall value is real but not reachable by reconciliation. The 39 need a box no
combination of recorded observations selects, and the 30 need geometry no view produced at all.
Both are the same requirement: boundaries refined against pixels, which needs a segmentation route
that is not staged. Reconciliation is closed as a path, with the measurement recorded so it is not
reopened by plausibility.

`selection_separability` in [`lynceus.diagnostics`](../src/lynceus/diagnostics.py) makes this check
reproducible: it reports, for any proposed signal, how many cases it would actually separate.

## Measured: TTN as a retained-set pruner — negative result

TTN was staged and measured against the standing bar. It is not a standalone scorer: reading its
inference code, it is an in-context comparator that is shown ten ground-truth reference patch
embeddings of a class followed by the candidate, and the candidate's accept/reject logit is read
from the last sequence position. Nothing is trained or fine-tuned; the adaptation is the reference
set. Both released variants were measured — `ttn`, trained only on classification, and `ttnd`,
fine-tuned on detection by its authors.

The reference bank is 100 CLIP ViT-L-14 patch embeddings per ontology class, drawn from VisDrone
**train** under A3. Two leakage controls are enforced in code rather than assumed: references never
come from the evaluation panel, and the 24 sequences appearing in both splits are excluded, because
VisDrone frames of one sequence are near duplicates and the mapping already treats the sequence
prefix as the scene identity. **One of those shared sequences is in the held-out panel**, so
omitting that check would have leaked directly.

Of 2808 retained boxes, 2616 were scoreable. The other 192 carry `unknown_object` or `entity`,
which make no class claim, so no class reference set exists to compare them against; scoring them
against some other class's references would invent a decision. Those boxes leave both curves, so
the pruner and the baseline are compared on exactly the same boxes.

| scorer | best selectivity | matched removed | unmatched removed |
|---|---|---|---|
| **detector score** | **0.513** | 0.347 | 0.861 |
| TTN (`ttn`) | 0.119 | 0.647 | 0.765 |
| TTN (`ttnd`) | 0.178 | 0.655 | 0.832 |

At equal retention — 689 of 2616 boxes kept — the detector score keeps 412 matched boxes and TTN
keeps 223. TTN removes nearly twice as many correct boxes to remove slightly fewer wrong ones.
Selectivity 0.119 is close to the 0.000 of a pruner removing at random.

| box size bin | boxes | detector | `ttn` | `ttnd` |
|---|---|---|---|---|
| <16 | 866 | 0.361 | 0.077 | 0.036 |
| 16-32 | 1055 | 0.465 | 0.047 | 0.088 |
| 32-96 | 598 | 0.750 | 0.248 | 0.375 |
| >=96 | 97 | 0.892 | 0.519 | 0.335 |

The size breakdown confirms the mechanism predicted in D33 before measurement: TTN improves
steadily with object size, from 0.077 below 16 pixels to 0.519 above 96. This panel lives almost
entirely in the two bins where it has least to work with — a CLIP ViT-L-14 patch embedding of a
twelve-pixel crop upscaled to 224 carries little. But the prediction does not rescue the result:
even in its best bin TTN is beaten by the detector score, and it loses in every bin.

This is a transfer failure, not a refutation of the paper, which evaluates general-photograph
datasets with detector pseudo-labels rather than sub-32-pixel aerial objects from a prompted
open-vocabulary model. It is also one ten-image panel outside A4 and is descriptive only. Within
that scope the conclusion is clean: **TTN does not clear the bar here, with either variant, in any
size bin.** The detector's own score remains the best available pruner, and adding TTN would buy a
dependency, an unstated licence and a shared-ancestry disclosure for a loss.

The pruner interface stays. `miss_attribution.py --scores` measures any external scorer against the
same curve on the same boxes, so the next candidate is a run rather than a rewrite.

## Measured: boundary refinement against pixels — SAM 3 earns its place

Reconciliation was closed as a path in D37: the geometry the 55 `localized_below_match` misses
need is not present among recorded observations, and no available signal selects it. What remained
was the annotation policy's own requirement, boundaries refined against pixels. Both an ungated
SAM 1 and the gated SAM 3 were measured on the held-out panel, identically: every retained box is
used as a box prompt, the mask the model rates highest becomes a tight box, and no box is added or
removed, so every change comes from geometry alone.

Refinement runs inside overlapping source-resolution tiles rather than per box or over the whole
frame. A whole-frame pass resizes a two-thousand-pixel image to 1024, delivering a twelve-pixel
object at about six; a per-box crop pays a full encoder pass to upscale a 128-pixel window and
buys nothing the tile did not carry. One tile pass serves every box inside it, which is what makes
source-resolution refinement affordable: the panel refines in 1m17 with SAM 3.

| matches at | original | SAM 1 | SAM 3 |
|---|---|---|---|
| IoU >= 0.50 | 646 | 628 | 641 |
| IoU >= 0.75 | 370 | 382 | **397** |
| IoU >= 0.85 | 214 | 214 | **230** |
| mean IoU of matches | 0.7715 | 0.7766 | **0.7839** |

**SAM 3 trades five loose matches for twenty-seven tight ones.** Recall at IoU 0.50 falls slightly,
from 0.901 to 0.894, while matches at 0.75 rise 7.3% and at 0.85 rise 7.5%. That trade favours the
gates that matter most: accepted annotation precision (G1) and useful-label coverage (G4) are both
scored at IoU 0.75, and localization quality (G5) is the ratio at 0.85 over 0.50, which moves from
0.331 to 0.359. It remains far below G5's proposed 0.95, so this is a step, not a pass.

The difference between the two models is a difference in kind, not degree. SAM 1's refinement is a
coin flip -- 50.4% of near-reference boxes improve and 49.6% worsen, median IoU change +0.0004,
boxes barely moving at a 0.978 median area ratio. It is symmetric noise applied to boxes sitting
near a threshold, and since more sit just above it than just below, random nudging loses 26
references to recover 8. SAM 3 is directional: 58.1% improve against 41.9%, median +0.0123, mean
+0.0202.

Gating on the model's own mask quality does not convert that signal into recall. Sweeping the
threshold removes refinements rather than selecting good ones: at 0.90 only one box changes, and
at 0.95 and above none does, so the result converges to the unrefined set. Losses fall from 17 to
13 to 1, but gains fall from 12 to 7 to 0 alongside them, and no threshold is net positive at
IoU 0.50. The self-reported quality is not concentrated where refinement helps.

Costs, recorded before any adoption. SAM 3 is gated `manual` on Hugging Face and its weights carry
a `license: other` whose terms must be read into the bundle manifest under WP-02. It needs a
`transformers` recent enough to provide `Sam3TrackerModel`, which the pinned 4.49.0 is not; the
measurement used a newer `transformers` supplied on `PYTHONPATH` so the pinned stack, its bundle
manifest and every cache key stayed valid, and a real adoption must stage that bump as its own
change under D32. `Sam3Model` with `input_boxes` returns DETR-style detections rather than a mask
per prompt, so the box-prompt path is `Sam3TrackerModel`; it loads from this checkpoint with no
missing, unexpected or mismatched keys.

DART accelerates SAM 3 by sharing its class-agnostic backbone across class prompts, with batched
multi-class decoding and TensorRT. Being training-free it changes cost and not results, and its
optimization applies to the multi-class detection loop rather than to refinement, which issues one
prompt per box with no class loop. It belongs at deployment, after a decision, and it cannot make
this decision.

This is one ten-image panel outside A4 and is descriptive under that tier. Within that scope it is
the first candidate change in this sequence that returns more than it costs.

### Replication on the second panel

One panel is not a result. The same refinement was run on the seed 0 panel, which is the
development selection rather than the held-out one, with no parameter changed.

| panel | references | IoU >= 0.50 | IoU >= 0.75 | IoU >= 0.85 | G5 ratio |
|---|---|---|---|---|---|
| seed 1 (held out) | 717 | -5 (-0.8%) | **+27 (+7.3%)** | **+16 (+7.5%)** | 0.331 -> 0.359 |
| seed 0 (development) | 827 | +6 (+0.9%) | **+41 (+10.7%)** | **+34 (+15.9%)** | 0.314 -> 0.361 |

The direction holds on both panels and the magnitude is larger on the second. The small loss at
IoU 0.50 on the held-out panel does not replicate -- seed 0 gains six matches there -- so that
loss is panel noise rather than a systematic cost, and the tight-threshold gain is the effect.
Across 1544 eligible references refinement adds 68 matches at 0.75 and 50 at 0.85.

Both panels are ten images of aerial imagery outside A4, so this is descriptive under that tier
and establishes no gate outcome. What it does establish is direction: unlike every other candidate
measured in this report, boundary refinement returns more than it costs, and it does so twice.

## Implemented: refinement as a pipeline stage, and what integrating it costs

Refinement is now a stage rather than an offline script. `lynceus annotate --refine-bundle` loads a
typed refiner adapter, and the contract additions in D45 are enforced in code: the proposed
geometry, the refined geometry, the refiner's mask quality and the agreement between the two are
written to the candidate store for every object, and where agreement falls below the declared
tolerance the proposed box is retained as an alternative and the boundary dimension becomes
unresolved rather than being silently overwritten. Refinement adds no object and removes none.

An end-to-end run on one panel image completed and validated: 382 objects, 382 refinement records,
30 extent disagreements flagged, and the refiner's identity and configuration recorded in the
events log beside the detector's.

`preflight` previously hardcoded a single adapter and OWLv2's file layout, so the bundle format
could not express a second model. It now names the adapter and checks the file set that adapter
requires -- OWLv2 ships a preprocessor config where SAM 3 ships a processor config -- and a bundle
cannot pass by carrying the other adapter's files.

### The dependency bump changes the detector's output

SAM 3 needs a `transformers` newer than the pinned 4.49.0. Preflight refused the OWLv2 bundle
under the newer version, which is the dependency-invalidation rule working as designed. OWLv2
itself runs on 5.17.0, so a shared dependency set is feasible. But the results are not the same.

| | objects on one panel image |
|---|---|
| transformers 4.49.0 (pinned) | 394 |
| transformers 5.17.0 | 382 |
| boxes identical between them | **0 of 394 / 382** |

The image's normalized-pixel hash is identical and every other input is unchanged, so this is the
dependency alone. Not one box survives the bump unmoved.

That is a concrete cost, and it bounds what the refinement result currently supports. The measured
gain -- 68 additional matches at IoU 0.75 across two panels -- was obtained by refining boxes that
4.49.0 produced. Whether the same gain appears on 5.17.0's different boxes is unmeasured. Two
routes are open and both have a price: keep detection on the pinned version and run refinement in
a separate process against its own bundle, paying the cost of two dependency sets; or adopt 5.17.0
for both and re-measure the baseline, the ablations and the refinement gain on it, because every
number in this report was taken on 4.49.0.

Nothing here is adopted by default. Refinement is off unless a refiner bundle is named, and the
pinned stack is untouched by a run that does not ask for refinement.

### The SAM 3 licence, read

SAM 3 is **not** Apache-2.0, unlike SAM 1 and SAM 2. It ships under Meta's **SAM License dated
19 November 2025**, a bespoke agreement whose defined "SAM Materials" explicitly include trained
model weights. Three sources agree: the code repository reports no recognised SPDX licence, the
model repository is gated with `license: other`, and the licence text is a Meta agreement rather
than a standard one.

Its terms are more permissive than that metadata suggested. The grant is non-exclusive, worldwide,
royalty-free, and covers use, reproduction, distribution, derivative works and modification, so
**redistribution is permitted** and a bundle is therefore possible. Three obligations attach: a
copy of the agreement must accompany any redistribution, published research using it must
acknowledge SAM Materials, and reverse engineering is forbidden.

The field-of-use restrictions are the part that matters for this delivery, because they are
conditions on the product and not only on the file. The licence prohibits use for military or
warfare purposes, nuclear applications, espionage, and guns or illegal weapons, and binds use to
ITAR and trade controls. The qualification domain currently exercised here is aerial drone
imagery, which is adjacent to several of those, so a deployment built on this refiner inherits a
use restriction that a deployment built on SAM 2 would not. That is a product constraint to
declare in a qualification manifest, not a packaging detail.

The licence text is stored beside the checkpoint and its hash is recorded in the bundle manifest,
so a later change of terms is detectable rather than assumed away.

## Re-measured on the new pin

Everything that informed a decision was re-run on transformers 5.17.0: both panels, the probe run,
miss attribution, and the refinement gain. The question was whether the conclusions survived a
change that moved every box.

They did. The direction and the magnitude both hold; only the third decimal moved.

### Miss attribution, held-out panel

| bucket | on 4.49.0 | on 5.17.0 |
|---|---|---|
| localized_below_match | 55 (77.5%) | 60 (80.0%) |
| below_threshold | 9 | 8 |
| merged_with_neighbour | 4 | 4 |
| merged_away | 1 | 1 |
| **never_proposed** | **2** | **2** |
| total misses | 71 | 75 |

The finding that closed the detector question is unchanged: two references out of 717 were never
proposed at any score in any view. Boundary quality still carries four fifths of the misses, and
boundary headroom still splits them 42 recoverable by selection against 31 needing geometry no
view produced, against 39 and 30 before.

### Refinement gain

| panel | references | IoU >= 0.50 | IoU >= 0.75 | IoU >= 0.85 | G5 ratio |
|---|---|---|---|---|---|
| seed 1 (held out) | 717 | -5 | **+19** | **+14** | 0.335 -> 0.360 |
| seed 0 (development) | 827 | +8 | **+45** | **+31** | 0.309 -> 0.350 |
| **combined** | 1544 | **+3** | **+64** | **+45** | |

On 4.49.0 the same comparison gave +1, +68 and +50. The trade is the same one: a handful of loose
matches exchanged for a large gain at the thresholds G1, G4 and G5 are scored against, positive on
both panels at both tight thresholds, and near neutral at IoU 0.50. The small loss at 0.50 on the
held-out panel again fails to replicate on the development panel, which gains eight there.

What this establishes is narrow and worth stating exactly. It does not show that 5.17.0 is better
or worse than 4.49.0 as a detector; that comparison was not run and the panels are far too small to
support it. It shows that the refinement result is not an artifact of one library version, which is
the only question the re-measurement was asked.

## Implemented: the exhaustive-panel harness

The finite-category ceiling is the oldest open item in this report and the only one that blocks
three things at once. The protocol for lifting it is now specified (D49) and the harness that makes
it verifiable is implemented and tested.

`prepare` lays out the procedure as files: one source-resolution review crop per tile, and a
skeleton carrying the image's content hash, scene identity and near-duplicate group already filled
in. `ingest` checks what comes back and refuses five specific failures, each of which is invisible
in a finished panel: an unvisited review tile, which looks like empty space; a missing granularity
or scene-layer tag, which makes a part look like an instance; ambiguity resolved by fiat instead of
marked unresolved; absent leakage identity; and a panel that cannot declare it was annotated before
predictions were visible. That last one is a declaration rather than a check, because nothing in a
box records what its annotator had seen -- so a panel that cannot declare it is refused rather than
silently treated as independent.

Only a resolved physical instance is an eligible reference. Parts, groups, stuff, reflected and
depicted content, and entities the policy does not settle are all retained and counted, so their
absence from a denominator is recorded rather than hidden.

The round trip was exercised end to end: ten images prepared into 6 review tiles each, ingestion
correctly refusing an unswept panel with `incomplete_sweep: 6 of 6 tiles unvisited`, and a filled
panel passing through `lynceus evaluate` with `category_scope: null`, no out-of-scope exclusions,
and denominators intact.

The default selection is the held-out panel already measured, so annotating those images converts
the existing precision bound into a number rather than producing a second result comparable to
nothing. What remains is the annotation itself, which is human work this delivery cannot do for
itself, and it is now the single blocker between this system and an interpretable precision figure,
a first calibration, and any G9 outcome at all.

### The review page

Filling sixty tile skeletons by hand is not a workflow, so preparation now also writes a local
review page per image. It opens over `file://` beside a copy of its image, with no server, no
network and no external dependency, which is what lets the panel be built under the same offline
conditions the annotator itself requires.

The page is organised around the sweep rather than around the image: it walks the planned tiles,
zooms each to fill the viewport so a twelve-pixel entity is actually visible, tracks which tiles
have been reviewed, and writes the `visited_tiles` record that ingestion refuses to do without. A
box is drawn by dragging, and the form that follows asks for exactly the four things the protocol
requires and ingestion enforces -- class from the declared ontology, kind, scene layer, and whether
the policy settles the entity or leaves it unresolved. Unresolved entities are drawn in a different
colour from eligible ones, so the distinction is visible while annotating rather than discovered at
ingestion.

It shows no prediction and cannot load a run directory, which the tests assert rather than assume:
the page carries only the reference skeleton, the tile plan and the ontology's class list, and it
contains no URL, no `fetch`, and no external script or stylesheet.

What is tested is the page's contract -- its structure, its embedded state, the absence of any
network reference, and that it offers every field ingestion requires. The click-through itself is
not tested, because there is no browser in this environment to drive it. That is a real gap: the
page's interaction has been reasoned about and not exercised, and the first annotator will be the
first to run it.

## Measured: automatic verification, and what it says is actually wrong

Manual adjudication of 2374 retained boxes was not going to happen, so the boxes were verified
against an independent model instead. SAM 3 was prompted with the ontology's class names, tiled at
source resolution so a twelve-pixel object is actually visible, and every retained box was checked
for a confirming detection.

Agreement is not truth, and a second model cannot establish that a box is correct. What makes this
more than a vote is that half the population is already settled: a box matching an eligible
reference is a known true positive, so the verifier's confirmation rate on those measures its
sensitivity directly, on this data at this object scale.

| quantity | value | status |
|---|---|---|
| retained boxes | 2374 | |
| settled by the reference | 631 | measured |
| undecided | 1743 | |
| precision lower bound | 0.266 | measured, assumes nothing undecided is real |
| **verifier sensitivity on known positives** | **0.967** | **measured** |
| confirmation rate on undecided boxes | 0.357 | measured, not a precision figure |
| precision if every confirmed undecided box is real | 0.528 | assumption |
| precision under equal sensitivity | 0.537 | assumption |

The assumption that would ordinarily wreck this is that the verifier behaves the same on both
populations, and the usual way it fails is a verifier that cannot see small objects. That was
checked rather than hoped, and it does not fail here:

| size bin | known positives | sensitivity | undecided | confirmed |
|---|---|---|---|---|
| <16 | 152 | 0.947 | 708 | 0.496 |
| 16-32 | 256 | 0.961 | 685 | 0.337 |
| 32-96 | 193 | 0.985 | 302 | 0.133 |
| >=96 | 30 | 1.000 | 48 | 0.021 |

Sensitivity is above 0.94 in every bin including the smallest, which is where this panel lives. The
verifier is not blind to the objects in question, so a low confirmation rate among undecided boxes
is more readily read as those boxes being absent than as the verifier missing them. That is still
an inference and not a measurement, and it remains descriptive.

Taken at face value it says roughly 644 of the 1743 undecided boxes are real objects VisDrone never
annotated, and that precision is nearer 0.53 than the 0.266 bound. It is consistent with what the
overlays show by eye, which was that the unmatched boxes are a mixture rather than uniformly wrong.

Per-class sensitivity rests on small denominators -- three known positives for motorcycle, none for
truck -- so only the aggregate and the size split carry weight. The aggregate is dominated by
`vehicle` at 410 known positives and `person` at 136.

### Classification: the disagreements are mostly not errors

Comparing emitted labels against the verifier's on the 1233 boxes both models place, exact
agreement is 0.432, which read alone would suggest a serious naming problem. It is the wrong test.
An ontology has a hierarchy, and emitting `vehicle` where the verifier says `car` is the
conservative naming policy working as designed, not a contradiction.

| outcome | boxes | rate |
|---|---|---|
| exact agreement | 533 | 0.432 |
| compatible, emitted label is broader | 413 | |
| **compatible overall** | 946 | **0.767** |
| genuine conflict | 287 | 0.233 |

The 413 compatible-but-broader cases are the specificity cost measured directly, and they are
concentrated: `vehicle` where the verifier says `car` (207), `motorcycle` (115), `van` (32),
`bicycle` (21), `tricycle` (18). This is the same quantity G4's specific-label floor exists to
constrain, now visible as a list rather than a share.

The genuine conflicts are concentrated too, and they are about people:

| emitted | verifier | count |
|---|---|---|
| unknown_object | person | 109 |
| bicycle | person | 51 |
| vehicle | person | 44 |
| bicycle | motorcycle | 12 |

Two hundred and four of the 287 conflicts involve a person the system either abstained on or called
a vehicle. That is the failure the conservative naming measurement already predicted from a
different direction: `person` has no useful ancestor in this ontology, so when it competes with
`bicycle` on a rider the fallback emits `unknown_object` rather than guess. The verifier says those
are people. Neither model is a reference, so this localises the problem rather than settling it,
and it points at the ontology rather than at the detector.

## Measured: searching a prompt vocabulary, and two things it taught

Phrasing is configuration, and it was never chosen: both the detector and the SAM 3 verifier built
their prompts from the ontology's class names verbatim, in separately written lines, with the
`synonyms` field populated on none of the twelve classes and read by nothing. Vocabularies are now
their own versioned artifacts, one per model family, and they are searched rather than written.

The search runs on the development panel, caches image features and box predictions -- which no
phrasing changes -- and re-runs only the text tower and class head per candidate, so a candidate
costs seconds and the scores are exactly those of a full forward pass. Forty candidate phrasings
across nine classes, coordinate ascent over the whole vocabulary because the emitted label is
whichever class wins across all prompts.

### The first objective was wrong, and held-out measurement caught it

The first search maximised class-agnostic localization and gained 32.9% on development. The gain
replicated on the held-out panel, so it was not noise. It was also worthless:

| | baseline | searched v1 |
|---|---|---|
| retained boxes | 2374 | 1797 |
| its own objective | 0.408 | **0.473** |
| `unknown_object` emitted | 166 | **278** |
| `person` emitted | 264 | **138** |
| useful-label coverage (G4) | 0.480 | **0.386** |

A vocabulary can buy cleaner localization by pushing contested boxes into abstention: fewer boxes,
better placed, worse named. The measure being optimized rose sixteen per cent while the measure the
gate scores fell nineteen. Reporting only the development number would have shipped this as a
one-third improvement. The artifact is kept in
[`docs/rejected/`](rejected/vocabulary-owlv2-localization-objective.json) rather than deleted,
because a rejected result is evidence, and the reason is written into the objective's docstring so
it is not re-derived.

### The corrected objective improves the objective and still costs coverage

The objective now counts what G1 and G4 count: a reference found at IoU 0.75 under a permitted,
non-abstaining label. That search gained 43% on development.

| held-out panel, 717 references | boxes | localized @.50 | named @.75 | G4 coverage | precision @.50 | objective |
|---|---|---|---|---|---|---|
| baseline | 2374 | 631 | 344 | 0.4798 | 0.2658 | 0.2226 |
| searched v1 (rejected) | 1797 | 595 | 277 | 0.3863 | 0.3311 | 0.2204 |
| **searched v2** | **1683** | 601 | 334 | **0.4658** | **0.3571** | **0.2783** |

Two things to read here and neither is a clean win.

The development gain was 43% and the held-out gain on the same objective was 25%, so about two
fifths of the improvement did not survive the move. That gap is what selecting among forty
candidates on ten images costs, and it is the reason the development score is never the result.

More importantly, **G4 itself went down**, from 0.4798 to 0.4658. The objective rose because it
includes precision, which improved by a third, on twenty-nine per cent fewer boxes. That is a
defensible trade -- far fewer false boxes for one and a half points of coverage -- but it is a
trade, and the gate it is nominally aimed at regressed. So `owlv2-aerial-searched-v2` ships as an
available artifact and not as the default; adopting it is a product decision about which gate
matters more, and the evidence for it is here rather than in a headline.

The abstention problem did not recur: `unknown_object` fell from 166 to 130 under v2, against 278
under v1.

## Measured: the abstention was a category error, and fixing it removes it entirely

Every one of the 166 `unknown_object` labels on the held-out panel came from `person` competing with
a vehicle class, and the SAM 3 verifier independently called 109 of those abstentions people. Two
measurements from different directions pointed at the same place, and neither pointed at the
detector.

The cause was in the naming rule rather than in the ontology's vocabulary or the model's vision. The
rule compares the two strongest classes on a box and, treating them as competing readings of one
entity, falls back to the class they jointly support -- or to `unknown_object` where that class is
not useful. But a person on a motorcycle is two entities the policy counts separately, and a box
around the rider contains both. The detector scoring `person` and `motorcycle` there was describing
what is present, not hesitating; the rule resolved a conflict that did not exist and discarded a
supported class to do it. It is the same error the policy forbids elsewhere, where containment is
taken to imply identity.

Ontology version 2 declares which classes can co-occur in one region -- a person rides or is carried
by each vehicle class -- and naming does not fall back on a declared co-occurrence. The class list is
unchanged, and version 1 remains valid for the runs measured under it.

| held-out panel, 717 references | ontology v1 | v2 | change |
|---|---|---|---|
| **`unknown_object` emitted** | **166** | **0** | **-166** |
| retained boxes | 2374 | 2311 | -63 |
| class-correct @ IoU 0.50 | 585 | 599 | +14 |
| class-correct @ IoU 0.75 | 344 | 346 | +2 |
| G4 useful-label coverage | 0.4798 | 0.4826 | +0.0028 |
| precision @ IoU 0.50 | 0.2658 | 0.2713 | +0.0055 |
| recall @ IoU 0.50 | 0.8801 | 0.8745 | -0.0056 |

The rule fired on 121 boxes. Abstention is gone as a category, `person` rose from 264 to 327 and
`bicycle` from 318 to 347, and precision and useful-label coverage both improved slightly against
four localized matches lost.

The size of the coverage gain is the more informative number. Removing 166 abstentions added only
two correctly labelled references at IoU 0.75, because those boxes were mostly small riders whose
geometry does not clear the tight threshold. The abstention was concealing a localization limit
rather than causing a naming loss: the objects were found and are now named, and they still do not
clear 0.75. That returns the question to boundary quality, which is where miss attribution and
boundary headroom already placed it, and it is a reminder that removing a visible failure is not
the same as removing the thing it was standing in front of.

