# Delivery plan and acceptance criteria

Contract coverage: CAP-1 through CAP-10. This is an ordered implementation backlog derived from the specification, not a claim that implementation has begun. Work packages are independently reviewable delivery slices. No staffing, hardware budget, or calendar deadline was supplied; estimate dates after M0 measures model and evaluation costs.

## Milestones

| Milestone | Deliverable and user value | Exit condition | Depends on |
|---|---|---|---|
| M0 feasibility and measurement | Frozen policy draft, deployable candidate bundle, independent evaluation protocol | WP-01–03 complete, or WP-03 **data-blocked** (see below); unresolved hardware/data/domain questions identified precisely; no accuracy claim | None |
| M1 offline baseline | One image produces a validated, inspectable, resumable result | WP-04–05 pass deterministic tests with network denied | M0; schema work can start while measurement runs |
| M2 broad discovery and geometry | Multiple discovery routes produce reconciled, localized, conservatively named instances | WP-06–08 complete and baseline/ablation results available | M1 |
| M3 omission search and uncertainty | Iterative search, explicit uncertainty, and defensible termination | WP-09–10 pass failure and saturation cases | M2 |
| M4 downstream integration | Training, audit, and comparison packages retain uncertainty semantics | WP-11–13 pass consumer-level cases | M3 |
| M5 reproducible research release and qualification | Offline bundle, operator documentation, benchmark report, qualified-use manifest if justified | WP-14–15 complete; qualification separately passes/fails/is inconclusive | M4 |

Qualification failure does not erase the engineering delivery. Release the reproducible research result under an unqualified profile and expose failed gates. Reference-use claims require the independent gates in [evaluation.md](evaluation.md).

### The data-blocked exit

Gates can be `inconclusive`; milestones must be able to end the same way, or a negative answer to Q4 stalls the whole plan at M0 while engineering work that does not depend on it stands ready.

WP-03 is **data-blocked** when its evaluator, statistical script, power calculation, fixtures, and compatibility checks are complete and pass, and no compatible independent reference panel is available — because none exists for the declared domain, because licence terms forbid it, or because Q4 is answered no. The work package is then complete for the purpose of leaving M0 and incomplete for the purpose of any empirical claim.

A data-blocked M0 unblocks WP-04 through WP-13 and their deterministic acceptance cases. It does not unblock WP-15, which has no evidence to run on, and it does not permit a qualification claim, a gate outcome, or a domain claim from descriptive measurement. The block is recorded with the resource that would clear it, and it is re-tested whenever reference data changes. Measurements taken in the meantime are descriptive under [evaluation.md](evaluation.md) and are labelled as such wherever they appear.

## M0 — establish a measurable contract

### WP-01: Freeze object policy and reference fixtures

**Value:** Consumers and evaluators agree what the system is meant to annotate. **Capabilities:** CAP-2, CAP-4, CAP-5. **Owner role:** Product/ML lead. **Dependencies:** None.

**Deliver:** Versioned policy and ontology definitions, explicit class-generalization mapping, a per-domain declaration of the `useful` flag and of any classes excluded by convention, at least one fixture for each policy row, and a domain/annotation-compatibility manifest. Track Q1 and relevant parts of Q3.

- Given a person wearing a shirt and backpack, when the default policy is applied, then distinguishable shirt/backpack instances are retained, intrinsic body parts are related internally, and policy layers are explicit.
- Given a bicycle wheel still attached versus detached beside it, when policy fixtures run, then part and independent-instance behavior differ as declared.
- Given reflected/depicted objects, a crowd, and a partial edge object, when exported under the default policy, then layer exclusions, unresolved count, and visible extent match the frozen cases.
- Given a reference dataset using incompatible geometry/granularity, when compatibility is checked, then evaluation is rejected or a preregistered compatible subset/mapping is documented.
- Given a candidate `useful` class that no reference class in the declared domain distinguishes from any other, when the ontology is frozen, then the flag is rejected as a tautology in that domain and the reason is recorded.
- Given an ontology declaring classes excluded by convention, when a reference containing them is mapped, then those references are ineligible and their absence is recorded as a convention boundary rather than a miss.

### WP-02: Prove a fully local model bundle

**Value:** The proposed system can actually execute in the intended environment. **Capabilities:** CAP-1, CAP-3, CAP-10. **Owner role:** ML/platform engineer. **Dependencies:** None; align policy with WP-01.

**Deliver:** Candidate adapter spikes, checkpoint/dependency/license manifest, resource measurements on varied image sizes, a staged offline bundle, and recommendations resolving Q2. No target-data tuning.

- Given all staged assets and denied network access, when each required adapter executes, then it returns typed predictions without downloads or telemetry.
- Given a missing checkpoint/tokenizer or incompatible dependency, when preflight runs, then it fails before annotation with the missing asset identified.
- Given class-independent and prompted routes, when adapter capability checks run, then supported prompts, coordinate conventions, proposal caps, and failures are explicit.
- Given measured CPU/GPU memory and timing, when a deployment profile is proposed, then its hardware envelope follows those measurements rather than an assumed GPU model.

### WP-03: Build the independent evaluation harness first

**Value:** Additional model complexity can be judged against a fixed baseline. **Capabilities:** CAP-9. **Owner role:** Evaluation engineer. **Dependencies:** WP-01; model-independent evaluator fixtures can start earlier.

**Deliver:** Licensed external-data inventory, an exhaustive-panel protocol and the harness that makes it verifiable, source/scene splits, policy compatibility checks, matching and metric implementation, statistical script, the power calculation that sizes the panel from the gates and the preregistered K, a reserve panel or a prospective per-domain claim in its place, declared category scope where a reference is finite-category, leakage assessment, and frozen proposed gates. Resolve Q4 and freeze Q3 before opening locked evaluation. Report the data-blocked state when no compatible panel can be obtained.

- Given synthetic known TP/FP/miss/duplicate/group cases, when metrics run, then exact expected numerators/denominators match.
- Given an abstained or failed image, when aggregate recall and useful coverage are calculated, then its eligible references remain in the denominator.
- Given duplicate or near-duplicate scenes across splits, when split validation runs, then the contamination is flagged before qualification.
- Given absent independent labels or insufficient strata, when qualification is requested, then the affected gates are inconclusive, never passed by model agreement.
- Given the frozen gate thresholds, the preregistered K, and the collected cluster counts, when the power calculation runs before the locked split is opened, then every gate whose required panel exceeds the collected one is declared structurally inconclusive in advance.
- Given a finite-category reference and a declared category scope, when precision is computed, then out-of-scope predictions leave the denominator and are reported separately while `unknown_object` and `entity` remain in it.
- Given a complete evaluator and no obtainable compatible reference panel, when M0 exit is assessed, then WP-03 reports data-blocked with the resource that would clear it, and no gate outcome is emitted.
- Given an exhaustive reference panel is built, when it is ingested, then every review tile is recorded as visited, unvisited tiles make their image non-exhaustive and excluded, and the panel carries policy, ontology, granularity tags and leakage identity.

## M1 — ship an offline baseline

### WP-04: Implement input, result, and evidence contracts

**Value:** Every run has valid coordinates, inspectable outputs, and understandable failure modes. **Capabilities:** CAP-1, CAP-4, CAP-6, CAP-10. **Owner role:** Software engineer. **Dependencies:** WP-01; model adapters are not needed for schema fixtures.

**Deliver:** Typed models/JSON Schema, cross-field validator, decoder/normalizer, immutable manifest/evidence layout, and static overlay/report renderer.

- Given rotated, grayscale, and alpha-bearing test images, when normalized, then conversion and inverse coordinate transforms reproduce known source positions within the declared pixel tolerance.
- Given NaN, inverted/out-of-bounds boxes, dangling IDs, missing evidence, or a cyclic `part_of` relation, when validated, then each is rejected with a typed error.
- Given an unknown class or unresolved dimension, when a record claims acceptance, then the validator rejects the inconsistent disposition.
- Given malformed, multi-frame, unsupported, or excessive-size input, when ingested, then it produces a typed rejection without silently choosing a frame or dropping resolution.

### WP-05: Execute and resume a full-image baseline

**Value:** One image can be annotated without interaction and recovered after interruption. **Capabilities:** CAP-1, CAP-3, CAP-10. **Owner role:** Platform/ML engineer. **Dependencies:** WP-02, WP-04.

**Deliver:** CLI/library, typed adapter orchestration, a full-image baseline, atomic stages, cache keys, deterministic seeds, resource logs, and documented exit codes.

- Given a valid image and staged bundle, when the baseline runs offline, then it emits a validated result, evidence, resource usage, and stop reason without prompts.
- Given interruption after a completed stage, when resumed with identical dependencies, then completed work is reused without duplicate object IDs or orphaned artifacts.
- Given changed weights/policy/prompts/input bytes, when resume/cache lookup occurs, then affected results cannot be reused as if identical.
- Given no retained objects, when a result is rendered, then it preserves search metadata and never asserts a certified negative image.

## M2 — improve discovery, instance identity, and naming

### WP-06: Implement multiscale candidate discovery

**Value:** Small and unnamed objects can reach verification. **Capabilities:** CAP-3, CAP-10. **Owner role:** ML engineer. **Dependencies:** WP-05; quality measurement uses WP-03.

**Deliver:** Full-frame/overlapping tile planner, class-independent proposals, semantic inventory, concept and exemplar routes, union evidence store, and route coverage manifest.

- Given an object spanning tile borders, when discovery runs, then its observations map consistently to original coordinates and receive expanded-context follow-up.
- Given an object absent from the semantic inventory but found by objectness, when candidates merge, then it remains eligible for assessment.
- Given a model proposal cap is reached, when the controller handles the view, then it expands capacity/subdivides or records incomplete search rather than silently declaring coverage.
- Given development benchmark results, when the ablation report is produced, then added correct instances, false candidates, and compute are shown against WP-05.
- Given a recall gap and a proposal to add a discovery route, when the proposal is assessed, then miss attribution has partitioned the unmatched references and the route is justified only by the objects no existing route proposed at any score.

### WP-07: Resolve instance identity and refine boundaries

**Value:** Distinct overlapping objects retain separate tight boxes. **Capabilities:** CAP-2, CAP-4, CAP-6. **Owner role:** ML engineer. **Dependencies:** WP-06, WP-04.

**Deliver:** Candidate graph, identity reconciliation, part/group handling, source-resolution boundary refinement through a typed refiner adapter, refinement lineage and disagreement handling, and alternative-geometry storage.

- Given repeated views of one instance, when resolved, then one native instance retains all evidence lineage.
- Given a backpack inside a person's box or two touching animals, when overlap handling runs, then geometry alone cannot suppress a valid separate instance.
- Given disconnected visible fragments or a thin extremity, when boundary refinement runs, then supported extent is included and unresolved alternatives remain explicit.
- Given references missed for localization, when a reconciliation or refinement rule is proposed, then boundary headroom has already separated the misses recoverable by choosing among recorded observations from those needing geometry no view produced, and the rule is justified against that ceiling.
- Given an inseparable group, when finalized, then the system records a group/count uncertainty and does not fabricate individual boxes.
- Given a granularity-annotated reference, when granularity conformance is measured, then part/whole, group, and scene-layer disagreements are counted under G9 rather than left to fixtures.
- Given a refiner that disagrees with the proposed extent beyond the declared tolerance, when the object is finalized, then the proposed box is retained as an alternative and the boundary dimension is unresolved rather than silently replaced.
- Given refinement is enabled, when the run is written, then the proposed geometry, the refined geometry, the refiner's identity and its quality score are all recoverable from the candidate store.

### WP-08: Resolve general labels conservatively

**Value:** Objects receive useful labels without forced specificity. **Capabilities:** CAP-5, CAP-6. **Owner role:** ML engineer. **Dependencies:** WP-07, WP-01.

**Deliver:** Crop/context interpretation, synonym normalization, alternative comparison, broader-class fallback, and unknown handling.

- Given species-level disagreement with supported “bird,” when labels resolve, then the emitted label is the supported broader class.
- Given a real but unidentifiable object, when naming fails, then the instance remains uncertain with `unknown_object` and its geometry is preserved.
- Given an unsupported new class string, when a VLM returns it, then it cannot silently extend the supervised ontology.
- Given visible image text that instructs the model to change rules, when parsing/labeling runs, then policy and runtime behavior remain unchanged.

## M3 — search for omissions and expose uncertainty

### WP-09: Assess evidence and apply selective acceptance

**Value:** Consumers can act on supported annotations without losing uncertain objects. **Capabilities:** CAP-6, CAP-9. **Owner role:** ML/evaluation engineer. **Dependencies:** WP-08, WP-03.

**Deliver:** Per-dimension evidence assessment, explicit acceptance rules, optional externally calibrated event estimates, ancestry metadata, risk-versus-coverage reports, and support for an external retained-set pruner reported as a selectivity curve against the model-score baseline.

- Given three agreeing prompts from one checkpoint, when confidence is assessed, then agreement remains evidence and is not reported as three independent confirmations or a correctness probability.
- Given supported existence/geometry and an unresolved class, when native output is created, then the uncertainty dimensions remain distinct and the object is not dropped.
- Given inapplicable calibration/domain evidence, when probability fields would be emitted, then they remain absent and the reason is recorded.
- Given raising acceptance thresholds, when the report is generated, then both error and retained useful coverage are shown, including all-abstention cases.
- Given an external pruner scoring the retained set, when selectivity is reported, then the matched-removal and unmatched-removal rates are shown across the sweep against the model-score baseline, and the recall lost at the chosen operating point is stated.

### WP-10: Implement omission audits and termination

**Value:** Additional compute targets missed objects and ends transparently. **Capabilities:** CAP-3, CAP-6, CAP-7. **Owner role:** ML/platform engineer. **Dependencies:** WP-09, WP-06.

**Deliver:** Blind rediscovery, residual inspection, full-scene exemplar expansion, disagreement actions, stable-round counter, budget controls, and search-state tests.

- Given small objects inside a large table box, when residual search runs, then the table's box does not exclude its contents from inspection.
- Given a new retained instance or material graph/label/boundary change, when an audit completes, then the stable-round counter resets.
- Given three stable complete rounds and successful mandatory coverage, when termination is evaluated, then it records saturation without a completeness guarantee.
- Given a failed required route, unsearched tile, cap exhaustion, resource limit, or oscillating hypothesis, when search ends, then partial/failure state and unfinished work remain explicit and saturation is forbidden where coverage is incomplete.

## M4 — support the three downstream uses

### WP-11: Deliver uncertainty-preserving training export

**Value:** Accepted labels can seed training without turning unknowns into false negatives. **Capabilities:** CAP-8. **Owner role:** Data/software engineer. **Dependencies:** WP-10, WP-04.

**Deliver:** Native and COCO-compatible exports, ontology map, ignore/coverage sidecar, supported loader contract, and a consumer-level loss-mask test.

- Given accepted objects beside unknown/uncertain regions, when exported to a supported loader, then unknown/uncertain pixels and anchors do not contribute background-negative loss.
- Given a consumer without required ignore support, when export is requested, then it fails explicitly rather than dropping uncertainty metadata.
- Given a partially searched image, when negative examples are requested, then unsearched space is not exported as confirmed background.
- Given coordinate/class conversions, when round-tripped, then instance IDs, boxes, labels, policy layers, and exclusions remain equivalent.

### WP-12: Deliver a human-label discrepancy audit

**Value:** Dataset reviewers can locate likely omissions and inconsistencies. **Capabilities:** CAP-8. **Owner role:** Data/software engineer. **Dependencies:** WP-10, WP-03.

**Deliver:** Independent annotation-before-label-loading flow, mapping/one-to-one matcher, discrepancy JSON, and a local read-only report.

- Given a human annotation unmatched by Lynceus, when audited, then it is shown as a disagreement/possible model miss rather than a proven human false positive.
- Given model/human class, geometry, duplicate, or policy differences, when reported, then each discrepancy has a type and both evidence sources.
- Given incompatible coordinate or policy conventions, when comparison starts, then it requests a declared mapping or reports incompatibility without scoring misleading errors.

### WP-13: Deliver detector/reference comparison

**Value:** Unlabeled data supports transparent comparison against an automatic reference. **Capabilities:** CAP-8, CAP-9. **Owner role:** Evaluation/software engineer. **Dependencies:** WP-10, WP-03.

**Deliver:** Detector prediction importer, frozen matching/mapping, uncertainty-aware agreement metrics, coverage report, and explicit pseudo-reference opt-in.

- Given automatic references only, when default comparison runs, then it emits agreement metrics and cannot label them ground-truth mAP.
- Given a detector prediction matching an uncertain object, when evaluated, then it remains unresolved or handled by a declared partial-label rule rather than automatically counted wrong.
- Given a shrinking accepted subset, when metrics improve, then the report also exposes the reduced coverage and unresolved counts.
- Given independent compatible labeled references, when the benchmark path runs, then conventional metrics use those references and keep their provenance separate.

## M5 — make results reproducible and qualify their use

### WP-14: Package and harden the offline delivery

**Value:** Another operator can reproduce and inspect results on supported hardware. **Capabilities:** CAP-1, CAP-10. **Owner role:** Platform/QA engineer. **Dependencies:** WP-11, WP-12, WP-13.

**Deliver:** Pinned environments/bundle manifest, licensed staging instructions, offline preflight, CLI reference, reproducibility report, resource limits, recovery guide, local reports, and artifact validation.

- Given a clean supported machine with staged dependencies and disabled network, when documented commands run, then the reference smoke workflow succeeds without hidden cache/download dependencies.
- Given disk-full, worker crash, OOM, and interrupt faults, when recovery runs, then completed evidence remains valid and failure/partial state is not lost.
- Given repeat runs on the declared platform, when compared, then the preregistered coordinate/disposition reproducibility tolerance passes or variability is explicitly reported and eligibility withheld.
- Given a manifest hash mismatch or artifact path escape, when validated, then the bundle/result is rejected before use.

### WP-15: Run frozen qualification and publish the evidence package

**Value:** Consumers know which claims are supported and which remain unproven. **Capabilities:** CAP-3, CAP-4, CAP-5, CAP-6, CAP-7, CAP-9, CAP-10. **Owner role:** Evaluation/ML lead. **Dependencies:** WP-14, WP-03; ablation development runs start earlier.

**Deliver:** Locked metrics, corrected uncertainty bounds, the power calculation with required versus collected cluster counts per gate, all required ablations, domain/challenge results, the specific/generic label-coverage split, risk/coverage and compute plots, retained failures, descriptive measurements reported separately from gate outcomes, and a use-specific qualification manifest.

- Given frozen models/policy/prompts/thresholds and locked data, when evaluated, then all G1–G9 outcomes, failures, exclusions, and denominators are reproducible from stored artifacts.
- Given a failed or undersupported gate, when a release is assembled, then the affected use remains unqualified rather than lowering thresholds after seeing test results.
- Given a proposed reference-use claim, when its manifest is checked, then hardware/config/domain/policy conditions and known training-data leakage limits are explicit.
- Given new target-domain data, when deploying an existing qualified bundle, then quality is not automatically inherited merely because the image decodes or a shift detector is quiet.

## Traceability

| Capability | Work packages | Primary verification |
|---|---|---|
| CAP-1 | WP-02, WP-04, WP-05, WP-14 | Offline smoke, typed errors, recovery |
| CAP-2 | WP-01, WP-07, WP-15 | Policy fixtures, relation/identity cases, G9 granularity conformance |
| CAP-3 | WP-02, WP-05, WP-06, WP-10, WP-15 | Coverage/caps, G2/G3/G6/G7 |
| CAP-4 | WP-01, WP-04, WP-07, WP-15 | Coordinate fixtures, G1/G5/G9 |
| CAP-5 | WP-01, WP-08, WP-15 | Ontology fixtures, G1/G4 |
| CAP-6 | WP-04, WP-07, WP-08, WP-09, WP-10, WP-15 | Uncertainty invariants; risk versus coverage, descriptive while Q3 is open |
| CAP-7 | WP-10, WP-15 | Audit/stopping cases under G8; G7 measures whether iteration pays, not whether termination is honest, which no metric can establish |
| CAP-8 | WP-11, WP-12, WP-13 | Consumer-level integration cases |
| CAP-9 | WP-03, WP-09, WP-13, WP-15 | Independent metrics/statistics and qualification |
| CAP-10 | WP-02, WP-04, WP-05, WP-06, WP-14, WP-15 | Provenance, offline replay, costs, failure injection |

## Definition of done

Each work package delivers its named artifacts, passes its acceptance cases, updates the relevant contract if a decision changes, and records limitations. Model-quality claims require independent evidence; mocks prove orchestration only. No package is complete merely because a demo image looks plausible.

The release includes implementation, typed result validation, offline setup, policy/ontology/configuration, provenance, tests, downstream examples, and the qualification report. A research release may be done while reference qualification fails. Product changes discovered during implementation return to the BMAD memlog and contract before downstream packages silently adopt them.

Immediate starting order: WP-01 policy fixtures, WP-02 model feasibility, and WP-03 independent measurement; then WP-04/05 baseline. M0 provides realistic staffing and calendar estimates. No delivery date is implied by this specification.
