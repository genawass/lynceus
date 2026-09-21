---
id: SPEC-lynceus
companions:
  - annotation-policy.md
  - architecture.md
  - output-contract.md
  - evaluation.md
  - delivery-plan.md
  - failure-modes.md
sources:
  - ../../sources/lynceus-intent.md
---

> **Canonical contract.** This specification and its six companions define what to build, test, and qualify. Its append-only `.memlog.md` records decisions; changes must be rederived through BMAD's `bmad-spec`. Quality targets below are proposed release criteria, not measured performance.

# Lynceus delivery specification

## Why

Dataset engineers and detector evaluators need complete, consistent object annotations without image-by-image human labeling or target-specific model training. Lynceus will spend substantial offline compute discovering objects, resolving their extent and granularity, assigning defensible general labels, and exposing uncertainty. It will support training-set bootstrapping, human-label audits, and qualified automated reference comparisons while preserving the distinction between a model-generated annotation and independently established ground truth.

## Capabilities

- **CAP-1**
  - **intent:** An operator can annotate one image autonomously in an offline environment.
  - **success:** A locally staged bundle processes a valid image with network access disabled and emits a valid result or typed failure without prompts; interrupted work can resume safely.
- **CAP-2**
  - **intent:** Objects are annotated consistently at a declared level of granularity.
  - **success:** Whole objects, parts, groups, stuff, depictions, and reflections follow the versioned policy; every exclusion or unresolved interpretation is inspectable.
- **CAP-3**
  - **intent:** The system discovers relevant objects throughout the image, including small, low-contrast, occluded, and unfamiliar instances.
  - **success:** Required search coverage is recorded, weak discoveries are preserved until assessed, and the discovery and challenge-stratum gates in `evaluation.md` are measured without treating raw proposals as detections.
- **CAP-4**
  - **intent:** Distinct objects receive tight boxes without duplicate annotations or inappropriate merging.
  - **success:** Output coordinates satisfy the geometry contract; duplicate, overlap, part, and group fixtures pass; box quality is evaluated against compatible visible-extent references.
- **CAP-5**
  - **intent:** Each object receives a defensible general class, a broader supported label, or explicit uncertainty.
  - **success:** Labels resolve to a versioned ontology; unsupported specificity is removed; an unknown class cannot erase an otherwise plausible object or enter a supervised export as a known class.
- **CAP-6**
  - **intent:** Consumers can distinguish uncertainty about existence, count, extent, class, granularity, and missing objects.
  - **success:** Results expose separate uncertainty and provenance, preserve competing hypotheses, abstain automatically where needed, and never present uncalibrated scores as correctness probabilities.
- **CAP-7**
  - **intent:** The system searches for missed objects and explains why search ended.
  - **success:** Blind and residual audits include the whole image and areas inside existing boxes; termination records saturation, exhaustion, interruption, or failure, without claiming completeness.
- **CAP-8**
  - **intent:** Consumers can use annotations for training, human-label auditing, and automated reference comparisons with appropriate uncertainty handling.
  - **success:** Each export follows `output-contract.md`; uncertain or unsearched regions do not silently become negatives; auditing and evaluation disclose reference uncertainty and excluded coverage.
- **CAP-9**
  - **intent:** A release can establish where its annotations are sufficiently reliable for a declared use.
  - **success:** A frozen independent evaluation produces the metrics, uncertainty intervals, subgroup evidence, ablations, leakage assessment, and pass/fail/inconclusive decisions in `evaluation.md`.
- **CAP-10**
  - **intent:** Operators can reproduce, inspect, validate, and recover an annotation run.
  - **success:** The delivery contains a runnable offline bundle, pinned model/dependency manifests, evidence artifacts, stage/resource logs, a validator, and a reproducibility/resumption report.

## Constraints

- Runtime inference and evaluation require no network and no per-image human intervention; missing local dependencies cause explicit failure.
- No training, fine-tuning, dataset-level prompt optimization, or calibration on target images or labels; within-image adaptive search follows frozen rules and batch images remain independent.
- No image-specific category list is required. General pretrained knowledge and a fixed annotation policy are allowed under assumption A2.
- Additional compute is permitted without a latency objective; optional operational limits must produce partial results and explicit incompleteness rather than fabricated success.
- Only source-image evidence supports final labels and geometry; generated detail and hidden-extent guesses cannot become reference annotations.
- A required route failure, incomplete scan, or exhausted budget prevents a saturated search status.
- Every output carries the policy, ontology, model, prompt, configuration, and schema identities needed to interpret it.
- Search saturation, ensemble agreement, or a qualifying benchmark score cannot certify correctness or completeness for an arbitrary individual image.
- Precision, recall, useful coverage, and unresolved output must be evaluated together; dropping difficult images or emitting unlimited uncertain boxes cannot satisfy qualification.

## Non-goals

- A universal guarantee of complete ground truth from one arbitrary image.
- Video tracking, 3D reconstruction, amodal boxes, OCR datasets, or exhaustive component-level annotation in the first delivery.
- Target-specific learning, online services, real-time latency, a hosted web application, or human annotation/correction workflows.
- Treating agreement with automatically generated references as unbiased detector accuracy on unlabeled data.

## Success signal

- An engineer runs the delivered bundle on a new image while disconnected and obtains inspectable annotations, uncertainties, and a defensible stop reason without a human decision.
- A separate qualification report demonstrates which declared use/domain gates pass, fail, or remain inconclusive; training and evaluation consumers preserve those limits rather than silently treating every output as ground truth.

## Assumptions

- **A1:** Delivery starts as a local Python CLI/library for still images; batch execution repeats the single-image contract.
- **A2:** Generic pretrained weights and dependencies may be staged before deployment; “no prior knowledge” means no image-specific instructions or class inventory.
- **A3:** Existing independent human annotations may be used outside runtime for evaluation and external calibration, with target data excluded.
- **A4:** General photographs are the first qualification candidate; unfamiliar domains can be processed but do not inherit that qualification.
- **A5:** The initial architecture uses Python/PyTorch workers and local files/SQLite; exact dependencies are frozen after feasibility.
- **A6:** Qwen3-VL, WeDetect-Uni / WeDetect-Anything, OWLv2, SAM 3, and an additional detector are candidates to
  benchmark, not required winners. OWLv2 is the implemented prompted baseline; WeDetect is the preferred next
  class-independent proposal candidate under memlog D22, conditional on a same-data ablation and on GPL-v3
  distribution compatibility.
- **A7:** Numeric gates, sample sizes, initial policy details, and work-package ordering are proposed delivery defaults, not user-approved performance promises.
- **A8:** The default policy includes distinguishable removable clothing/accessories, treats intrinsic components as parts, and tags reflections/depictions separately; domain-specific conventions may replace it before qualification.

## Open Questions

- **Q1:** Which deployment domain and annotation convention should receive the first qualification claim? Resolve by M0; foundation work can begin with the supplied default policy.
- **Q2:** Which hardware, storage envelope, and weight licenses are available? Resolve through M0 measurement before choosing and distributing the model bundle.
- **Q3:** Which error and retained-coverage thresholds are acceptable for each downstream use? Proposed gates are in `evaluation.md`; freeze before locked evaluation.
- **Q4:** Are existing external human-labeled evaluation/calibration sets acceptable outside inference? Default A3 permits them; without them, empirical reference qualification remains unavailable.
