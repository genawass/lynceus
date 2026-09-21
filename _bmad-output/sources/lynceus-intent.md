# Lynceus source intent

Date: 2026-09-19. Source: this project's user/assistant conversation. This is a claim-preserving extraction, not a verbatim transcript. User requirements and proposed design choices have different authority.

## User requirements

- **U1:** Given one image with no prior knowledge of its content, automatically produce a complete, trustworthy object-detection annotation: every object of interest tightly localized and assigned a general class label or uncertainty.
- **U2:** Inference is offline, requires no human in the loop, and uses no model trained on the target data. Per-image compute is unconstrained; large models, repeated passes, tiling, and verification are allowed.
- **U3:** Outputs should support auditing human annotations, evaluating detectors on unlabeled data, and bootstrapping training sets.
- **U4:** Build a delivery specification using BMAD.
- **U5:** The three coupled difficulties are exhaustive discovery including small/low-contrast objects, appropriate object granularity, and correct naming with uncertainty.

## Proposed design from the discussion

- **P1:** Universal ground truth cannot be guaranteed for arbitrary single images. Information loss, ambiguous boundaries, and shared model blind spots remain despite additional compute.
- **P2:** Use a fixed policy: distinguish discrete entities, preserve parts internally, separate stuff and groups, and describe visible rather than invented occluded extent.
- **P3:** Combine class-independent proposals, semantic inventory, and concept search; use exhaustive overlapping spatial/multiscale passes and preserve their candidate union.
- **P4:** Resolve duplicates, separate overlapping instances, parts, and group hypotheses; use segmentation as evidence for boxes; select the most specific defensible general class, retaining unknown objects.
- **P5:** Separate existence, count, boundary, class, granularity, and omission uncertainty. Search for omissions using blind rediscovery, residual inspection, exemplar search, and disagreements, including regions inside existing boxes.
- **P6:** Use an adaptive search controller with an explicit saturation rule. Saturation and repeated agreement are not proof of completeness. Evidence must be supported by original pixels.
- **P7:** Preserve accepted and uncertain records, unresolved regions, policy, coverage, model versions, and provenance. Training ignores unresolved regions; auditing reports discrepancies; automatic reference evaluation exposes its limitations.
- **P8:** Freeze the system and evaluate it against independent annotated images, measuring omissions by challenge strata, false objects, localization, classification, fully correct images, and error versus retained coverage.
- **P9:** Candidate local building blocks are Qwen3-VL, OWLv2, and SAM 3; model composition and quality still need empirical validation. Begin experiments with discovery and omission reduction.

## Interpretation for delivery

The user asked to develop the proposal into a spec, but did not separately approve particular models, deployment hardware, numeric quality targets, export conventions, ontology, qualification domain, or delivery dates. Those choices must remain explicit assumptions or qualification decisions, not claims of user approval or achieved performance. No supplied image, benchmark, repository implementation, or performance measurement exists yet.
