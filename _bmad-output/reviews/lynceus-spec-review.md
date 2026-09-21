# Lynceus specification review

Date: 2026-09-19. Scope: specification quality and delivery readiness, not model performance or implemented software. Method: local BMAD 6.11.0 [bmad-spec](../../_bmad/skills/bmad-spec/SKILL.md), headless express extraction from the supplied discussion.

## Verdict

**Coherence: PASS. Preservation: PASS. Engineering kickoff: READY for the ordered M0 work. Reference-use qualification: NOT ESTABLISHED.**

The [contract](../specs/spec-lynceus/SPEC.md) contains 10 stable capabilities and six companions. The delivery plan provides 15 work packages across six milestones, including dependency ordering, outputs, owner roles, Given/When/Then acceptance criteria, and capability coverage.

The system is not implemented, no image has been annotated, no selected model bundle has been exercised, and no quality benchmark has run. Gate values are proposed targets. This review must not be cited as evidence of detection accuracy or delivery qualification.

## Coherence check against BMAD Spec Law

| Rule | Finding |
|---|---|
| Capability intent and success | All 10 capabilities have both, with inspectable or measurable outcomes |
| WHAT separated from HOW | Kernel states capabilities; model/architecture decisions live in a companion |
| Design-bending constraints | Offline/no-target-learning/automation, evidence limits, termination, and downstream semantics constrain implementation |
| Explicit non-goals | Universal guarantees, target learning, hosted UI, video/3D, and unbiased metrics from pseudo-references excluded |
| Testable success signal | Offline end-to-end delivery and separately reported qualification are demonstrable |
| Stable unique IDs | CAP-1 through CAP-10 unique; each maps to work packages and verification |
| Source preservation | Claim-by-claim mapping below preserves user requirements and proposal limits |
| Lean contract | Kernel routes policy matrices, architecture, metrics, work packages, and failure cases to content-named companions |

## Preservation check

Source: [extracted user intent and discussion](../sources/lynceus-intent.md). The proposal's defaults are not mislabeled as explicit user approval.

| Source claim | Contract location |
|---|---|
| U1 single-image automatic annotation, tight boxes/general labels | CAP-1, CAP-3–6; annotation-policy; output-contract |
| U2 offline, no human inference, no target-trained model, unrestricted compute | Constraints; architecture offline execution/search controls; evaluation leakage limits |
| U3 training, human-label audit, detector evaluation | CAP-8; output-contract profiles; WP-11–13 |
| U4 BMAD delivery specification | Local skill/provenance; derived kernel/companions; append-only memory; delivery-plan |
| U5 completeness, granularity, naming/uncertainty | CAP-2–7; annotation-policy; G1–G7 |
| P1 universal ground truth is unprovable from arbitrary pixels | Constraints/non-goals; qualification boundary; failure-modes |
| P2 fixed policy, parts, stuff/groups, visible extent | CAP-2/4; annotation-policy; WP-01/07 |
| P3 diverse discovery, tiling, candidate union | CAP-3; architecture algorithm; WP-06 |
| P4 instances, masks/boxes, conservative labels/unknowns | CAP-4/5; policy and native records; WP-07/08 |
| P5 separate uncertainties and omission audits inside boxes | CAP-6/7; architecture; WP-09/10 |
| P6 adaptive search, explicit saturation, original-pixel evidence | Constraints; architecture stopping rule; search-state invariants |
| P7 provenance and downstream treatment | CAP-8/10; output-contract; WP-11–14 |
| P8 independent evaluation, failure strata, full-image correctness/risk coverage | CAP-9; evaluation; WP-03/15 |
| P9 candidate model stack and discovery-first experiment | Architecture candidate roles; WP-02/03/06; staged ablations |

No load-bearing user claim was dropped. The universal desired outcome is preserved as the objective and explicitly separated from an impossible unconditional guarantee. Narrower first-domain assumptions do not convert the product goal into a claim that every domain is already supported.

## Review findings resolved during derivation

- Prevented apparent recall gains from unbounded uncertain proposals by adding final retained-instance precision and excluding raw proposals from recall.
- Kept failed/abstained images in coverage and recall denominators; empty accepted sets cannot score 100% precision.
- Defined partial-label export semantics and rejected positive patches that silently turn surrounding unknown pixels into negative examples.
- Distinguished a completed single-pass baseline from exhaustive saturation with `profile_complete`.
- Clarified that target-data prompt tuning is forbidden while adaptive per-image queries under frozen rules remain allowed.
- Kept unknown benchmark/pretraining overlap explicit rather than promising that foundation models never saw target images.
- Added image-cluster statistical treatment, multiple-gate correction, and inconclusive handling for degenerate/undersupported intervals.
- Kept workflow metadata out of product assumptions and maintained explicit source authority for inferred defaults.

## Open decisions and release consequences

| Question | Working assumption | Required resolution point | Consequence if unresolved |
|---|---|---|---|
| Q1 domain and annotation convention | General photographs; declared visible physical-entity policy | M0 | No domain-specific reference claim |
| Q2 deployment hardware/storage/weight terms | Local Python model workers; measure candidate bundle | M0 | No runnable-bundle or capacity promise |
| Q3 acceptable error/retained coverage | Proposed G1–G8 thresholds | Before locked evaluation | Report research metrics; withhold use qualification |
| Q4 independent human-labeled external evaluation allowed | Allowed outside inference; target data excluded | M0 | Engineering evidence only; no empirical reference qualification |

These questions do not block delivering this specification or starting policy/schema/model-feasibility work. They prevent unsupported later promises. There is no approval request or implementation permission gate in this document.

## Validation performed

Static checks parse the kernel frontmatter and JSON example, verify every companion/source/local document link, confirm unique CAP/work-package IDs, check full capability-to-work coverage, and verify copied BMAD resource hashes. The two BMAD configuration resolvers execute successfully in offline mode. These checks validate the specification package only; they are not product tests.

## Wrapper-only content

Conversational greetings, research chronology, and repeated summaries were excluded from the contract. BMAD activation/configuration, copy provenance, and this review remain process artifacts outside `companions:`. No `stories.yaml` was produced: the invoked skill's headless mode excludes interactive story dispatch checkpoint selection; the delivery work packages still specify implementable slices.

Updates must append decisions to the spec's `.memlog.md`, rederive the kernel/companions through `bmad-spec`, and repeat both validation passes. Do not treat this report as permanent approval of future contract changes.
