# Failure modes and required behavior

Contract coverage: CAP-1, CAP-2, CAP-3, CAP-4, CAP-5, CAP-6, CAP-7, CAP-8, CAP-9, CAP-10. These cases are acceptance obligations and qualification probes, not claims that every failure can be detected from a single image.

| ID | Failure mode | Required response | Verification |
|---|---|---|---|
| FM-01 | All models miss the same object | Preserve the limit of the completeness claim; measure shared misses on independent references | Blind/semantic ablations and G2/G3; saturation cannot certify absence |
| FM-02 | Tiny/low-contrast object lacks enough pixels | Search original-resolution crops; retain supported uncertainty; do not invent detail | Size/contrast strata, original-pixel evidence |
| FM-03 | Semantic inventory omits a class | Class-independent routes and spatial coverage still run | Fixture with an inventory omission and surviving proposal |
| FM-04 | Texture, shadow, or glare looks like an object | Seek source-pixel and context support; reject or retain uncertain with reason | Confuser set; G1/G6 |
| FM-05 | Touching objects merged | Preserve count ambiguity and group hypotheses; retry separation where evidence permits | Merge/split cases and rates |
| FM-06 | One object split into fragments | Reconcile same-instance evidence; preserve disconnected supported extent | Fragment/occlusion fixtures |
| FM-07 | Part suppressed or exported as independent by accident | Apply declared granularity and relations, not containment-only rules | Person/backpack, wheel/bicycle fixtures |
| FM-08 | Tile-edge duplicate or clipped boundary | Map to source coordinates and inspect an expanded crop | Border and inverse-transform tests |
| FM-09 | Mask leaks into another object or drops a thin extension | Recheck competing masks; retain boundary uncertainty | Boundary challenge cases and G5 |
| FM-10 | Model confidence is confidently wrong | Treat scores as evidence; externally measure risk and calibration | Calibration/risk-versus-coverage report |
| FM-11 | Same checkpoint agrees across prompts | Record common provenance; no independence assumption | Evidence-lineage validator |
| FM-12 | More passes create more false objects | Assess final retained precision and marginal gain together | G6/G7 and compute ablations |
| FM-13 | Unknown class is dropped | Retain an uncertain unknown instance with geometry evidence | Unknown-object fixture and coverage metrics |
| FM-14 | Broad labels game correctness | Freeze acceptable useful parent labels; exclude generic entity/object from useful coverage | Evaluator mapping tests and G4 |
| FM-15 | Images or objects are selectively omitted from evaluation | Include failed/abstained targets in recall/coverage; publish exclusions | Denominator audit and run-manifest reconciliation |
| FM-16 | Out-of-domain input appears plausible | Withhold inherited qualification; annotate under research profile | Declared domain rules and shifted-domain evaluation |
| FM-17 | Pretraining/benchmark overlap is unknown | Record disclosure limits; do not promise unseen-image generalization | Leakage assessment |
| FM-18 | Unresolved regions become training background | Ignore with supported consumer semantics or reject export | Consumer loss-mask test |
| FM-19 | Human/model disagreement is treated as a proven human error | Retain both interpretations and label discrepancy type | Audit integration tests |
| FM-20 | Detector is penalized for finding something absent from auto-reference | Preserve unresolved matches and report pseudo-reference limitations | Comparison fixtures and coverage report |
| FM-21 | Missing local model or accidental download | Fail preflight; no runtime network fallback | Network-denied clean-environment test |
| FM-22 | Worker OOM, crash, or malformed model output | Bounded retry with logged changes; partial/failure state if required work remains | Failure injection and schema checks |
| FM-23 | Proposal cap, cycle, or resource limit stops search | Expand/revisit where permitted; otherwise explicit partial outcome | Saturation and budget invariants |
| FM-24 | Input corruption, decompression overflow, unsupported frames | Typed rejection before unsafe allocation or silent conversion | Decoder boundary tests |
| FM-25 | Image text attempts to control the pipeline | Treat it as visual content; never execute model output or alter policy from it | Adversarial image-text fixture |
| FM-26 | Artifact tampering, path traversal, or stale cache | Validate hashes/paths/dependency identities and refuse reuse | Manifest and cache invalidation tests |

When a missing-object failure is not detectable from the single image, the system cannot invent a region-level uncertainty marker for it. That is why global completeness remains unverified even when all known uncertain regions are resolved. The independent benchmark is the mechanism for estimating these failures, not a claim that inference can always identify them.
