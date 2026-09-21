# Output and integration contract

Contract coverage: CAP-1, CAP-4, CAP-5, CAP-6, CAP-7, CAP-8, CAP-10. Native schema version: `lynceus.annotation/1.0`. M1 must deliver JSON Schema plus cross-field validation implementing this contract.

## Artifact layout

```text
run/
  annotation.json             canonical result, including partial outcomes
  manifest.json               input, policy, ontology, models, prompts, config, environment
  events.jsonl                stage actions, failures, retries, resource usage, stop reason
  evidence/candidates.jsonl   all hypotheses, including rejected ones and their lineage
  evidence/masks/             source-coordinate masks and alternative masks where available
  evidence/coverage.json      planned/completed jobs, crop geometry, route results, scan coverage
  evidence/regions.json      unsearched and unresolved region geometry
  overlay.png                nonauthoritative visualization, never a source image
  report.html                local read-only explanation of results; no external dependencies
```

Large masks may use compressed binary artifacts with format, dimensions, and digest recorded in the manifest. File references must be local, relative to the run root, and remain inside it. Every referenced artifact has an integrity hash; manifest hashing excludes its own hash field. A consumer must fail validation on missing, altered, or escaping references.

The report explains unknown classes, unresolved boundaries, excluded policy layers, and the stop reason. It does not request annotations or human approval; all inference decisions are automatic.

## Required fields and invariants

| Field | Meaning and constraints |
|---|---|
| `schema_version`, `run_id` | Versioned contract and immutable run identity |
| `image` | Source hash, normalized-pixel hash, original/normalized sizes, orientation transform, color/alpha conversion |
| `policy`, `ontology` | IDs, versions, hashes, and local definitions used for this run |
| `manifest_ref` | Local immutable model/prompt/config/environment manifest |
| `qualification` | Applicable qualification ID or null, declared domain, and status `unqualified`, `eligible`, or `ineligible`; eligibility describes bundle/use/domain conditions, never per-image truth |
| `execution` | `completed`, `interrupted`, or `failed`; typed reason and failure references |
| `search` | Profile ID and status `saturated`, `profile_complete`, `budget_exhausted`, `interrupted`, or `failed`; mandatory-job counts, stable audit rounds, coverage reference, and stop-rule configuration |
| `objects` | Accepted and uncertain retained hypotheses with unique IDs; groups/parts/stuff tagged by kind |
| `regions` | Unsearched, unresolved, and policy-excluded geometry with reasons and evidence links |
| `summary` | Counts by disposition/kind/class, unresolved counts, coverage metrics, resource summary, and a required `calibration` block; recomputable counts must equal underlying records |
| `artifacts` | Logical names, relative paths, formats, and integrity hashes |

Every object record contains:

- `id`; `kind` in `instance`, `part`, `group`, `stuff`; `status` in `accepted`, `uncertain`; and `scene_layer` in `physical`, `reflected`, `depicted`.
- `bbox_xyxy` satisfying [the geometry policy](annotation-policy.md); a best-estimate box is allowed for an uncertain boundary. Optional mask reference and alternative boxes/masks preserve disagreement.
- `label` with ontology ID and canonical name; alternatives reference valid ontology IDs. Unknown is `unknown_object`, never a missing object record.
- `uncertainty` fields `existence`, `count`, `boundary`, `class`, `granularity`, each `supported`, `unresolved`, or `not_assessed`, plus reason codes.
- `evidence_refs` identifying discovery and verification evidence; supported decisions cannot have empty evidence.
- `relations` with relation type, target ID, and supporting evidence; target IDs must exist. `part_of` must be acyclic.
- `occluded` and `truncated` flags, each either null for an unknown state or a boolean supported by `flag_evidence_refs`; a flag asserted without evidence is rejected. `count` for groups is null or an explicit supported estimate/interval, never invented from density.
- `calibrated_estimates`, empty by default; entries define an exact event, probability, applicable calibration artifact, population assumptions, and localization threshold where relevant. An entry is permitted only while `summary.calibration.status` is `applied`.

Where geometry was refined by a model other than the one that proposed it, the candidate store records the proposed box, the refined box, the refiner's identity and its self-reported mask quality where one exists. Refinement is a second model acting on the first model's output, so its ancestry is recorded like any other route and its quality score stays in evidence, never in a field that reads as a probability of correctness.

An `accepted` instance must have all five object uncertainty dimensions supported, a useful supported class, and validated geometry. An unknown class or unresolved dimension makes it `uncertain`. Acceptance does not establish that the image is complete. Rejected candidates are excluded from `objects` and remain in the candidate store with rejection reasons.

Existence, class, and geometry can be supported independently for an uncertain object. Consumers may use those fields for partial supervision only if their loss/evaluation semantics support it; a broad acceptance flag cannot erase unresolved dimensions.

Model logits, objectness, mask quality scores, repeated-prompt agreement, and ensemble votes belong in evidence with model-specific names. They must not be copied into a field named “probability correct.” Calibrated events apply only under the recorded external calibration assumptions. No numeric `probability_all_objects_found` or `is_ground_truth` field is permitted in v1.

### Calibration applicability

`summary.calibration` is required and carries a `status` of `applied` or `inapplicable` with a `reason`. It states whether any probability in this run rests on a calibration fitted elsewhere, so a consumer can distinguish “no estimate was made” from “the estimate is zero.”

`applied` requires a calibration artifact present in the run, hash-verified, and fitted on a population whose declared domain matches this run's ontology and domain. A domain mismatch records `inapplicable` with reason `calibration_domain_mismatch` and emits no numbers: a fit from another domain does not establish reliability here. With no artifact staged, the status is `inapplicable` with reason `no_calibration_artifact_staged`.

No object may carry a `calibrated_estimates` entry unless the status is `applied`, and every entry must reference a calibration artifact recorded in `artifacts`.

## Search and failure semantics

- `execution=completed` can coexist with `search=budget_exhausted`: the application successfully produced a partial artifact. The CLI uses the documented partial-result exit code.
- `search=saturated` requires every mandatory job completed successfully and the stable-round criterion met. It can coexist with unresolved object records or regions.
- `search=profile_complete` is limited to narrower profiles such as the single-pass baseline; it cannot imply exhaustive search or reference qualification.
- Geometric coverage means planned views were processed, not that all objects were seen. Report route-specific covered area and planned/completed counts; never name this metric semantic recall.
- A nonempty unsearched region or a failed required model route prevents saturation. Missing proposals are not evidence of background.
- Failed/aborted stages retain completed artifacts and produce an explicit terminal event. Resume preserves lineage; materially changed inputs/configuration produce a new run.
- An empty object list is valid and means “no retained objects,” not certified absence. It must retain the same search and uncertainty metadata as other results.

## Example object

This is illustrative contract data, not output from a model or a complete run artifact:

```json
{
  "id": "obj-0007",
  "kind": "instance",
  "status": "uncertain",
  "scene_layer": "physical",
  "bbox_xyxy": [148.0, 93.0, 174.0, 121.0],
  "mask_ref": "evidence/masks/obj-0007.rle.json",
  "alternative_boxes": [],
  "label": {"id": "unknown_object", "name": "unknown object"},
  "label_alternatives": [{"id": "cup", "name": "cup"}, {"id": "bowl", "name": "bowl"}],
  "uncertainty": {
    "existence": "supported",
    "count": "supported",
    "boundary": "supported",
    "class": "unresolved",
    "granularity": "supported",
    "reasons": ["insufficient_class_detail"]
  },
  "evidence_refs": ["ev-0042", "ev-0081"],
  "relations": [],
  "occluded": false,
  "truncated": false,
  "flag_evidence_refs": ["ev-0042"],
  "count": null,
  "calibrated_estimates": []
}
```

## Downstream profiles

| Use | Included output | Required treatment of uncertainty |
|---|---|---|
| Research/native | All retained hypotheses, evidence, unresolved regions, coverage | No qualification inferred; full fidelity |
| Training | Policy-eligible accepted instances with supported classes/geometry | Unknowns, groups, uncertain objects, unsearched regions, and unqualified negative space are ignored or handled by explicit partial-label losses |
| Human-label audit | Matched labels and missing/extra/class/geometry/granularity discrepancies | Report model and human evidence symmetrically; discrepancies are suspected issues, never automatically proven human errors |
| Detector comparison | Matched detector/reference outputs under a frozen policy mapping | Report pseudo-reference agreement, ignored cases, and evaluated coverage; do not call it unbiased detector accuracy |

### Training export

Native output is authoritative. A COCO-compatible view converts xyxy to xywh, emits stable integer class mappings, and ships an ignore/coverage sidecar with a documented loader contract. Standard COCO `iscrowd` alone does not represent arbitrary unknown-class or unsearched regions.

Before export, verify that the target training consumer supports the required ignore semantics. If it does not, fail with an actionable typed incompatibility. An explicitly chosen positive-only mode may emit patches only when all other patch regions are either resolved or ignored by the consumer; a positive box never certifies surrounding background. Do not export partially searched images as ordinary fully labeled negative examples.

### Human-label audit

Use one-to-one matching after coordinate, class, and policy normalization. Distinguish potential human omissions, potential model false positives, class disagreements, extent disagreements, duplicate labels, and policy incompatibilities. A human label unmatched by Lynceus remains evidence of a possible model miss. Run annotation independently before loading human labels to prevent anchoring. Deliver a machine-readable discrepancy file and a local read-only overlay report.

### Detector comparison

Run Lynceus independently of the detector being evaluated. Uncertain class/geometry matches become unresolved comparisons, not automatic detector errors. Report accepted-reference recall, matched-label/box agreement, unmatched predictions, excluded region area, excluded records, and the qualification context with their exact denominators.

Default mode must not emit ordinary ground-truth mAP for an automatic reference. An explicit `--allow-pseudo-reference-metrics` option may compute named `pseudo_reference_AP`/`pseudo_reference_recall` with a conspicuous provenance label and coverage report. Ordinary benchmark AP requires independently labeled, compatible references and is provided by the independent evaluation path. No image without annotated references can yield an empirical true missed-object rate.
