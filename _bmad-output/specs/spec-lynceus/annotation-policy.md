# Annotation policy

Contract coverage: CAP-2, CAP-4, CAP-5, CAP-6, CAP-8. Proposed policy ID: `physical-entities-visible-v1`. Policy changes require a new ID and invalidate incompatible qualification results.

## Scope and granularity

The default target is each visually distinguishable discrete physical entity in the depicted scene. Relevance does not depend on salience, familiarity, or a VLM mentioning its class. Search must attempt small and low-contrast entities; an arbitrary pixel-size threshold cannot silently remove them. Qualification declares the measurable size range and reports unresolved or below-resolution evidence separately.

| Case | Default treatment | Example |
|---|---|---|
| Individually distinguishable entity | Separate instance | Person, chair, bottle, tree, animal |
| Independently manufactured/worn/carried item | Separate instance when distinguishable; relation to wearer/carrier | Backpack, shirt, shoe, handheld phone |
| Intrinsic or assembled component | Preserve as `part` with `part_of`; exclude from default whole-object export | Hand, bicycle wheel, shirt button |
| Detached component | Instance if independently distinguishable | Wheel lying beside bicycle |
| Several resolvable touching objects | Separate instances even with overlapping boxes | Stacked distinguishable plates |
| Inseparable pile or crowd | Group with unresolved count; no fabricated instance boxes | Distant crowd, indistinguishable screws |
| Continuous material or background | Optional `stuff`; exclude from instance export | Sky, grass, water, road surface |
| Shadow, glare, printed texture | No physical instance; retain conflicting evidence if uncertain | Tree shadow, painted wood grain |
| Reflection | Tag `reflected`; retain separately in native output and exclude from default physical-scene export | Person visible in a mirror |
| Object depicted on a screen or print | Tag `depicted`; retain separately and exclude from default physical-scene export | Cat on a poster; poster itself is an instance |
| Partial or image-edge object | Include visible portion; mark occluded/truncated | Half a cup at the image edge |
| Semantic ambiguity | Preserve competing interpretation and mark uncertain | Pole versus narrow trunk |

Relationships describe `part_of`, `worn_by`, `carried_by`, `supported_by`, and `depicted_on` only when supported. Overlap or containment alone cannot imply identity, part membership, or exclusion. Derived exports select the declared policy layer, never an unrecorded interpretation.

Default building-level annotation treats a building as an entity and its integrated architectural elements as parts. Removable furnishings are entities. Plants are whole entities by default; leaves and branches are parts. These are explicit design defaults requiring domain-specific replacement when the use case differs.

## Geometry

- Normalize EXIF orientation before inference. Persist original-byte hash, oriented-pixel hash, original dimensions, normalized dimensions, and the orientation transform.
- Box format is `[x_min, y_min, x_max, y_max]` in continuous pixel-edge coordinates of the normalized original-resolution image. A box covers the half-open interval `[x_min, x_max)`; its coordinates satisfy `0 <= x_min < x_max <= width`, likewise for y.
- A box encloses the supported visible extent, including supported thin extremities and disconnected visible portions of one object. Do not infer unseen occluded extent.
- Internal masks retain disconnected components when they belong to the same instance. A mask is a hypothesis, not proof of membership.
- Transform every crop/tile prediction back to source coordinates before reconciliation. Preserve crop-edge truncation flags and revisit an expanded crop before accepting the boundary.
- A refined boundary replaces a proposed one only as a claim about the same instance, never as a new instance. Refinement may tighten or loosen a box; it may not move it onto a different object, and the proposed geometry stays in evidence so the change is inspectable.
- Where a refiner and the proposal that fed it disagree about extent beyond a declared tolerance, that disagreement is boundary uncertainty. The refined box may be the best estimate, the proposed box is retained as an alternative, and the boundary dimension is unresolved rather than silently overwritten. Two models agreeing on where an object is and disagreeing on how far it extends is exactly the case the uncertainty dimensions exist to carry.
- Support alternative boxes or masks when evidence does not identify a unique boundary. A best-estimate box may accompany `boundary=unresolved`; downstream training cannot accept it as precise geometry.
- A mask outside image bounds, nonfinite coordinates, zero area, or noninvertible transform fails validation.

## Class policy

Use a versioned local ontology with stable IDs, canonical names, synonyms, and broader-parent relationships. Begin with broad physical classes; M0 freezes the initial ontology and evaluator mapping. Free-text descriptions remain evidence and cannot silently create a new supervised class ID.

Choose the most specific general label supported by the crop and context. Do not require species, brand, material, or functional identity that the pixels cannot establish. Fall back to a supported broader class. If no useful class is supported, use the reserved `unknown_object` ID and preserve the instance as uncertain. Generic `object`/`entity` and unknown labels do not count toward useful-label coverage.

Two views of one instance may disagree about its class without either being uncertain. Conservative naming runs per observation and cannot reach that case: each view had a clear winner and nothing to fall back from, so the disagreement exists only across views. Where such views describe the same instance, the most specific claim they jointly support is their common ancestor, and they are reconciled under it when that ancestor is a useful class. Where the ancestor is not useful they remain separate objects, and `unknown_object` never merges into a class, because it asserts that no useful class is supported and folding it into one would manufacture a claim no view made.

An ontology extension needs a versioned change and a compatible evaluation mapping; it cannot be invented per image. The evaluator declares which broader labels are acceptable per reference class before test predictions are seen. Matching an animal to “entity” cannot satisfy label correctness.

### The `useful` flag is constrained by reference granularity

Each ontology class carries a `useful` flag deciding whether it counts toward useful-label coverage. That flag is not free: a class may be marked useful only if, within the declared domain, it excludes at least one other class the reference distinguishes. A parent that covers every reference class an annotator is likely to meet is a tautology in that domain, not a useful label, whatever its name.

Declaring the flag is therefore a domain-specific act performed with the evaluation mapping, before predictions are seen, and recorded in the ontology. `entity`, `object`, `stuff`, and `unknown_object` are never useful. A class that is useful in one domain may be a tautology in another: `vehicle` distinguishes in a general-photograph domain and does not in a road-user domain where every annotated reference is a vehicle or a person.

Because a useful broad label still costs specificity, useful-label coverage is always reported split into a specific share and a generic share against the reference granularity, as defined in [evaluation.md](evaluation.md). A release cannot satisfy the coverage gate on generic labels alone.

### Classes excluded by convention

An ontology may declare `excluded_by_convention`: entity kinds the deployment's annotation convention does not annotate at all, such as buildings and vegetation in a road-user deployment. This is a boundary of the declared convention, not a measured negative. Absence of such an annotation is not evidence about that entity kind, references containing it are not eligible, and predictions carrying such a class are handled by the declared category scope in [evaluation.md](evaluation.md) rather than scored as errors. Removing a class from an ontology changes the qualification identity and requires a new ontology version.

## Qualification boundary

The same pixels can support several reasonable annotation conventions. Compare only references with compatible granularity, visible-extent geometry, depiction policy, and class mapping. A mismatch is a policy incompatibility, not necessarily a model error.

Benchmark cases with unresolved reference ambiguity must be identified independently of predictions and reported separately. Inference-time uncertainty is never a reason to remove a resolvable reference object from recall denominators.
