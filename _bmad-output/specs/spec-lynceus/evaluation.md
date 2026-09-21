# Evaluation, qualification, and release evidence

Contract coverage: CAP-3, CAP-4, CAP-5, CAP-6, CAP-7, CAP-8, CAP-9, CAP-10.

All numeric gates here are proposed engineering targets under A7. None has been measured or approved as a performance guarantee. Missing evidence produces `inconclusive`; a failed gate produces `fail`. Neither can be relabeled as qualification through a successful demonstration.

## Separate engineering delivery from qualification

- **Engineering pass:** offline execution, policy/schema conformance, failure recovery, provenance, and consumer semantics meet deterministic acceptance tests.
- **Research delivery:** the complete pipeline and evaluation report exist, even if accuracy is insufficient. Outputs remain unqualified.
- **Qualified use:** the frozen model/config/policy bundle meets preregistered quality gates on a declared population and use profile. Qualification does not guarantee any individual annotation or generalize to arbitrary images.
- **Descriptive measurement:** a run of the evaluator on data that cannot support qualification at all — an out-of-domain panel under A4, a finite-category reference, or a panel far below the sizes below. It exercises the software and produces comparable numbers between configurations of this system. It is reported with its disqualifying property named, never as a gate outcome, and never compared against a gate threshold. A descriptive result cannot be promoted to a qualification result by collecting more of the same data; the disqualifying property has to be removed first.

Qualification IDs bind model hashes, policy/ontology, prompts, thresholds, search schedule, hardware/numerical configuration, evaluator version, dataset manifests, domain definition, and consumer profile. Changes require an impact assessment and rerunning affected qualification; model/policy/threshold changes always require a new ID.

## Independent evaluation data

1. Establish the domain and annotation convention before collecting predictions. Use existing independently annotated data where licensing, completeness, and policy permit. No annotator target data enters training, dataset-level prompt selection, threshold tuning, or calibration. Within-image search still generates queries under frozen rules.
2. Split external data into development, optional calibration, and locked evaluation. Split by source and scene where possible; hash-based and near-duplicate checks prevent images or near-identical scenes crossing splits.
3. Select candidate benchmarks such as dense open-world annotations and long-tail instance datasets only after checking annotation coverage. A finite-category or federated-label dataset cannot establish exhaustive open-world recall by treating every unannotated object as absent. The [UVO paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Unidentified_Video_Objects_A_Benchmark_for_Dense_Open-World_Segmentation_ICCV_2021_paper.pdf) explains the consequences of incomplete references.
4. Derive visible boxes from compatible reference masks where available. Annotator and reference must share granularity, depiction/reflection policy, crowd rules, and acceptable class generalization. Record any category/region exclusions before examining predictions.
5. Proposed locked panel floors: at least 500 images and 5,000 eligible instances across at least five independent source families or domains. Each claimed challenge stratum needs at least 100 instances across 30 images. These are coverage floors, not sizes: they say which populations must be represented, and they do not establish that any gate can be reached. The size that decides whether a gate is reachable comes from the power calculation below, and it governs when the two disagree.
6. Before data collection is closed, WP-03 delivers a power calculation that derives the required number of image clusters per gate and per claimed stratum from the gate threshold, the corrected alpha, and a measured or assumed intra-image correlation. It is run with the frozen statistical script on development data or simulation, and its result is part of the frozen protocol. A gate whose required cluster count exceeds the collected panel is declared **structurally inconclusive** before any prediction is scored, and the release says so rather than reporting a bound that could never have cleared. G1 at a 0.99 lower bound and G3 at a 0.85 lower bound on a 30-image stratum are the two most likely to fail this check; discovering that after the locked run is a wasted panel, not a finding.
7. Report small size, low contrast, occlusion, truncation, crowding, unfamiliar classes, unusual domains, and background/texture confusers. Define bins in the frozen manifest: proposed size bins use original-resolution equivalent side length `sqrt(visible area)` of `<16`, `16–32`, `32–96`, and `>=96` pixels. Low contrast uses a preregistered local foreground/background contrast statistic where masks permit; unavailable strata are disclosed.
8. Report benchmark annotation ambiguity and exclusions separately. Resolvable reference instances remain in all recall/coverage denominators even when the system abstains, fails, exceeds budget, or calls them unknown.
9. Record foundation-model training-corpus disclosures and known benchmark overlap. Distinguish “no project training on target data” from “the pretrained model never saw this image,” which generally cannot be guaranteed. Use a holdout with documented provenance for stronger claims where available.

Synthetic fixtures validate geometry, semantics, and failures. They cannot substitute for real-image quality qualification. If independent reference data is forbidden or incompatible, deliver engineering evidence and explicitly withhold empirical accuracy/completeness claims.

## Metrics and matching

Use one-to-one maximum matching under a frozen criterion; duplicates count as unmatched predictions. Report micro aggregates and per-image/domain distributions. Every metric records numerator, denominator, eligibility rules, and exclusions.

Every metric is either **gated** — a named gate decides pass/fail/inconclusive on it — or **descriptive**, reported without a threshold. A descriptive metric can still block a release by being absent; it cannot pass or fail one. The role column is part of the frozen protocol, so a metric cannot acquire a threshold after its numbers are seen.

| Metric | Role | Definition |
|---|---|---|
| Localized instance recall | Gated (G2, G3) | Eligible reference instances matched by final retained `instance` boxes, accepted or uncertain, at IoU >= 0.50 / all eligible reference instances |
| Final instance precision | Gated (G6) | Final retained instance boxes matching unique eligible reference instances at IoU >= 0.50 / all retained instance boxes; accepted and uncertain both count |
| Accepted annotation precision | Gated (G1) | Accepted, policy-eligible instance records with a permitted useful label and a unique reference match at IoU >= 0.75 / all such accepted records |
| Useful-label coverage | Gated (G4) | Eligible reference instances matched by accepted records with permitted useful labels at IoU >= 0.75 / all eligible reference instances. Always reported split into a specific share and a generic share (below) |
| Specific-label coverage | Gated (G4) | The subset of useful-label coverage whose emitted label is the reference class itself or a permitted parent that is not the reference's broadest useful ancestor / all eligible reference instances |
| Localization quality | Gated (G5) | Accepted class-correct matches at IoU >= 0.85 / accepted class-correct matches at IoU >= 0.50; also report full IoU and boundary-error distributions |
| Granularity conformance | Gated (G9) | Retained records whose `kind`, `scene_layer`, and part/group relations agree with the reference's declared granularity, over all retained records matched to a granularity-annotated reference. Counts a part exported as an instance, an instance demoted to a part, a fabricated instance inside an inseparable group, and a physical object emitted as `depicted` or `reflected` |
| Classification accuracy | Descriptive | Label correctness on localized matches, with unknown and broad-parent rates reported separately |
| Duplicate/merge/split rates | Descriptive | Count each failure under fixed instance-matching rules; groups do not count as individual discoveries |
| Image completeness | Descriptive | Fraction of reference-compatible images with no missed eligible instances; separately report fully correct images with no extras, class, or geometry errors |
| Risk versus coverage | Descriptive | Error among accepted records versus fraction of eligible reference instances accepted, across a fixed threshold sweep selected only on development/calibration data. Ungated while Q3 is open; freezing Q3 converts the declared operating point into a gate |
| Unresolved burden | Descriptive | Uncertain-object counts, unresolved groups/counts, unsearched area, excluded instances/regions, and per-image abstention rate |
| Operational cost | Descriptive | Calls, crops, passes, candidates, wall time, GPU hours, peak VRAM/RAM, and artifact bytes per image |
| Miss attribution | Descriptive | Every unmatched eligible reference assigned to the mechanism that lost it: below threshold, merged away, merged with a neighbour, localized but below the match criterion, or never proposed at any score in any view |
| Pruner selectivity | Descriptive | For a scorer that removes retained boxes, the rate at which it removes matched boxes against the rate at which it removes unmatched ones, across a threshold sweep |
| Boundary headroom | Descriptive | For each reference no retained box matched, the best IoU any recorded observation reached, at any score, in any view. Separates misses a better choice among existing boxes could recover from misses that need geometry no view produced |

### Granularity conformance and CAP-2

CAP-2 is the policy capability, and synthetic fixtures cannot qualify it any more than they qualify detection. Granularity conformance is its empirical measurement, so a reference is usable for G9 only if it annotates granularity at all: parts distinguished from wholes, groups marked as groups, depicted and reflected content tagged. Most references do not. Where the reference lacks a granularity annotation, G9 is inconclusive for lack of evidence and the release says which policy rows went unmeasured — it does not fall back to fixtures and call the capability verified.

### Category scope on a finite-category reference

A reference that annotates only some classes cannot score a correct prediction of an unannotated class as a false positive. Where such a reference is used under the documented-mapping route above, the frozen manifest declares a **category scope**: the set of ontology classes the reference annotates, with a written reason, registered before predictions are examined.

Predictions labelled outside that scope are removed from precision denominators and counted and reported separately as out-of-scope predictions. Predictions carrying `unknown_object` or `entity` make no category claim and therefore stay in every precision denominator, so abstention cannot buy precision. Recall denominators are unaffected: scope narrows what the system may be blamed for, never what it must find.

A scoped result is a descriptive measurement under the tier above unless the scope is the whole declared domain convention. Two scoped results are comparable only when their scopes are identical.

Raw pre-verification proposals do not count toward final recall. Groups cannot earn instance recall from one box covering many objects. Precision on an empty accepted set is undefined, never 100%; useful coverage is zero. Qualification recall/coverage includes failed runs as zero retrieved/accepted instances rather than dropping those images.

### Building an exhaustive reference panel

A finite-category reference bounds what can be learned: an unmatched box is not evidence of a
false box, so precision is a lower bound of unknown tightness, calibration cannot be fitted because
an unmatched box is not a scoreable error, and granularity conformance has nothing to score
against. One small exhaustively annotated panel removes all three limits at once, and no amount of
additional finite-category data removes any of them.

**Exhaustive** here means annotated to the declared policy's own standard: every visually
distinguishable discrete physical entity the convention includes, not every entity a particular
category list happens to name. The panel declares its policy and ontology by ID like any run, and
the ontology's `excluded_by_convention` entries are the only permitted absences.

The protocol has five obligations, and each exists to stop a specific way the panel could be
quietly worthless.

1. **Blind first.** The reference is annotated without the system's predictions visible. Reviewing
   predictions produces agreement, not an independent reference, and the anchoring is invisible in
   the result. Predictions may be loaded only after the reference is frozen.
2. **Swept, not scanned.** Completeness is a property of the procedure, not of the annotator's
   confidence. The image is divided into overlapping source-resolution review tiles, every tile is
   visited, and the panel records which tiles were visited by whom. An unvisited tile makes the
   image non-exhaustive, and it is excluded rather than counted as empty.
3. **Ambiguity recorded, not resolved by fiat.** An entity whose extent or granularity the policy
   does not settle is marked unresolved and kept. Those instances stay in recall denominators and
   leave precision denominators, because a prediction cannot be scored wrong against a reference
   that does not know its own answer.
4. **Granularity annotated.** Parts, groups, and depicted or reflected content are tagged as such.
   Without this the panel cannot score G9, and granularity conformance stays inconclusive no matter
   how many boxes it holds.
5. **Leakage identity carried.** Each image records its content hash, scene identity and
   near-duplicate group, so the panel can be checked against any development or calibration data
   already used, including an in-context reference bank.

Size is governed by what it unblocks rather than by the gate floors, because this panel is not a
qualification panel and cannot become one. Twenty to thirty images is enough to convert precision
from a bound into a measurement, to fit a first calibration, and to score granularity conformance
at all. A gate claim still requires the panel the power calculation sizes.

A panel meeting all five obligations declares no category scope, because there is nothing outside
it to exclude: every retained prediction is adjudicable, and precision becomes a number rather than
a bracket.

### Searching for a prompt vocabulary

The phrasing a model is asked with is configuration, and configuration chosen by hand is chosen
badly: nobody knows whether a model finds more small overhead figures under `person`, `pedestrian`
or `person seen from above` without trying. So the vocabulary is searched rather than written, and
the search is an experiment with the same obligations as any other.

**It is fitted on development data and never on the evaluation panel.** Selecting phrasing against
the locked panel is dataset-level prompt optimization on target data, which the constraints forbid,
and it invalidates the panel. The fitted vocabulary is frozen, and only then is it measured once on
the held-out panel. The development score is not the result; the held-out score is.

**Candidates are enumerated in advance** and recorded with the vocabulary, so a later reader can
see what was considered and not only what won. A vocabulary that beat two alternatives and one that
beat forty are different artifacts.

**Classes interact and the search must respect it.** A prompted detector scores every class against
each region and the emitted label is the winner, so changing how one class is phrased changes which
boxes other classes win. Per-class objectives optimized independently would therefore not compose.
Coordinate ascent over classes, re-scoring the whole vocabulary at each step, is the minimum that
accounts for this; it finds a local optimum and does not pretend to find the best vocabulary.

**Evaluation is exact, not approximate.** A prompted detector's image features and box predictions
do not depend on the text, so they are computed once per view and reused for every candidate, which
leaves only the text tower and the class head to re-run. This is a cost reduction and not an
approximation: the scores are the same ones a full forward pass would produce.

**The gap between development and held-out performance is reported.** Selecting among candidates on
a small panel fits noise as well as signal, and a search that gains on development and loses on the
held-out panel has found noise. Reporting only the winning score would hide exactly that.

A searched vocabulary records what it was fitted on, the objective it maximised, the candidates
considered and the development score of each step, so the search is inspectable rather than a
number with a provenance claim attached.

### Auditing unmatched boxes to recover precision

On a finite-category reference the ambiguity is confined to one place. A box that matches an
eligible reference is a true positive and needs no further judgement. A box that matches nothing is
either a false positive or a real object the reference never annotated, and the whole precision
problem is that the two are indistinguishable from the data. Every matched box is settled; only the
unmatched ones are open.

That makes the question far smaller than annotating a panel, and smaller still because it does not
need every unmatched box. Judging a sample of them estimates the proportion that are real, and that
proportion converts the precision bound into an interval:

    corrected precision = (matched + p_real * unmatched) / retained

where `p_real` is estimated from the sample with its own interval, which propagates into the
result. The uncorrected figure is the case `p_real = 0`, which is why it is a lower bound rather
than a measurement.

The sample is drawn under a frozen seed and stratified by the two things that plausibly move the
answer, object size and emitted class, so a rare class or a small-object regime cannot be
accidentally absent. The draw is recorded with the audit, so the same sample can be re-judged or
extended rather than silently redrawn.

Each sampled box is judged on two questions, and the second is only asked when the first is
answered yes. Whether it is a real entity the declared policy includes, and if so which class it
belongs to. The second gives classification accuracy on exactly the population the reference cannot
adjudicate, which is otherwise unmeasurable: class correctness on matched boxes is already known
from permitted labels, and class correctness on unannotated objects is known from nothing.

An audit reports three counts and never collapses them: real, false, and undecidable. A box a human
cannot resolve from the pixels stays undecidable and leaves both numerators, because forcing a
verdict to keep the arithmetic tidy is the failure this whole section is written against. The
undecidable rate is reported beside the estimate, since an audit that could not decide half its
sample has not measured precision however clean its interval looks.

An audit does not make the panel exhaustive and says nothing about recall. It resolves what the
retained boxes are, not what is missing from them.

### Prediction-assisted verification, and what it cannot establish

Annotating a dense panel from raw pixels is expensive enough that it may not happen, and a protocol
nobody completes measures nothing. The practical alternative is to show the annotator the system's
retained boxes and have them mark each valid or invalid. That is a legitimate and useful artifact,
and it is not a reference panel. Treating it as one is the failure this section exists to prevent.

**What it establishes.** Every retained box is adjudicated, so the true and false positive split is
directly observed. Precision becomes a measurement rather than a lower bound, and class correctness
on retained boxes is measurable alongside it. On a finite-category reference neither was available
at all, so this answers the question that was blocked.

**What it cannot establish.** Recall. By construction every reference in such a panel is something
the system proposed, so recall computed against it is trivially near one and means nothing. The
annotator cannot mark what was never shown, and an object the system missed is invisible in exactly
the same way it was invisible before. A verification panel that reports recall is reporting an
artifact of its own construction.

**What it degrades.** Geometry. A kept box is the model's box, so localization quality measured
against it is measured against the thing being tested. Where the annotator adjusts a box the
adjusted geometry is usable; where they accept one unchanged it is not, and the two cases are
recorded separately so the distinction survives into the metrics.

Objects the annotator notices and adds are kept and counted, because a found object is worth having.
They do not convert the panel into an exhaustive one: attention anchored by existing boxes is not a
sweep, and completeness claimed from an anchored pass is the same unverifiable assertion the blind
protocol exists to replace. Recall from such a panel is descriptive and anchored, never a gate
input.

A verification panel therefore declares `anchoring: prediction_assisted`, and carries
`exhaustive: false` and `recall_interpretable: false`. The evaluator refuses to report its recall
and coverage metrics as measurements, reporting them as anchored instead, so a later reader cannot
mistake which question the panel answered.

### Two diagnostics that survive a finite-category reference

Most measurement here is blocked by the same fact: on a finite-category reference an unmatched box is not evidence of a false box, so precision is a lower bound of unknown tightness and any change that adds boxes cannot be judged. Two diagnostics escape that, and they are the ones to run while the blocker stands.

**Miss attribution** partitions the misses, and misses are fully determined by the reference: a resolvable reference instance either was localized or was not. Attributing each miss to a mechanism therefore needs no assumption about unannotated objects. It is a prerequisite, not an optional report: a discovery route may not be added to close a recall gap until the misses it claims to address have been attributed and the `never_proposed` bucket is shown to be large enough to matter. A miss that was proposed below threshold, merged away, or localized loosely is a ranking, reconciliation or boundary defect, and another detector does not fix it.

The bucket boundaries depend on the adapter. Record the proposal cap per route: a route with a hard top-k cap can lose an object to cap saturation, which is its own bucket and a recorded-incomplete-search condition under D2; a route that emits one prediction per patch token above a score threshold has no such bucket, and its absence is a property of that adapter rather than a finding.

**Boundary headroom** bounds what reconciliation can return before any reconciliation rule is written. A miss whose cluster already contains an observation clearing the criterion is lost by choosing the wrong representative, and is recoverable by selection alone. A miss where no observation in any view clears it cannot be recovered by any selection rule, however clever: the geometry was never produced, and returning it needs magnification, refinement against pixels, or a different route. Measure the split first; a refinement rule written before it is a guess about its own ceiling.

**Pruner selectivity** applies to any component that removes retained boxes rather than adding them. The removals it makes among matched boxes are exactly measurable, because those boxes are matched to known references; the removals among unmatched boxes are of unknown value. That asymmetry is enough: a scorer that removes nothing but noise and a scorer that removes at random are distinguishable without knowing what the unmatched boxes are, since a random pruner removes both classes at the same rate. Report both rates across the sweep, name the operating point, and report the recall lost there alongside the boxes removed. A selectivity result is descriptive: it establishes that a pruner discriminates, never that the retained set became correct.

## Proposed gates

These gates apply together for the initial general-photograph reference-use candidate. A narrower qualified domain must be declared prospectively, not selected after finding failures in the test panel.

| Gate | Proposed threshold | Why it exists |
|---|---|---|
| G1 accepted annotation precision | One-sided 95% lower confidence bound >= 0.99 | Incorrect accepted labels cannot become reference truth cheaply |
| G2 localized instance recall | Lower bound >= 0.95 overall | Measures missed objects independently of class acceptance |
| G3 challenge-stratum recall | Lower bound >= 0.85 in every claimed challenge stratum | Overall performance cannot hide small/low-contrast failures |
| G4 useful-label coverage | Lower bound >= 0.80, of which specific-label coverage lower bound >= 0.50 | Abstaining on nearly everything cannot satisfy the product, and a permitted parent applied to everything is abstention wearing a class name |
| G5 localization quality | Lower bound >= 0.95 | Accepted labels need tight boxes, not only loose overlap |
| G6 final instance precision | Lower bound >= 0.95 | Prevents uncertain proposal spam from manufacturing high recall |
| G7 iterative benefit | At least 20% of baseline misses recovered, with paired lower bound supporting that gain; accepted-precision reduction no worse than 1 percentage point | Tests the central claim that additional compute buys useful completeness |
| G8 engineering contract | All mandatory deterministic acceptance cases pass | Accuracy cannot compensate for wrong coordinates or unsafe exports |
| G9 granularity conformance | Lower bound >= 0.90 on a granularity-annotated reference | CAP-2 is a product claim; without it, policy conformance rests on fixtures the spec already says cannot qualify anything |

Gate thresholds, the panel, and the multiplicity correction are three independent choices that jointly decide whether a gate can be reached at all. They are reconciled by the power calculation before collection closes; a threshold that no affordable panel can support is lowered prospectively, dropped, or declared structurally inconclusive, and is never rescued after the locked run by relaxing it.

For G7 the baseline is the same frozen bundle on one full-image pass, without tiling or iterative omission search, evaluated on the same images. If the baseline has too few misses to support a relative improvement estimate, mark G7 inconclusive and report absolute differences; do not invent a gain. Incremental passes must be assessed on final retained objects after verification.

G1–G6 and G9 must also be reported per source domain. A domain failing or lacking evidence is excluded from qualification, with aggregate conclusions recalculated only on a separately declared qualification panel or a new holdout. A development-time domain choice must not become a test-time selection loophole.

Because that rule makes one failing domain invalidate the aggregate claim, WP-03 collects a **reserve panel** alongside the locked one: independently sampled, held unopened, and sized by the same power calculation for the aggregate claim over the domains that remain if one is excluded. Without a reserve, the first domain failure costs a full re-collection under licence terms that may not permit it, and the schedule pressure that follows is exactly what the anti-selection rule exists to resist. If no reserve is affordable, declare each domain as its own qualification claim prospectively and make no aggregate claim.

## Statistical protocol

Freeze the statistical script and the power calculation it drives in M0, before the locked split is opened. Use 10,000 stratified image-cluster bootstrap replicates, preserving all objects within an image, for one-sided 95% lower bounds and paired comparisons. Report distributions and sample counts; do not treat detections within one image as independent observations.

Bootstrap intervals are empirical estimates, not distribution-free guarantees. A degenerate zero-error/zero-miss bootstrap cannot establish a certainty bound of 1.0. Such gates remain inconclusive unless a preregistered conservative image-level method provides adequate evidence; collect an independently selected larger panel where allowed. Undersampled strata also remain inconclusive. For a joint qualification claim, preregister the number K of mandatory gate/stratum tests — counting every gated metric, every claimed stratum, and every per-domain repetition — and use Bonferroni-adjusted one-sided bounds at alpha = 0.05/K; increase replicates if needed to estimate that tail. Uncorrected 95% intervals are descriptive only. K enters the power calculation as an input, so growing the claim by adding strata or domains shrinks alpha and raises the panel each gate needs; a release that cannot afford that panel narrows the claim rather than dropping the correction.

Calibration may map scores to precisely defined correctness events using external calibration data. Report calibration error and selective risk on the locked set and shifted-domain sets. A model's self-reported confidence, agreement count, or a calibration fit from another domain does not establish per-image reliability. Shift detection can withhold eligibility, but cannot prove domain membership or absence of shift from one image.

## Required ablations

Run the same locked evaluation for the baseline and these staged additions, with model/config costs recorded:

1. Full image only.
2. Overlapping multiscale tiles.
3. Class-independent proposal union.
4. Semantic inventory and concept expansion.
5. Instance/boundary refinement and conservative naming.
6. Retained-set pruning by an external scorer, reported as a selectivity curve with the recall lost at the chosen operating point.
7. Blind/residual/exemplar omission search.
8. Additional model-family verification, added only after miss attribution shows an unproposed remainder large enough to justify the compute.

Use development data to choose the final configuration before opening locked results. Report marginal objects recovered, new false instances, useful-label coverage, duplicate/split/merge changes, and compute per recovered correct object. Negative findings must remain in the report. Plot marginal gains versus compute and risk versus coverage using exportable plots.

## Qualification report contents

Bundle/domain identity; benchmark and split provenance; policy compatibility; leakage limits; the preregistered K and the power calculation with each gate's required and collected cluster counts; all gate outcomes including any declared structurally inconclusive before scoring; point estimates and corrected confidence bounds; per-domain and challenge metrics; the specific/generic split of label coverage; declared category scope and out-of-scope prediction counts; failed/abstained cases; ablations; offline and resumption evidence; representative errors; resource profile; and permitted use profiles.

Descriptive measurements are reported in a separate section that names each result's disqualifying property and states no gate outcome for it.

The report may conclude `engineering pass / reference qualification fail` or `inconclusive`. That is a valid research delivery. It must not claim the original universal ground-truth objective has been achieved. Downstream detector results remain automatic-reference agreement unless evaluated against independent compatible ground truth.
