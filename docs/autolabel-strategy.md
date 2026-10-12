# Autolabel quality strategy

## Purpose

Lynceus is a standalone, offline object-detection autolabeler, not a labeling platform. Its target
is the highest defensible annotation quality that can be obtained with autonomous computation,
followed by bounded agent or human escalation only where the image and policy require a judgement.

There is no single state-of-the-art direction for that target. Zero-shot open-vocabulary detection
is valuable when there are no domain labels and the vocabulary is not fixed, but it is not assumed
to be the final or best detector for a stable domain. The production strategy is a measured hybrid:

1. use zero-shot and class-agnostic models to bootstrap coverage and preserve unknown objects;
2. optionally adapt a closed-set specialist from a small, exhaustive domain seed;
3. separate localization, classification, geometry, and policy assessment where their errors differ;
4. generate candidates for recall before applying specialized precision evaluators;
5. use trusted pseudo-labels to improve a domain student only under locked evaluation; and
6. invoke an agent, then a human, only for residual decisions that deterministic evidence cannot settle.

"Close to perfect" is always conditional on a declared domain, ontology, annotation policy, and
reference protocol. The optimization target is maximum automatically accepted recall subject to a
predeclared lower bound on accepted precision and explicit limits on duplicate, merge, split,
classification, and localization errors. Unresolved output and escalation rate are reported beside
quality; they cannot be hidden by abstention.

## Two strategic profiles

### General zero-shot profile

The general profile requires no target-domain labels. It retains the current contract: fixed models,
policy, prompts, thresholds, and acceptance rules; no target-specific training; and every result
qualified only for domains supported by independent evaluation.

```text
open-vocabulary and class-agnostic proposals
    -> source-resolution geometry refinement
    -> conservative naming
    -> zero-shot verification and calibrated acceptance
    -> accepted, uncertain, or unresolved output
```

This profile is the portable fallback and the unknown-object safety net. It is not expected to beat
a well-trained domain specialist on a stable closed ontology.

### Proposed qualified-domain profile

The proposed qualified-domain profile would permit training on a small, independently annotated
development seed.
It does not weaken the locked evaluation rules: development, calibration, and evaluation remain
separate by source and scene, and a trained artifact receives a new qualification identity.

```text
domain specialist + general proposal routes
    -> independent geometry and classification stages
    -> domain-trained verification
    -> calibrated acceptance
    -> agent/human escalation for residual ambiguity
```

A versioned domain pack would contain the policy and ontology, seed manifest, model and verifier
weights,
visual exemplars, calibration artifacts, thresholds, dependency hashes, and qualification report.
Inference stays offline and deterministic under that frozen pack.

The current specification forbids target-specific training. The qualified-domain profile therefore
requires an explicit contract change before implementation; it must be added as a separate profile,
not introduced silently into the existing zero-shot claim.

## Candidate technical directions

### Zero-shot open-vocabulary detection

Use OWLv2, WeDetect, Grounding DINO, SAM 3 concept prompting, or later candidates as proposal and
recognition sources, not as automatic authorities. This direction is strongest when classes are
well described by language and labeled examples are unavailable. It remains useful in domain mode
for rare, novel, or omitted concepts.

Published benchmark rank does not choose a Lynceus route. Every candidate is compared on the same
images, ontology, tile schedule, proposal budget, and evaluator. Incremental class-neutral recall,
new false instances, correlated failures, cap hits, and compute decide whether its union is useful.

### Few-shot or fully supervised closed-set adaptation

When the domain and ontology are stable, adapt a strong pretrained detector on a small exhaustive
seed. Compare head-only, frozen-backbone, parameter-efficient, and full fine-tuning instead of
assuming one regime. Grow the seed through learning curves and challenge-stratum coverage rather
than a fixed image count.

The seed must contain empty images, confusing negatives, rare classes, small and occluded instances,
crowds, truncation, viewpoint and lighting changes, and every annotation convention the specialist
is expected to learn. A closed-set specialist supplies domain accuracy; general proposal routes
remain active so its learned background does not erase unknown objects.

### Decoupled localization and classification

Classification-coupled proposals can learn that objects outside the training vocabulary are
background. A complementary route therefore proposes regions from localization or mask quality
without naming them. Each region is then classified independently using some combination of:

- a domain closed-set classifier;
- detector class heads;
- open-vocabulary region-to-text matching;
- visual exemplar retrieval; and
- a wider context crop when the isolated object is ambiguous.

Objectness, localization quality, class support, and context compatibility remain separate evidence
fields. A class failure cannot erase plausible geometry, and an unknown region remains available for
future search or escalation.

### Segmentation-first discovery

Run class-agnostic mask generation as a complementary route, then filter regions, reconcile
fragments, classify surviving instances, and derive visible-extent boxes. Mask-first evidence can
improve tight boundaries, touching-instance separation, whole/part reasoning, and duplicate matching.
It is not a replacement for semantic detection: unconstrained mask generation overproduces parts,
textures, and stuff, so it must pass objectness and policy assessment.

### Visual exemplar and retrieval recognition

Text prompts are weak for visually defined product variants, defects, medical findings, custom
parts, and other low-text-describability concepts. A domain pack may therefore hold positive and
confusing-negative exemplars, prototype embeddings, and per-class similarity calibration. Exemplar
evidence augments rather than replaces a specialist because nearest-neighbour decisions can be
brittle under viewpoint, scale, and background changes.

### Semi-supervised teacher-student improvement

After a trusted seed exists, train an initial specialist, generate pseudo-labels on unlabeled domain
images, retain only verifier-calibrated labels, and train a student on the seed plus trusted labels.
Repeat only while a locked human reference improves.

Safeguards are mandatory because self-training amplifies systematic errors:

- human labels keep greater authority than pseudo-labels;
- uncertain and unsearched regions are ignored, never used as background;
- rare classes and hard negatives are sampled explicitly;
- teacher updates are frozen or deliberately slow;
- every iteration is reversible and versioned; and
- regression in any protected class or challenge stratum blocks adoption.

### High-recall proposals followed by specialized evaluators

Offline autolabeling need not force recall and precision into one real-time pass. First take the
union of low-threshold domain detection, open-vocabulary detection, class-agnostic proposals,
segmentation proposals, multiscale tiles, test-time transforms, exemplar search, and residual search.
Then apply evaluators specialized by failure type:

| Evaluator | Question |
|---|---|
| Existence | Is a physical object supported inside this region? |
| Classification | Is the proposed class supported, is only a parent supported, or is it unknown? |
| Localization | Is the box tight enough, and what IoU or mask quality is supported? |
| Instance identity | Are candidates duplicates, neighbours, fragments, or a whole/part pair? |
| Context | Does wider context resolve an otherwise ambiguous crop? |
| Policy | Is this an instance, part, group, stuff, depiction, or reflection? |

TTN is one candidate existence/class pruning signal, not the architecture. The released zero-shot
TTN did not beat detector score on Lynceus's tiny aerial objects, while its published direction and
task-specific variant motivate a better experiment: train a domain multi-head verifier on real,
out-of-fold proposal errors. Inputs may include high-resolution object and context crops, masks,
route scores, supporting views, magnification, truncation, class margins, exemplar distances, and
neighbour overlaps. Outputs should separately estimate existence, class correctness, localization
quality, duplicate/part relations, and the need for semantic review.

Verifier training must use out-of-fold proposals. For each scene-held-out fold, train or configure
the proposer on the other folds, generate candidates on the held-out fold, match them to exhaustive
references, and collect the resulting false positives, misses, class errors, and bad boxes. A
verifier trained on in-sample detector output would learn unrealistically easy errors and leak the
detector's fit.

## Recall search before precision pruning

The search controller should treat detection as an iterative evidence-acquisition problem:

1. run full-frame and source-resolution tiled proposal routes;
2. retain low-threshold candidates with route and view provenance;
3. revisit tile-edge objects with expanded context;
4. examine apparently empty residual regions at greater magnification;
5. search inside large objects for independently annotatable contents;
6. use confident examples as optional exemplar queries for further instances;
7. send merge, split, geometry, and class disagreements to the relevant specialist; and
8. stop under a frozen budget or after complete audit rounds add no material evidence.

No precision stage may silently remove a proposal. Every removal records the evaluator, threshold,
evidence, and reason, so recall lost by each stage can be measured against the exhaustive reference.

## Agentic escalation

An agent is a bounded controller over perception tools, not a free-form detector. It is invoked only
for a named unresolved decision such as class-versus-parent, whole-versus-part, duplicate-versus-
neighbour, physical-versus-depicted, unresolved count, or insufficient crop context.

Allowed actions include expanding a crop, selecting another tile, invoking another proposal route,
lowering a threshold locally, requesting segmentation, comparing exemplars, querying a synonym or
parent class, and inspecting wider image context. Every action has a cost and the run has a fixed
action budget.

The agent returns typed output containing the candidate and decision IDs, evidence references,
actions used, reason codes, remaining uncertainty, and one of `resolved`, `unresolved`, or
`needs_human`. It may not invent unsupported geometry, modify the ontology or global thresholds,
treat verbal confidence as probability, hide failed calls, or turn missing evidence into certainty.

Agent value is measured only on the residual population routed to it. Report errors corrected, new
errors introduced, abstention reduction, human escalations avoided, compute, and results by decision
type. A general VLM claim does not qualify the agent.

## Frozen experiment ladder

Evaluate the directions cumulatively on one representative domain before changing the default:

| ID | Configuration |
|---|---|
| A | Current prompted OWLv2 baseline |
| B | Stronger zero-shot and class-agnostic proposal union |
| C | Closed-set detector adapted on the exhaustive seed |
| D | Class-agnostic proposals plus a separately trained classifier |
| E | Union of B, C, and D under the same reconciliation rules |
| F | E plus mask-based geometry refinement and instance separation |
| G | F plus released zero-shot TTN or an equivalent generic pruner |
| H | F plus the out-of-fold domain multi-head verifier |
| I | H plus one guarded teacher-student iteration |
| J | I plus bounded agent escalation |

For every rung report candidate recall before pruning, accepted precision and recall at the declared
IoU thresholds, recall lost by each evaluator, class and localization error, duplicate/merge/split
rates, calibration, unresolved and escalation rates, wall time, memory, and model calls. Report all
metrics by class, domain, source, object size, occlusion, truncation, crowding, and other claimed
challenge strata. Negative results remain part of the decision record.

## Implementation sequence

1. Finish the exhaustive development/calibration/evaluation protocol and all required metrics.
2. Add a reproducible configuration sweep with risk-coverage and compute-quality Pareto reports.
3. Generalize proposal adapters and test direct OWLv2 objectness plus a class-independent route.
4. Implement the iterative residual and omission search controller.
5. Add an optional qualified-domain pack and train a closed-set specialist on a small seed.
6. Add independent region classification and exemplar evidence.
7. Train and evaluate out-of-fold specialized verifiers against detector score and TTN.
8. Trial one guarded teacher-student round only after verifier calibration passes.
9. Add bounded agent escalation only after deterministic error attribution defines its workload.

## Primary references

- [Scaling Open-Vocabulary Object Detection (OWLv2)](https://papers.neurips.cc/paper_files/paper/2023/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html)
- [Grounding DINO](https://arxiv.org/abs/2303.05499)
- [SAM 3: Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [Open-vocabulary versus closed-set few-shot detection](https://arxiv.org/abs/2410.15315)
- [Few-Shot Object Detection with Foundation Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_Few-Shot_Object_Detection_with_Foundation_Models_CVPR_2024_paper.pdf)
- [Learning Open-World Object Proposals without Learning to Classify](https://arxiv.org/abs/2108.06753)
- [Label, Verify, Correct](https://openaccess.thecvf.com/content/CVPR2022/papers/Kaul_Label_Verify_Correct_A_Simple_Few_Shot_Object_Detection_Method_CVPR_2022_paper.pdf)
- [Soft Teacher](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.pdf)
- [Taming Self-Training for Open-Vocabulary Object Detection](https://research.google/pubs/taming-self-training-for-open-vocabulary-object-detection/)
- [The Label Imitation Game / TTN](https://arxiv.org/abs/2606.30875)
