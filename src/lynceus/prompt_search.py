"""Searching for a prompt vocabulary instead of writing one by hand.

Nobody knows whether a detector finds more small overhead figures under `person`, `pedestrian` or
`person seen from above` without trying, so the phrasing is searched. The search is an experiment
and carries an experiment's obligations: it runs on development data, the winner is frozen before
the held-out panel is touched, and the candidates considered are recorded alongside the winner.

Two properties make it affordable and honest.

A prompted detector's image features and box predictions do not depend on the text, so they are
computed once per view and reused for every candidate. Only the text tower and the class head
re-run. That is a cost reduction and not an approximation -- the scores are the ones a full forward
pass would produce.

The classes interact. The emitted label is the class that wins over all prompts, so rephrasing one
changes which boxes the others win, and per-class objectives optimized in isolation would not
compose. Coordinate ascent over classes, re-scoring the whole vocabulary each step, is the minimum
that respects this. It finds a local optimum and claims nothing more.
"""
import numpy as np

from .evaluation import iou_matrix, match_matrix


def score_vocabulary(views,text_embeds,classes,references,threshold,match_iou=0.5,label_iou=0.75):
    """Per-class and overall quality of one vocabulary, from cached per-view features.

    `views` carries, per view, the cached class-head input, the predicted boxes in source
    coordinates, and the image index the view came from. `text_embeds` is one query embedding per
    class in `classes` order.
    """
    per_image={}
    for view in views:
        logits=view['predict'](text_embeds)          # [patches, classes]
        scores=logits.max(axis=-1)
        winners=logits.argmax(axis=-1)
        keep=scores>=threshold
        image=per_image.setdefault(view['image'],[])
        for box,score,winner in zip(view['boxes'][keep],scores[keep],winners[keep]):
            image.append({'bbox_xyxy':list(box),'label':classes[int(winner)],'score':float(score)})
    totals={'retained':0,'matched':0,'named':0}
    per_class={c:{'retained':0,'matched':0,'references':0} for c in classes}
    for image,refs in references.items():
        boxes=per_image.get(image,[])
        for reference in refs:
            for label in reference['permitted_labels']:
                if label in per_class:per_class[label]['references']+=1
        totals['retained']+=len(boxes)
        for box in boxes:
            if box['label'] in per_class:per_class[box['label']]['retained']+=1
        if not boxes or not refs:continue
        values=iou_matrix(boxes,refs)
        pairs=match_matrix(values,values>=match_iou)
        totals['matched']+=len(pairs)
        # The gates are scored on a correctly labelled reference found at a tight threshold, not on
        # a box that merely landed somewhere right. Optimising the looser quantity buys localization
        # by trading away naming, which is measurable here rather than after the fact.
        permitted=np.array([[boxes[a]['label'] in refs[b]['permitted_labels']
                             and boxes[a]['label'] not in ('entity','unknown_object')
                             for b in range(len(refs))] for a in range(len(boxes))])
        totals['named']+=len(match_matrix(values,(values>=label_iou)&permitted))
        for a,b,_ in pairs:
            label=boxes[a]['label']
            if label in per_class and label in refs[b]['permitted_labels']:
                per_class[label]['matched']+=1
    eligible=sum(len(v) for v in references.values())
    return {'recall':round(totals['matched']/eligible,6) if eligible else 0.0,
            'retained':totals['retained'],'matched':totals['matched'],'named':totals['named'],
            'useful_label_coverage':round(totals['named']/eligible,6) if eligible else 0.0,
            'precision':round(totals['matched']/totals['retained'],6) if totals['retained'] else 0.0,
            'per_class':per_class}


def objective(report,eligible):
    """What the search maximises: correctly labelled references found, per retained box.

    The counted event is a reference found at the tight threshold under a label its mapping permits,
    which is what G1 and G4 are scored on. An earlier version of this counted any localized box
    regardless of label; it improved on held-out data and cost nineteen per cent of useful-label
    coverage, because a vocabulary can buy localization by pushing contested boxes into abstention.
    Optimising the loose quantity and hoping the tight one follows does not work.

    Recall alone would reward a vocabulary that floods the image and precision alone one that says
    almost nothing, so their harmonic mean is taken.
    """
    coverage=report['named']/eligible if eligible else 0.0
    precision=report['named']/report['retained'] if report['retained'] else 0.0
    if coverage<=0 or precision<=0:return 0.0
    return round(2*coverage*precision/(coverage+precision),6)


def coordinate_ascent(classes,candidates,evaluate,baseline=None,rounds=2,log=None):
    """Improve one class's phrasing at a time, re-scoring the whole vocabulary at each step.

    `evaluate` takes a full {class: phrase} mapping and returns a score. Starting from the baseline
    vocabulary, each class's candidates are tried with every other class held at its current
    phrasing, and a candidate is adopted only if it improves the whole-vocabulary score. Because
    the classes compete, that is the only score that means anything.

    A second round is run by default: adopting a phrasing for one class changes what the others
    compete against, so a candidate rejected early can win later.
    """
    current=dict(baseline or {c:c for c in classes})
    best=evaluate(current)
    trace=[{'step':'baseline','vocabulary':dict(current),'score':best}]
    if log:log('baseline', best)
    for round_index in range(rounds):
        improved=False
        for class_id in classes:
            for phrase in candidates.get(class_id,[]):
                if phrase==current[class_id]:continue
                trial=dict(current);trial[class_id]=phrase
                score=evaluate(trial)
                trace.append({'step':f'round{round_index}:{class_id}','phrase':phrase,'score':score,
                              'adopted':score>best})
                if log:log(f'  {class_id} = {phrase!r}', score, score>best)
                if score>best:
                    best=score;current=trial;improved=True
        if not improved:break
    return {'vocabulary':current,'score':best,'trace':trace,'rounds_used':round_index+1}
