"""Ordering retained boxes by how much a human looking at one would learn.

Every box this pipeline emits carries five uncertainty dimensions, a detector score, a view count, a
truncation flag and -- where a verifier ran -- a confirmation and the overlap that produced it. None
of that reaches a person in a usable order, so a reviewer with time for a hundred boxes out of two
thousand has no way to spend it on the hundred that matter.

The order here is a conversion of evidence already recorded and not a new signal. It is declared
rather than fitted: nothing in it is tuned against a reference, because a ranking fitted on the
panel it is then measured on measures the fitting. The rule is lexicographic, worst first:

1. unresolved dimensions, most first -- the pipeline's own count of what it could not settle;
2. unconfirmed before confirmed, since a box no second model found is the one whose existence is
   least supported;
3. ascending confirming overlap, so a confirmed box whose extent two models read differently comes
   before one they agree about;
4. ascending detector score;
5. ascending area, because a small box is both likelier wrong and cheaper to adjudicate.

What this cannot do is say a box is wrong. It is an order, and the measurement that matters is
whether a reviewer following it reaches the corrections sooner than one following the detector's
score, which is free.
"""

DIMENSIONS=('existence','class','boundary','count','granularity')


def evidence_from_run(result,candidates,ontology):
    """Recover each retained object's evidence from a finished run's own candidate store.

    Retained candidates are written in object order and the verification stage pairs them that way,
    so this pairs them that way too. Keying on geometry was wrong as soon as refinement existed: a
    refined object carries the refiner's box while its retained candidate still carries the
    proposal, so every lookup missed and every object read back as an unverified zero-score
    detection. A count mismatch is an error rather than something a fallback should hide.
    """
    useful={c['id']:c['useful'] for c in ontology['classes']}
    leads=[c for c in candidates if c['disposition']=='retained']
    if len(leads)!=len(result['objects']):
        raise ValueError(f'retained_candidate_mismatch: {len(leads)} for {len(result["objects"])} objects')
    evidence=[]
    for obj,lead in zip(result['objects'],leads):
        verification=lead.get('verification') or {}
        evidence.append({'score':lead['model_score'],
                         'views':len(lead.get('merged_views',[])) or 1,
                         'view_edge':bool(lead['view_edge']),
                         'alternatives':obj['alternative_boxes'],
                         'useful':useful.get(obj['label']['id'],False),
                         'verified':verification.get('confirmed'),
                         'verifier_score':verification.get('verifier_score'),
                         'verifier_overlap':verification.get('overlap')})
    return evidence


def unresolved(obj):
    return sum(1 for d in DIMENSIONS if obj['uncertainty'].get(d)!='supported')


def area(obj):
    x1,y1,x2,y2=obj['bbox_xyxy']
    return max(0.0,(x2-x1))*max(0.0,(y2-y1))


def difficulty(obj,ev):
    """The sort key and the components it was built from, so an order is explainable per box."""
    confirmed=ev.get('verified')
    overlap=ev.get('verifier_overlap')
    components={'unresolved_dimensions':unresolved(obj),
                'verified':confirmed,
                'verifier_overlap':overlap,
                'model_score':ev.get('score'),
                'views':ev.get('views'),
                'view_edge':ev.get('view_edge'),
                'area':round(area(obj),2),
                'label':obj['label']['id']}
    key=(-components['unresolved_dimensions'],
         1 if confirmed else 0,
         overlap if overlap is not None else -1.0,
         components['model_score'] if components['model_score'] is not None else -1.0,
         components['area'])
    return key,components


def queue(objects,evidence):
    """Retained boxes worst first, each carrying why it sits where it does."""
    rows=[]
    for position,(obj,ev) in enumerate(zip(objects,evidence)):
        key,components=difficulty(obj,ev)
        rows.append({'object':obj['id'],'bbox_xyxy':list(obj['bbox_xyxy']),
                     'label':obj['label']['id'],'status':obj['status'],
                     'reasons':list(obj['uncertainty'].get('reasons',[])),
                     'difficulty':components,'_key':(key,position)})
    rows.sort(key=lambda row:row['_key'])
    for rank,row in enumerate(rows,start=1):
        row['rank']=rank
        del row['_key']
    return rows


def review_run(run,limit=None):
    """The review order for a finished run, read off its own artifacts.

    This is a read of recorded evidence and runs no model, so it cannot change what a run decided.
    It deliberately does not accept a reference or a second run: a queue built by looking at which
    boxes are wrong is not a queue, it is the answer.
    """
    import json
    from pathlib import Path
    from .policy import load_ontology
    root=Path(run)
    result=json.loads((root/'annotation.json').read_text())
    candidates=[json.loads(line) for line in
                (root/'evidence/candidates.jsonl').read_text().splitlines()]
    ontology=load_ontology(result['ontology']['id'])
    rows=queue(result['objects'],evidence_from_run(result,candidates,ontology))
    return {'run_id':result['run_id'],'image':result['image']['source_sha256'],
            'ontology':result['ontology']['id'],'retained':len(rows),
            'budget':limit,'order':rows[:limit] if limit else rows,
            'rule':'declared and not fitted: unresolved dimensions, then unconfirmed first, then '
                   'ascending confirming overlap, detector score and area',
            'reading':'an order over boxes, not a judgement that any box is wrong'}
