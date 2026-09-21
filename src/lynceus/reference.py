"""Building and checking an exhaustively annotated reference panel.

A finite-category reference bounds three things at once: precision is a lower bound of unknown
tightness, calibration cannot be fitted because an unmatched box is not a scoreable error, and
granularity conformance has nothing to score against. One small exhaustive panel removes all three,
and more finite-category data removes none of them.

This module does not annotate. It plans the review sweep that makes completeness checkable, and it
ingests what a human produced, refusing anything that would make the panel quietly worthless -- an
unvisited tile, a missing granularity tag, a box outside its image, an absent leakage identity.

Exhaustive is a property of the procedure. An annotator's confidence that they saw everything is
not evidence; a record of which tiles were visited is.
"""
from .artifacts import canonical,digest
from .geometry import validate_box
from .policy import available_ontologies,load_policy

KINDS=('instance','part','group','stuff')
LAYERS=('physical','reflected','depicted')
# An entity the policy does not settle stays in recall denominators and leaves precision ones: a
# prediction cannot be scored wrong against a reference that does not know its own answer.
RESOLUTIONS=('resolved','unresolved')
# How the annotator's attention was directed, which decides what the panel can support.
ANCHORING=('blind','prediction_assisted')
# Only relevant when the annotator was shown predictions: whether this box is the model's geometry.
ORIGINS=('added','accepted','adjusted')


def plan_review(width,height,tile=512,overlap=0.5):
    """Overlapping source-resolution tiles a human must visit for the image to count as swept.

    The same geometry the scan planner uses, for the same reason: a small entity is only reliably
    seen in a view small enough to show it at its own resolution.
    """
    if width<=0 or height<=0:raise ValueError('invalid_image_size')
    step=max(1,int(tile*(1-overlap)))
    xs=sorted({min(x,max(0,width-tile)) for x in range(0,max(1,width),step)})
    ys=sorted({min(y,max(0,height-tile)) for y in range(0,max(1,height),step)})
    return [{'id':f'r{row}c{column}','bbox_xyxy':[float(x),float(y),float(min(x+tile,width)),float(min(y+tile,height))]}
            for row,y in enumerate(ys) for column,x in enumerate(xs)]


def check_image(record,ontology,tile=512,overlap=0.5):
    """Validate one annotated image, returning it in benchmark form or raising a typed error.

    Every obligation the protocol states is enforced here rather than trusted, because each one
    fails silently: an unvisited tile looks like an empty region, an untagged part looks like an
    instance, and a missing scene identity looks like an image nobody has seen before.
    """
    for key in ('id','sha256','scene_id','near_duplicate_group','size','visited_tiles','annotations'):
        if not record.get(key) and record.get(key)!=[]:raise ValueError(f'missing_panel_field: {key}')
    width,height=record['size']
    planned={t['id'] for t in plan_review(width,height,tile,overlap)}
    visited=set(record['visited_tiles'])
    unknown=visited-planned
    if unknown:raise ValueError('unknown_review_tile: '+','.join(sorted(unknown)[:3]))
    missing=planned-visited
    if missing:raise ValueError(f'incomplete_sweep: {len(missing)} of {len(planned)} tiles unvisited')

    labels={c['id']:c for c in ontology['classes']}
    references=[]
    seen=set()
    for entry in record['annotations']:
        for key in ('id','bbox_xyxy','label','kind','scene_layer','resolution'):
            if key not in entry:raise ValueError(f'missing_annotation_field: {key}')
        if entry['id'] in seen:raise ValueError('duplicate_annotation_id: '+entry['id'])
        seen.add(entry['id'])
        box=validate_box(entry['bbox_xyxy'],width,height)
        if entry['kind'] not in KINDS:raise ValueError('invalid_kind: '+str(entry['kind']))
        if entry['scene_layer'] not in LAYERS:raise ValueError('invalid_scene_layer: '+str(entry['scene_layer']))
        if entry['resolution'] not in RESOLUTIONS:raise ValueError('invalid_resolution: '+str(entry['resolution']))
        if entry['label'] not in labels:raise ValueError('unknown_label: '+str(entry['label']))
        permitted=entry.get('permitted_labels') or [entry['label']]
        if not set(permitted)<=set(labels):raise ValueError('unknown_permitted_label')
        # Only whole physical instances the policy settled are scoreable references. The rest are
        # retained and counted, so their absence from a denominator is recorded rather than hidden.
        origin=entry.get('origin','added')
        if origin not in ORIGINS:raise ValueError('invalid_origin: '+str(origin))
        eligible=entry['kind']=='instance' and entry['scene_layer']=='physical' and entry['resolution']=='resolved'
        reference={'id':entry['id'],'bbox_xyxy':box,'permitted_labels':sorted(permitted),
                   'kind':entry['kind'],'scene_layer':entry['scene_layer'],'resolution':entry['resolution'],
                   # An accepted box is the model's own geometry, so localization measured against
                   # it is measured against the thing under test; an adjusted one is the
                   # annotator's. The distinction has to survive into the metrics.
                   'origin':origin}
        if not eligible:
            reference['eligible']=False
            reference['exclusion_reason']=('unresolved_by_policy' if entry['resolution']=='unresolved'
                                           else f"not_a_physical_instance:{entry['kind']}/{entry['scene_layer']}")
        references.append(reference)
    return {'id':record['id'],'sha256':record['sha256'],'scene_id':record['scene_id'],
            'near_duplicate_group':record['near_duplicate_group'],'size':[width,height],
            'domain':record.get('domain','undeclared'),'split':record.get('split','evaluation'),
            'run_status':record.get('run_status','missing'),'references':references,
            'review':{'planned_tiles':len(planned),'visited_tiles':len(visited),'tile':tile,'overlap':overlap}}


def build_panel(records,ontology_id,*,domain,annotator,tile=512,overlap=0.5,blind=True,
                anchoring='blind',rejected=0):
    """Assemble checked images into a benchmark payload an exhaustive reference can carry.

    `blind` records that the reference was annotated before any prediction was visible. It is not
    verifiable from the data -- nothing in a box says what its annotator had seen -- so it is a
    declaration, and a panel that cannot declare it is refused rather than silently accepted as
    independent.
    """
    if anchoring not in ANCHORING:raise ValueError('invalid_anchoring: '+str(anchoring))
    # A blind panel is the only kind that can claim exhaustiveness, and it has to declare it. A
    # prediction-assisted one declares the opposite and is refused if it claims otherwise.
    if anchoring=='blind' and not blind:raise ValueError('reference_annotated_against_predictions')
    if anchoring=='prediction_assisted' and blind:raise ValueError('prediction_assisted_panel_cannot_claim_blind')
    if not records:raise ValueError('empty_panel')
    registry=available_ontologies()
    if ontology_id not in registry:raise ValueError('unknown_ontology: '+str(ontology_id))
    ontology=registry[ontology_id]

    images=[];identities={}
    for record in records:
        checked=check_image(record,ontology,tile,overlap)
        checked['domain']=checked['domain'] if checked['domain']!='undeclared' else domain
        for key in ('sha256','scene_id','near_duplicate_group'):
            identities.setdefault(key,{}).setdefault(checked[key],[]).append(checked['id'])
        images.append(checked)
    duplicates=[f'{key}={value}' for key,groups in identities.items() for value,ids in groups.items()
                if key=='sha256' and len(ids)>1]
    if duplicates:raise ValueError('duplicate_image_content: '+','.join(duplicates[:3]))

    eligible=sum(1 for im in images for r in im['references'] if r.get('eligible',True))
    exhaustive=anchoring=='blind'
    origins={}
    for im in images:
        for r in im['references']:origins[r['origin']]=origins.get(r['origin'],0)+1
    return {'schema_version':'lynceus.benchmark/1.0','frozen':True,'policy_compatible':True,
            'reference_provenance':'independent','exhaustive':exhaustive,'anchoring':anchoring,
            # Recall against a prediction-assisted panel is an artifact of its construction: every
            # reference in it is something the system proposed, so it cannot be a measurement.
            'recall_interpretable':exhaustive,
            'rejected_predictions':rejected,'origins':origins,
            'declared_before_predictions_seen':exhaustive,'blind_annotation':exhaustive,
            'ontology':ontology_id,'policy':load_policy()['id'],'domain':domain,'annotator':annotator,
            # No category scope: an exhaustive panel has nothing outside it to exclude, so every
            # retained prediction is adjudicable and precision is a number rather than a bracket.
            'category_scope':None,
            'review_protocol':{'tile':tile,'overlap':overlap,
                               'sweep':'every planned tile visited' if exhaustive
                                       else 'tiles visited with predictions shown; attention anchored, not a sweep'},
            'counts':{'images':len(images),'references':sum(len(im['references']) for im in images),
                      'eligible_references':eligible},
            'images':images,
            'panel_sha256':digest(canonical([im['sha256'] for im in images]))}
