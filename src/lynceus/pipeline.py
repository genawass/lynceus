import json
import resource as rusage
import time
import uuid
from pathlib import Path
from io import BytesIO
from PIL import ImageDraw
from .images import normalize_image
from .policy import load_policy,load_ontology,common_ancestor,co_occurring,is_useful
from .contracts import identity,summarize,validate_annotation
from .artifacts import canonical,digest,write_artifact,verify_artifacts
from .bundle import preflight,doctor
from .geometry import validate_box,plan_tiles,covered_fraction,ios,iou


def peak_rss_bytes():
    return rusage.getrusage(rusage.RUSAGE_SELF).ru_maxrss*1024


def render_overlay(image,objects):
    """Nonauthoritative visualization: never the bare source image."""
    canvas=image.convert('RGB');draw=ImageDraw.Draw(canvas)
    for o in objects:
        draw.rectangle(o['bbox_xyxy'],outline=(255,0,0),width=2)
        draw.text((o['bbox_xyxy'][0]+2,o['bbox_xyxy'][1]+2),f"{o['id']} {o['label']['id']} ({o['status']})",fill=(255,0,0))
    banner=f'UNQUALIFIED baseline: {len(objects)} uncertain proposal(s); not a reference annotation'
    draw.rectangle([0,canvas.height-12,canvas.width,canvas.height],fill=(0,0,0))
    draw.text((2,canvas.height-11),banner,fill=(255,255,0))
    buffer=BytesIO();canvas.save(buffer,format='PNG');return buffer.getvalue()


def render_report(objects,regions,status,coverage,profile='owlv2-single-pass-v1'):
    """Local read-only explanation; no external dependencies and no request for human input."""
    esc=lambda v:str(v).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    unknown=[o for o in objects if o['label']['id']=='unknown_object']
    unresolved_boundary=[o for o in objects if o['uncertainty']['boundary']!='supported']
    excluded=[l for l in ['reflected','depicted'] if not any(o['scene_layer']==l for o in objects)]
    rows=''.join(f"<tr><td>{esc(o['id'])}</td><td>{esc(o['label']['id'])}</td><td>{esc(o['status'])}</td><td>{esc(o['uncertainty']['boundary'])}</td><td>{esc(o['uncertainty']['class'])}</td><td>{esc([round(c,1) for c in o['bbox_xyxy']])}</td></tr>" for o in objects)
    region_rows=''.join(f"<tr><td>{esc(r['kind'])}</td><td>{esc(r['reason'])}</td></tr>" for r in regions)
    return ('<!doctype html><meta charset="utf-8"><title>Lynceus baseline</title>'
        '<h1>Unqualified object-discovery baseline</h1>'
        f'<p><b>Stop reason:</b> execution {esc(status)}; search profile <code>{esc(profile)}</code> '
        f'({esc(len(coverage["completed_jobs"]))} of {esc(len(coverage["planned_jobs"]))} planned jobs completed, '
        f'{esc(len(coverage["failed_jobs"]))} failed). This scan is not exhaustive search and cannot establish absence.</p>'
        f'<p><b>Unknown classes:</b> {esc(len(unknown))} object(s) carry <code>unknown_object</code>; their geometry is retained and they never enter a supervised export as a known class.</p>'
        f'<p><b>Unresolved boundaries:</b> {esc(len(unresolved_boundary))} object(s) have a boundary that is not supported; their boxes are best estimates, not precise geometry.</p>'
        f'<p><b>Excluded policy layers:</b> no object was retained under layer(s) {esc(", ".join(excluded)) or "none"}; the baseline does not separate reflections or depictions yet, so their absence is not evidence.</p>'
        f'<table border="1"><tr><th>id</th><th>label</th><th>status</th><th>boundary</th><th>class</th><th>bbox_xyxy</th></tr>{rows}</table>'
        f'<h2>Regions</h2><table border="1"><tr><th>kind</th><th>reason</th></tr>{region_rows}</table>'
        '<p>All retained proposals are uncertain. Existence, boundaries, classes and counts have not been independently verified. '
        'No annotation or human approval is requested.</p><a href="annotation.json">Native result</a>').encode()


def touches_edge(box,size,tolerance=1.0):
    return box[0]<=tolerance or box[1]<=tolerance or box[2]>=size[0]-tolerance or box[3]>=size[1]-tolerance


def discover(adapter,image,jobs):
    """Run the adapter over every planned view and map observations back to source pixels.

    A view is cropped at source resolution, so a smaller tile reaches the model at higher
    magnification. Mapping back is a pure translation by the crop origin.
    """
    observations=[];completed=[];failed=[]
    for job in jobs:
        x1,y1,x2,y2=(int(round(v)) for v in job['bbox_xyxy'])
        view=image if job['id']=='full' else image.crop((x1,y1,x2,y2))
        try: predictions=adapter.predict(view)
        except KeyboardInterrupt: raise
        except Exception as exc:
            # One failed view leaves the rest of the scan usable, but forbids a complete status.
            failed.append({'id':job['id'],'bbox_xyxy':job['bbox_xyxy'],'error':type(exc).__name__,'message':str(exc)})
            continue
        for index,p in enumerate(predictions):
            box=[p['bbox_xyxy'][0]+x1,p['bbox_xyxy'][1]+y1,p['bbox_xyxy'][2]+x1,p['bbox_xyxy'][3]+y1]
            observations.append({'id':f"{job['id']}-{index}",'job':job['id'],'level':job['level'],
                'bbox_xyxy':box,'view_bbox_xyxy':p['bbox_xyxy'],'label':p['label'],
                'model_score':p['model_score'],'class_scores':p.get('class_scores'),
                'view_edge':bool(p.get('crop_edge')) or (job['id']!='full' and touches_edge(p['bbox_xyxy'],view.size))})
        completed.append(job['id'])
    return observations,completed,failed


def same_instance(lead,other,ios_threshold,iou_threshold):
    """Do two same-label observations describe one instance?

    Tiling and dense scenes pull the geometry in opposite directions. A detection truncated at a
    tile border overlaps its whole-object counterpart with low IoU but high intersection-over-
    smaller, so IoU alone under-merges exactly the duplicates slicing creates. But IoS is also high
    whenever any small box sits inside a larger one, which is the ordinary geometry of a crowd, so
    IoS alone collapses distinct neighbours -- the suppression of a valid separate instance that
    the annotation policy forbids.

    The recorded truncation flag separates the two cases: a tile-truncated duplicate almost always
    has an observation whose box reaches its view border, and a crowd neighbour does not. So IoS
    is trusted only when one side is flagged, and unflagged observations must agree on IoU.
    """
    if iou(lead['bbox_xyxy'],other['bbox_xyxy'])>=iou_threshold:
        return 'iou_agreement'
    if (lead['view_edge'] or other['view_edge']) and ios(lead['bbox_xyxy'],other['bbox_xyxy'])>=ios_threshold:
        return 'view_edge_truncation'
    return None


def mergeable_label(lead,other,ontology):
    """The label two observations of one instance can share, or None if they describe different things.

    Exact agreement merges and keeps the label. Where the labels differ, the most specific claim the
    pair jointly supports is their common ancestor, and they merge under it only if that ancestor is
    a useful class: two views calling one object `car` and `van` jointly support `vehicle`, which is
    a real claim, while `bicycle` and `person` support only `entity`, which is not and must stay two
    objects.

    Conservative naming runs per observation, so it cannot reach this case: each view was individually
    confident and had no runner-up to fall back from. The disagreement only exists across views, and
    so can only be resolved here.

    `unknown_object` never merges into a class. It asserts that no useful class is supported, and
    folding it into one would manufacture a claim no view made.
    """
    if lead['label']==other['label']:return lead['label']
    if ontology is None:return None
    if 'unknown_object' in (lead['label'],other['label']):return None
    ancestor=common_ancestor(ontology,lead['label'],other['label'])
    if ancestor is None or not is_useful(ontology,ancestor):return None
    return ancestor


def merge_observations(observations,ios_threshold,iou_threshold=0.5,ontology=None):
    """Greedy highest-score-first merge of repeated views of one instance.

    Only same-label observations merge, because overlap alone cannot suppress a different object --
    a backpack legitimately sits inside a person's box. The winning box is kept rather than a
    union, since policy forbids asserting extent no single view supported; the losing boxes are
    preserved as alternatives instead of being discarded.
    """
    groups=[];reasons={}
    for i in sorted(range(len(observations)),key=lambda i:(-observations[i]['model_score'],observations[i]['id'])):
        obs=observations[i]
        for group in groups:
            lead=observations[group[0]]
            label=mergeable_label(lead,obs,ontology)
            if label is None:
                continue
            reason=same_instance(lead,obs,ios_threshold,iou_threshold)
            if reason:
                if label!=lead['label']:
                    # The pair jointly supports only the broader class, so the surviving record
                    # makes that claim and keeps both specific readings as alternatives.
                    lead['label']=label
                    lead['label_alternatives']=sorted({*lead.get('label_alternatives',[]),
                                                       *other_labels(lead,obs)})
                    reason=reason+'+broader_class_across_views'
                group.append(i);reasons[i]=reason;break
        else:
            groups.append([i])
    return groups,reasons


def other_labels(lead,obs):
    return [l for l in (obs['label'],*obs.get('label_alternatives',[])) if l!=lead['label']]


def fuse_boxes(members):
    """Score-weighted consensus box for one cluster of observations of the same instance.

    Every fused coordinate is a weighted mean of observed coordinates, so it lies inside their
    range. That matters for policy: a union would push the extent beyond what any single view
    supported, which the annotation policy forbids, while a mean only interpolates between
    observations that each saw the object.
    """
    weights=[max(o['model_score'],1e-6) for o in members]
    total=sum(weights)
    return [sum(o['bbox_xyxy'][k]*w for o,w in zip(members,weights))/total for k in range(4)]


def consolidate(observations,ios_threshold,iou_threshold,rounds=2,fuse=True,ontology=None):
    """Merge repeated views into instances, optionally fusing each cluster's geometry and repeating.

    Both extras default off, because both were measured and neither helped.

    Score-weighted fusion was meant to improve boxes by consensus. It does the opposite: when a
    tile-truncated observation and a whole-object one describe the same instance, averaging drags
    the good box toward the bad one. Measured on a held-out panel it cost 18 of 646 matches and
    moved mean IoU of matches from 0.772 to 0.759.

    Re-merging is a no-op unless geometry changes, because greedy merging is idempotent -- feeding
    the same boxes back produces the same groups. Measured at 1, 2 and 3 rounds without fusion the
    results were identical to three decimals.

    The under-merge they were meant to fix is real (roughly 800 retained boxes sit on an instance
    another box already describes) but it is not reachable by moving boxes around: those pairs
    disagree by more than the merge threshold and are not view-edge truncated, and the looser
    criterion that would capture them also collapses crowd neighbours. It needs boundaries refined
    against pixels, not averaged between observations.

    Returns the member indices of each cluster, a representative observation per cluster, the merge
    reason per absorbed observation, and the number of rounds actually used.
    """
    clusters=[[i] for i in range(len(observations))]
    reps=[dict(o) for o in observations]
    reasons={};used=0
    for _ in range(max(1,rounds)):
        groups,round_reasons=merge_observations(reps,ios_threshold,iou_threshold,ontology)
        used+=1
        if len(groups)==len(reps):break
        merged_clusters=[];merged_reps=[]
        for group in groups:
            members=[m for idx in group for m in clusters[idx]]
            chosen=[reps[i] for i in group]
            lead=chosen[0]
            merged_reps.append({**lead,
                'bbox_xyxy':fuse_boxes(chosen) if fuse else lead['bbox_xyxy'],
                'model_score':max(o['model_score'] for o in chosen),
                # Truncated only if every contributing view saw it truncated: one unclipped view
                # is enough to support the boundary.
                'view_edge':all(o['view_edge'] for o in chosen)})
            merged_clusters.append(members)
            for idx in group[1:]:
                for m in clusters[idx]:reasons[m]=round_reasons.get(idx,'iou_agreement')
        clusters,reps=merged_clusters,merged_reps
    return clusters,reps,reasons,used


DEFAULT_REFINE_IOU=0.7


def verify_objects(objects,evidence,candidates,verifier,image,ontology,vocabulary_id,match_iou):
    """Record whether an independent model detects something at each retained box.

    Confirmation is evidence and not truth. Two models of overlapping lineage agree partly because
    they fail alike, so this supports a one-sided reading only: a confirmed box has independent
    support, an unconfirmed box is either absent or missed by the verifier, and nothing here
    distinguishes those two. Acceptance may require confirmation; nothing is removed.
    """
    from .vocabulary import load_vocabulary,resolve_prompts
    vocabulary=load_vocabulary(vocabulary_id) if vocabulary_id else None
    resolution=resolve_prompts(ontology,vocabulary)
    prompts={phrase:class_id for class_id,phrase in zip(resolution['classes'],resolution['prompts'])}
    detections=verifier.detect(image,prompts)
    records=[]
    for obj,ev,candidate in zip(objects,evidence,(c for c in candidates if c['disposition']=='retained')):
        best,best_iou=None,0.0
        for d in detections:
            value=iou(obj['bbox_xyxy'],d['bbox_xyxy'])
            if value>=match_iou and value>best_iou: best,best_iou=d,value
        record={'object':obj['id'],'confirmed':best is not None,
                'verifier_label':best['label'] if best else None,
                'verifier_score':round(best['score'],6) if best else None,
                'overlap':round(best_iou,6) if best else None,
                'label_agrees':bool(best and best['label']==obj['label']['id'])}
        ev['verified']=record['confirmed'];ev['verifier_score']=record['verifier_score']
        candidate['verification']=record
        if not record['confirmed']:
            obj['uncertainty']['reasons'].append('unconfirmed_by_independent_model')
        records.append(record)
    return {'detections':len(detections),'confirmed':sum(r['confirmed'] for r in records),
            'objects':len(records),'match_iou':match_iou,
            'independence':'shared_transformer_lineage; agreement is reproducibility, not confirmation'}


def refine_objects(objects,evidence,candidates,refiner,image,tolerance,max_alternatives):
    """Replace proposed geometry with refined geometry, keeping the proposal inspectable.

    Refinement is a second model acting on the first model's output. It is a claim about the same
    instance, so nothing is added or removed here and no acceptance decision is taken.

    Where the two disagree about extent beyond the tolerance, the refined box is kept as the best
    estimate and the proposed box becomes an alternative, with the boundary dimension unresolved.
    Two models agreeing where an object is and disagreeing how far it extends is boundary
    uncertainty, and silently overwriting one with the other would discard that.
    """
    results=refiner.refine(image,[list(o['bbox_xyxy']) for o in objects])
    refined_count=disagreements=0
    for obj,ev,result in zip(objects,evidence,results):
        ev['refined']=False
        if result is None:continue
        proposed=[float(v) for v in obj['bbox_xyxy']]
        try: refined=validate_box(result['bbox_xyxy'],*image.size)
        except ValueError:
            candidates.append({'id':f"refine-{obj['id']}",'disposition':'rejected','object':obj['id'],
                               'proposed_bbox_xyxy':proposed,'bbox_xyxy':list(result['bbox_xyxy']),
                               'reason':'invalid_refined_geometry'})
            continue
        agreement=iou(proposed,refined)
        candidates.append({'id':f"refine-{obj['id']}",'disposition':'refined','object':obj['id'],
                           'proposed_bbox_xyxy':proposed,'bbox_xyxy':refined,
                           'mask_quality':result['mask_quality'],'job':result.get('view'),
                           'agreement_iou':round(float(agreement),6),
                           'reason':'boundary_refined_against_pixels'})
        obj['bbox_xyxy']=refined
        ev['refined']=True;refined_count+=1
        if agreement<tolerance:
            disagreements+=1
            obj['alternative_boxes']=([proposed]+[a for a in obj['alternative_boxes'] if a!=proposed])[:max_alternatives]
            ev['alternatives']=[proposed]+[a for a in ev['alternatives'] if a!=proposed]
            obj['uncertainty']['boundary']='unresolved'
            if 'refiner_extent_disagreement' not in obj['uncertainty']['reasons']:
                obj['uncertainty']['reasons'].append('refiner_extent_disagreement')
    return {'refined':refined_count,'extent_disagreements':disagreements,'tolerance':tolerance}


DEFAULT_NAME_RATIO=0.6


def resolve_label(class_scores,ontology,ratio):
    """Choose the most specific label the scores support, a broader one, or no class at all.

    Comparison is relative, not absolute: detector scores here are sigmoid outputs an order of
    magnitude apart between easy and hard views, so a fixed score gap would behave differently at
    each scale. The runner-up counts as competing when it reaches `ratio` of the winner.

    When two competing classes are siblings, their lowest common ancestor is the most specific
    claim the evidence actually supports, which is what the annotation policy asks for. If that
    ancestor is not a useful class the object keeps its geometry and takes `unknown_object`; a
    coin-flip between siblings is never emitted as a specific label.
    """
    ranked=sorted(class_scores.items(),key=lambda kv:(-kv[1],kv[0]))
    top,top_score=ranked[0]
    if len(ranked)==1 or top_score<=0 or ranked[1][1]/top_score<ratio:
        return top,[],'specific_label_clear_of_runner_up'
    runner=ranked[1][0]
    if co_occurring(ontology,top,runner):
        # Both may be true of this region, so there is no conflict to resolve. Falling back here
        # would discard a supported class to settle a disagreement that was never one.
        return top,[top,runner],'co_occurring_classes_not_competing'
    ancestor=common_ancestor(ontology,top,runner)
    if ancestor in (top,runner):
        # One is an ancestor of the other; the broader of the pair is the supported claim.
        return ancestor,[top,runner],'broader_of_nested_pair'
    if ancestor is not None and is_useful(ontology,ancestor):
        return ancestor,[top,runner],'broader_class_covers_competing_siblings'
    return 'unknown_object',[top,runner],'no_useful_common_ancestor'


def name_observations(observations,ontology,ratio):
    """Apply conservative naming before merging, so two views that disagree on a specific class
    but agree on the broader one describe the same instance instead of splitting into two."""
    for obs in observations:
        scores=obs.get('class_scores')
        if not scores:continue
        label,alternatives,reason=resolve_label(scores,ontology,ratio)
        obs['label']=label;obs['label_alternatives']=alternatives;obs['naming_reason']=reason
    return observations


def load_calibration(path,ontology_id):
    """Load a calibration artifact and decide whether it applies to this run.

    A calibration maps an evidence value to a probability for one exactly defined event, and it is
    only meaningful on the population it was fitted on. A fit from another domain does not
    establish reliability here, so a mismatch yields no estimates and a recorded reason rather
    than a transferred number.
    """
    if path is None:
        return None,{'status':'inapplicable','reason':'no_calibration_artifact_staged'}
    spec=json.loads(Path(path).read_text())
    required={'id','version','event','population','domain','feature','bins'}
    missing=required-spec.keys()
    if missing:
        return None,{'status':'inapplicable','reason':'malformed_calibration:'+','.join(sorted(missing))}
    if spec['domain']!=ontology_id:
        return None,{'status':'inapplicable','reason':f'calibration_domain_mismatch:{spec["domain"]}!={ontology_id}'}
    if spec['feature']!='model_score':
        return None,{'status':'inapplicable','reason':'unsupported_calibration_feature:'+spec['feature']}
    return spec,{'status':'applied','reason':'calibration_domain_matches_run',
                 'calibration_id':spec['id'],'calibration_version':spec['version'],
                 'event':spec['event'],'population':spec['population']}


def calibrated_probability(spec,value):
    """Probability for the calibration's event, or None when the value falls outside every bin."""
    for low,high,probability in spec['bins']:
        if low<=value<high or (value>=high and high>=1.0):
            return float(probability)
    return None


ACCEPTANCE_DEFAULTS={'score':0.5,'min_views':2,'boundary_iou':0.8,'granularity_ios':0.7,
                     'require_verification':False}


def assess(objects,evidence,planned_views,rules):
    """Assign uncertainty dimensions from recorded evidence under frozen rules.

    Model scores are raw detector outputs, so a threshold on one is a declared acceptance rule and
    never a probability of correctness; no probability field is emitted anywhere. Agreement across
    views comes from one checkpoint looking twice, which is evidence that a detection reproduces,
    not independent confirmation -- the reason codes say which is which.

    A dimension becomes `supported` only when a check ran and passed. Where the implemented routes
    cannot assess a dimension at all it stays unresolved, which keeps the object uncertain.
    """
    areas=[(o['bbox_xyxy'][2]-o['bbox_xyxy'][0])*(o['bbox_xyxy'][3]-o['bbox_xyxy'][1]) for o in objects]
    # Agreement cannot be demanded from a schedule that never planned a second look.
    required_views=min(rules['min_views'],planned_views)
    for index,(obj,ev) in enumerate(zip(objects,evidence)):
        reasons=[];states={}
        strong=ev['score']>=rules['score']
        # Independent confirmation is evidence a second model saw something here. It is not truth,
        # and an unconfirmed box may be absent or merely missed, so it withholds support rather
        # than asserting the box is wrong.
        unconfirmed=rules.get('require_verification') and ev.get('verified') is False
        states['existence']='supported' if strong and ev['views']>=required_views and not unconfirmed else 'unresolved'
        if states['existence']=='supported':
            reasons.append(f'reproduced_in_{ev["views"]}_views' if ev['views']>1 else 'single_view_schedule')
            if ev.get('verified'):reasons.append('confirmed_by_independent_model')
        elif unconfirmed and strong and ev['views']>=required_views:
            reasons.append('unconfirmed_by_independent_model')
        else:
            reasons.append('weak_score' if not strong else 'insufficient_view_agreement')
        states['class']='supported' if strong and ev['useful'] else 'unresolved'
        if not ev['useful']:reasons.append('no_useful_class')
        if ev['view_edge']:
            states['boundary']='unresolved';reasons.append('view_edge_truncation_unresolved')
        elif ev['alternatives'] and min(iou(obj['bbox_xyxy'],a) for a in ev['alternatives'])<rules['boundary_iou']:
            states['boundary']='unresolved';reasons.append('multi_view_boundary_disagreement')
        else:
            states['boundary']='supported'
        # Count is not estimated: an `instance` is one object by construction, and no group
        # hypothesis is produced, so nothing is inferred from density.
        states['count']='supported' if obj['kind']=='instance' else 'unresolved'
        container=next((j for j,other in enumerate(objects)
                        if j!=index and areas[j]>areas[index]
                        and ios(obj['bbox_xyxy'],other['bbox_xyxy'])>=rules['granularity_ios']),None)
        if container is None:
            states['granularity']='supported';reasons.append('no_competing_containment_in_discovered_set')
        else:
            states['granularity']='unresolved';reasons.append(f'possible_part_of:{objects[container]["id"]}')
        obj['uncertainty']={**states,'reasons':reasons}
        obj['status']='accepted' if all(v=='supported' for v in states.values()) and ev['useful'] else 'uncertain'
    return objects


def annotate(image_path,bundle_path,output,adapter=None,parent_run=None,
             tile_levels=0,tile_overlap=0.25,merge_ios=0.6,merge_iou=0.5,device=None,threshold=0.1,
             ontology_id=None,acceptance=None,name_ratio=DEFAULT_NAME_RATIO,
             merge_rounds=1,fuse_boxes_enabled=False,calibration=None,max_alternatives=8,
             refiner=None,refine_bundle=None,refine_iou=DEFAULT_REFINE_IOU,refine_tile_side=512,
             merge_labels='common-ancestor',vocabulary_id=None,
             verifier=None,verify_bundle=None,verify_iou=0.5,verify_threshold=0.3):
    started=time.perf_counter()
    image,metadata=normalize_image(image_path)
    bundle=preflight(bundle_path)
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    run_id=str(uuid.uuid4());artifacts={}
    ontology=load_ontology(ontology_id)
    calibration_spec,calibration_record=load_calibration(calibration,ontology['id'])
    jobs=plan_tiles(*image.size,levels=tile_levels,overlap=tile_overlap)
    profile='owlv2-single-pass-v1' if tile_levels==0 else f'owlv2-tiled-L{tile_levels}-v1'
    rules={**ACCEPTANCE_DEFAULTS,**acceptance} if acceptance is not None else None
    config={'profile':profile,'threshold':threshold,'verification':'none','device':device or 'auto',
            'tile_levels':tile_levels,'tile_overlap':tile_overlap,'merge_ios':merge_ios,'merge_iou':merge_iou,
            'planned_views':len(jobs),'max_alternatives':max_alternatives,'ontology':ontology['id'],'vocabulary':vocabulary_id,
            'verify_bundle':str(Path(verify_bundle).resolve()) if verify_bundle else None,
            'verify_iou':verify_iou,'verify_threshold':verify_threshold,
            'acceptance':rules,'name_ratio':name_ratio,
            'refine_bundle':str(Path(refine_bundle).resolve()) if refine_bundle else None,
            'refine_iou':refine_iou,'refine_tile_side':refine_tile_side,
            'merge_rounds':merge_rounds,'box_fusion':fuse_boxes_enabled,'merge_labels':merge_labels,
            'calibration':str(calibration) if calibration else None}
    manifest={'run_id':run_id,'parent_run':parent_run,'input_path':str(Path(image_path).resolve()),'bundle_path':str(Path(bundle_path).resolve()),'image':metadata,'bundle_sha256':bundle['manifest_sha256'],'policy':identity(load_policy()),'ontology':identity(ontology),'prompts':[x['name'] for x in ontology['classes'] if x['useful']],'config':config,'environment':doctor()}
    manifest['sha256']=digest(canonical(manifest))
    artifacts['manifest']=write_artifact(root,'manifest.json',manifest)
    objects=[];evidence=[];candidates=[];observations=[];completed=[];failed=[]
    error=None;capabilities=None;rounds_used=0;verification=None;verifier_capabilities=None;refinement=None;refiner_capabilities=None
    try:
        if adapter is None:
            from .adapters.owlv2 import Owlv2Adapter
            adapter=Owlv2Adapter(bundle_path,device=device,threshold=threshold,ontology_id=ontology['id'],
                                 vocabulary_id=vocabulary_id)
        capabilities=adapter.capabilities() if hasattr(adapter,'capabilities') else None
        observations,completed,failed=discover(adapter,image,jobs)
        labels={c['id']:c for c in ontology['classes']}
        for obs in observations:
            if obs['label'] not in labels:raise ValueError('adapter_invalid_label')
        name_observations(observations,ontology,name_ratio)
        groups,representatives,merge_reasons,rounds_used=consolidate(
            observations,merge_ios,merge_iou,rounds=merge_rounds,fuse=fuse_boxes_enabled,
            ontology=ontology if merge_labels=='common-ancestor' else None)
        for index,(group,lead) in enumerate(zip(groups,representatives)):
            try: box=validate_box(lead['bbox_xyxy'],*image.size)
            except ValueError:
                for member in group:
                    candidates.append({**observations[member],'disposition':'rejected','reason':'invalid_source_geometry'})
                continue
            alternatives=[]
            for member in (m for m in group if m!=group[0]):
                other=observations[member]
                candidates.append({**other,'disposition':'merged','merged_into':lead['id'],
                                   'reason':merge_reasons[member]})
                try: alternative=validate_box(other['bbox_xyxy'],*image.size)
                except ValueError: continue
                if alternative!=box and alternative not in alternatives:alternatives.append(alternative)
            alternatives=alternatives[:max_alternatives]
            candidates.append({**observations[group[0]],'disposition':'retained',
                               'reason':'unverified_model_proposal','bbox_xyxy':box,
                               'observed_bbox_xyxy':observations[group[0]]['bbox_xyxy'],
                               'fused_from':len(group),'model_score':lead['model_score'],
                               'view_edge':lead['view_edge'],
                               'merged_views':sorted({observations[m]['job'] for m in group})})
            reasons=['unverified_model_proposal']
            if alternatives:reasons.append('multi_view_boundary_disagreement')
            if lead['view_edge']:reasons.append('view_edge_truncation_unresolved')
            uncertainty={k:'not_assessed' for k in ['existence','count','boundary','class','granularity']}
            if alternatives or lead['view_edge']:uncertainty['boundary']='unresolved'
            evidence.append({'score':lead['model_score'],'views':len({observations[m]['job'] for m in group}),
                             'view_edge':lead['view_edge'],'alternatives':alternatives,
                             'useful':labels[lead['label']]['useful']})
            objects.append({'id':f'obj-{index}','kind':'instance','status':'uncertain','scene_layer':'physical',
                'bbox_xyxy':box,'mask_ref':None,'alternative_boxes':alternatives,
                'label':{'id':lead['label'],'name':labels[lead['label']]['name']},
                'label_alternatives':[{'id':a,'name':labels[a]['name']} for a in lead.get('label_alternatives',[])],
                'uncertainty':{**uncertainty,'reasons':reasons},
                'evidence_refs':['evidence/candidates.jsonl','evidence/coverage.json'],'relations':[],
                'occluded':None,'truncated':None,'flag_evidence_refs':[],'count':None,'calibrated_estimates':[]})
        if refine_bundle is not None and refiner is None:
            from .adapters.sam3 import Sam3Refiner
            refiner=Sam3Refiner(refine_bundle,device=device,tile_side=refine_tile_side)
        if refiner is not None and objects:
            refiner_capabilities=refiner.capabilities() if hasattr(refiner,'capabilities') else None
            refinement=refine_objects(objects,evidence,candidates,refiner,image,refine_iou,max_alternatives)
        if verify_bundle is not None and verifier is None:
            from .adapters.sam3 import Sam3Verifier
            verifier=Sam3Verifier(verify_bundle,device=device,threshold=verify_threshold)
        if verifier is not None and objects:
            verifier_capabilities=verifier.capabilities() if hasattr(verifier,'capabilities') else None
            verification=verify_objects(objects,evidence,candidates,verifier,image,ontology,vocabulary_id,verify_iou)
        if rules is not None:assess(objects,evidence,len(jobs),rules)
        if calibration_spec is not None:
            for obj,ev in zip(objects,evidence):
                probability=calibrated_probability(calibration_spec,ev['score'])
                if probability is None:continue
                obj['calibrated_estimates']=[{'event':calibration_spec['event'],
                    'probability':probability,'calibration_ref':'evidence/calibration.json',
                    'population':calibration_spec['population'],
                    'localization_threshold':calibration_spec.get('localization_threshold')}]
    except (Exception,KeyboardInterrupt) as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    if error and error['type']=='KeyboardInterrupt':status='interrupted'
    elif error:status='failed'
    else:status='completed'
    search_status=status if error else ('failed' if failed else 'profile_complete')
    reason=('baseline_finished' if not failed else 'view_failures_prevent_complete_scan') if not error else error['type']
    stage_resources={'wall_seconds':round(time.perf_counter()-started,6),'peak_rss_bytes':peak_rss_bytes(),
                     'model_calls':len(completed),'planned_views':len(jobs),'failed_views':len(failed),
                     'raw_observations':len(observations),'retained_objects':len(objects)}
    events=[{'stage':'discovery','status':status,'error':error,'capabilities':capabilities,'resources':stage_resources},
            {'stage':'verification','status':status,'summary':verification,
             'capabilities':verifier_capabilities,'resources':stage_resources},
            {'stage':'merge','status':status,'observations':len(observations),'retained':len(objects),
             'acceptance_rule':rules,
             'rounds_used':rounds_used,'box_fusion':fuse_boxes_enabled,
             'rule':{'same_label_only':True,'iou_threshold':merge_iou,
                     'ios_threshold':merge_ios,'ios_requires_view_edge':True},'resources':stage_resources},
            {'stage':'refine','status':status,'refinement':refinement,
             'capabilities':refiner_capabilities,'resources':stage_resources},
            {'stage':'terminal','status':status,'reason':reason,'resources':stage_resources}]
    artifacts['events']=write_artifact(root,'events.jsonl',b'\n'.join(canonical(e) for e in events)+b'\n','application/x-ndjson')
    artifacts['candidates']=write_artifact(root,'evidence/candidates.jsonl',b'\n'.join(canonical(c) for c in candidates)+b'\n','application/x-ndjson')
    done=[j for j in jobs if j['id'] in set(completed)]
    coverage={'planned_jobs':[{**j,'route':'owlv2'} for j in jobs],'completed_jobs':completed,
              'failed_jobs':failed,'route_covered_area_fraction':{'owlv2':round(covered_fraction(done,*image.size),6)},
              'meaning':'geometric processing coverage; not semantic recall'}
    artifacts['coverage']=write_artifact(root,'evidence/coverage.json',coverage)
    if calibration_spec is not None:
        artifacts['calibration']=write_artifact(root,'evidence/calibration.json',calibration_spec)
    regions=[{'kind':'unsearched','bbox_xyxy':j['bbox_xyxy'],'reason':'route_failed_for_view:'+j['id'],
              'evidence_refs':['evidence/coverage.json']} for j in failed]
    if error:
        regions.append({'kind':'unsearched','bbox_xyxy':[0,0,*image.size],'reason':'route_failed',
                        'evidence_refs':['evidence/coverage.json']})
    else:
        regions.append({'kind':'unresolved','bbox_xyxy':[0,0,*image.size],
                        'reason':'scan_cannot_establish_absence_or_complete_discovery','evidence_refs':['evidence/coverage.json']})
    artifacts['regions']=write_artifact(root,'evidence/regions.json',regions)
    artifacts['overlay']=write_artifact(root,'overlay.png',render_overlay(image,objects),'image/png')
    artifacts['report']=write_artifact(root,'report.html',render_report(objects,regions,status,coverage,profile),'text/html')
    coverage_summary={'planned_jobs':len(jobs),'completed_jobs':len(completed),'failed_jobs':len(failed),
                      'route_covered_area_fraction':coverage['route_covered_area_fraction']}
    # Evidence bytes only; annotation.json and its digest sidecar are written after this point.
    resources={**stage_resources,'artifact_bytes':sum((root/a['path']).stat().st_size for a in artifacts.values())}
    result={'schema_version':'lynceus.annotation/1.0','run_id':run_id,'image':metadata,'policy':identity(load_policy()),'ontology':identity(ontology),'manifest_ref':'manifest.json','qualification':{'id':None,'domain':'undeclared','status':'unqualified'},'execution':{'status':status,'reason':reason,'failure_refs':['events.jsonl'] if error else []},'search':{'profile_id':profile,'status':search_status,'mandatory_planned':len(jobs),'mandatory_completed':len(completed),'mandatory_failed':len(failed),'stable_rounds':0,'coverage_ref':'evidence/coverage.json','stop_rule':{'full_image_passes':1,'tile_levels':tile_levels,'tile_overlap':tile_overlap,'merge_ios':merge_ios,'merge_iou':merge_iou,'required_stable_rounds':0}},'objects':objects,'regions':regions,'summary':summarize(objects,regions,coverage_summary,resources,calibration_record),'artifacts':artifacts}
    result=validate_annotation(result,root)
    write_artifact(root,'annotation.json',result)
    # External sidecar anchors the canonical annotation for safe accidental-tamper detection.
    write_artifact(root,'annotation.sha256',digest((root/'annotation.json').read_bytes()).encode(),'text/plain')
    return result


def resume(run,output=None,adapter=None):
    root=Path(run)
    raw=(root/'annotation.json').read_bytes()
    if digest(raw)!=(root/'annotation.sha256').read_text():raise ValueError('annotation_hash_mismatch')
    result=validate_annotation(json.loads(raw),root)
    manifest=json.loads((root/result['manifest_ref']).read_text())
    stored=manifest.pop('sha256')
    if digest(canonical(manifest))!=stored:raise ValueError('manifest_hash_mismatch')
    image,metadata=normalize_image(manifest['input_path'])
    if metadata!=result['image']:raise ValueError('input_changed')
    bundle=preflight(manifest['bundle_path'])
    if bundle['manifest_sha256']!=manifest['bundle_sha256']:raise ValueError('bundle_changed')
    if doctor()!=manifest['environment']:raise ValueError('environment_changed')
    if result['execution']['status']=='completed':return result
    if output is None:raise ValueError('resume_requires_new_output_for_interrupted_run')
    config=manifest['config']
    return annotate(manifest['input_path'],manifest['bundle_path'],output,adapter=adapter,parent_run=result['run_id'],
                    tile_levels=config['tile_levels'],tile_overlap=config['tile_overlap'],merge_ios=config['merge_ios'],merge_iou=config['merge_iou'],
                    threshold=config['threshold'],ontology_id=config['ontology'],
                    acceptance=config['acceptance'],name_ratio=config['name_ratio'],
                    merge_rounds=config['merge_rounds'],fuse_boxes_enabled=config['box_fusion'],
                    calibration=config['calibration'],
                    max_alternatives=config['max_alternatives'],
                    merge_labels=config.get('merge_labels','exact'),vocabulary_id=config.get('vocabulary'),
                    refine_bundle=config.get('refine_bundle'),refine_iou=config.get('refine_iou',DEFAULT_REFINE_IOU),
                    refine_tile_side=config.get('refine_tile_side',512))
