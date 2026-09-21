"""Independent recorded-prediction evaluation; never grants qualification."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import iou,validate_box
from .policy import load_ontology


def iou_matrix(pred,refs):
    return np.array([[iou(p['bbox_xyxy'],r['bbox_xyxy']) for r in refs] for p in pred],dtype=float).reshape(len(pred),len(refs))


def match_matrix(values,valid):
    """One-to-one maximum matching over a precomputed IoU matrix and its validity mask.

    The frozen matching rule lives here alone. Diagnostics reuse it over subsets of one matrix
    rather than reimplementing it, so a diagnostic cannot quietly score under a different rule
    from the evaluator it is meant to explain.
    """
    if not values.size:return []
    # A valid edge dominates the total possible secondary IoU gain.
    rows,cols=linear_sum_assignment(-(valid*(min(values.shape)+1)+values*valid))
    return [(int(a),int(b),float(values[a,b])) for a,b in zip(rows,cols) if valid[a,b]]


def match(pred,refs,threshold,labelled=False):
    if not pred or not refs:return []
    values=iou_matrix(pred,refs)
    valid=values>=threshold
    if labelled:
        valid &= np.array([[p['label'] in r['permitted_labels'] and p['label'] not in ('entity','unknown_object') for r in refs] for p in pred])
    return match_matrix(values,valid)


def evaluate_benchmark(payload,bootstrap_replicates=10000,seed=0):
    if bootstrap_replicates<1:raise ValueError('invalid_bootstrap_replicates')
    if payload.get('schema_version')!='lynceus.benchmark/1.0':raise ValueError('invalid_benchmark_schema')
    if not payload.get('frozen') or payload.get('reference_provenance') not in ['independent','synthetic_fixture'] or not payload.get('policy_compatible'):raise ValueError('incompatible_reference')
    images=payload['images']
    if not images:raise ValueError('empty_benchmark')
    seen={}; ids=set(); ontology=load_ontology(payload.get('ontology')); labels={c['id'] for c in ontology['classes']}; rows=[]; distributions=[]
    # A finite-category reference annotates only some classes, so a correct prediction of an
    # unannotated class is not a scoreable false positive. Declaring that scope in advance is the
    # documented-mapping route required before such a reference may be used at all; without it the
    # evaluator scores every retained instance, as an exhaustive reference demands.
    # A prediction-assisted panel contains only what the system proposed, so recall against it is
    # an artifact of its construction rather than a measurement. It is reported as anchored.
    anchored=payload.get('recall_interpretable') is False
    ANCHORED_METRICS=('localized_instance_recall','useful_label_coverage','image_completeness','fully_correct_images')
    scope=payload.get('category_scope')
    if scope is not None:
        if not scope or not set(scope)<=labels:raise ValueError('invalid_category_scope')
        if not payload.get('category_scope_reason'):raise ValueError('missing_category_scope_reason')
        scope=set(scope)
    for im in images:
        if im['id'] in ids:raise ValueError('duplicate_image_id')
        ids.add(im['id'])
        if im['split'] not in ['development','calibration','evaluation']:raise ValueError('invalid_split')
        for key in ['sha256','scene_id','near_duplicate_group']:
            value=im.get(key)
            if not value:raise ValueError(f'missing_leakage_identity: {key}')
            token=(key,value)
            if token in seen and seen[token]!=im['split']:raise ValueError('split_leakage')
            seen[token]=im['split']
        w,h=im['size']; refs=im['references']; pred=im.get('predictions',[])
        for records in [refs,pred]:
            local_ids=set()
            for r in records:
                if r['id'] in local_ids:raise ValueError('duplicate_record_id')
                local_ids.add(r['id']);validate_box(r['bbox_xyxy'],w,h)
        for r in refs:
            if not r['permitted_labels'] or not set(r['permitted_labels'])<=labels:raise ValueError('invalid_reference_mapping')
        for p in pred:
            if p['label'] not in labels or p['status'] not in ['accepted','uncertain'] or p['kind'] not in ['instance','part','group','stuff'] or p['scene_layer'] not in ['physical','depicted','reflected']:raise ValueError('invalid_prediction')
            if p['status']=='accepted' and p['label'] in ['unknown_object','entity']:raise ValueError('invalid_accepted_label')
        status=im.get('run_status','missing')
        if status not in ['completed','failed','interrupted','missing']:raise ValueError('invalid_run_status')
        if status!='completed' and pred:raise ValueError('failed_run_has_predictions')
        if im['split']!='evaluation':continue
        eligible=[r for r in refs if r.get('eligible',True)]
        final=[p for p in pred if p['kind']=='instance' and p['scene_layer']=='physical']
        # unknown_object and entity make no category claim, so they stay in every precision
        # denominator; otherwise abstaining on everything would buy free precision (gate G6).
        scored=final if scope is None else [p for p in final if p['label'] in scope or p['label'] in ('unknown_object','entity')]
        out_of_scope=len(final)-len(scored)
        accepted=[p for p in scored if p['status']=='accepted']
        m50=match(final,eligible,.5);s50=match(scored,eligible,.5)
        a75=match(accepted,eligible,.75,True);a50=match(accepted,eligible,.5,True);a85=match(accepted,eligible,.85,True)
        correct=sum(final[a]['label'] in eligible[b]['permitted_labels'] and final[a]['label'] not in ['unknown_object','entity'] for a,b,_ in m50)
        matched_indices={a for a,_,_ in s50}
        duplicate=sum(i not in matched_indices and any(iou(p['bbox_xyxy'],r['bbox_xyxy'])>=.5 for r in eligible) for i,p in enumerate(scored))
        measures={'localized_instance_recall':[len(m50),len(eligible)],'final_instance_precision':[len(s50),len(scored)],'accepted_annotation_precision':[len(a75),len(accepted)],'useful_label_coverage':[len(a75),len(eligible)],'localization_quality':[len(a85),len(a50)],'classification_accuracy':[correct,len(m50)],'duplicate_rate':[duplicate,len(scored)],'image_completeness':[int(status=='completed' and len(m50)==len(eligible)),1],'fully_correct_images':[int(status=='completed' and len(a85)==len(eligible)==len(scored)),1],'unknown_rate':[sum(p['label']=='unknown_object' for p in final),len(final)]}
        rows.append((im['domain'],measures));distributions.append({'image_id':im['id'],'run_status':status,'metrics':measures,'localized_ious':[v for _,_,v in m50],'uncertain_objects':sum(p['status']=='uncertain' for p in pred),'groups':sum(p['kind']=='group' for p in pred),'excluded_references':len(refs)-len(eligible),'out_of_scope_predictions':out_of_scope})
    if not rows:raise ValueError('no_evaluation_images')
    rng=np.random.default_rng(seed)
    domains=defaultdict(list)
    for i,(domain,_) in enumerate(rows):domains[domain].append(i)
    metrics={}; per_domain={}
    eligibility='evaluation split; physical instances; failed and missing runs retain reference denominators'
    if scope is not None:eligibility+='; precision denominators exclude predictions labelled outside the declared category scope'
    for name in rows[0][1]:
        data=np.array([m[name] for _,m in rows],dtype=float);n,d=data.sum(axis=0);samples=[]
        for _ in range(bootstrap_replicates):
            selected=np.concatenate([rng.choice(indices,len(indices)) for indices in domains.values()]);bn,bd=data[selected].sum(axis=0)
            if bd:samples.append(bn/bd)
        degenerate=not samples or np.ptp(samples)==0
        metrics[name]={'numerator':int(n),'denominator':int(d),'value':float(n/d) if d else None,'descriptive_lower_95':float(np.quantile(samples,.05)) if samples and not degenerate else None,'interval_status':'inconclusive_degenerate' if degenerate else 'descriptive_only','eligibility':eligibility}
        if anchored and name in ANCHORED_METRICS:
            metrics[name]['interval_status']='not_interpretable_anchored_panel'
            metrics[name]['descriptive_lower_95']=None
            metrics[name]['eligibility']=('anchored panel: every reference is something the system proposed, so this '
                                          'number reflects the panel\'s construction and not the system\'s recall')
        for domain,indices in domains.items():
            dn,dd=data[indices].sum(axis=0);per_domain.setdefault(domain,{})[name]={'numerator':int(dn),'denominator':int(dd),'value':float(dn/dd) if dd else None}
    return {'schema_version':'lynceus.evaluation/1.0','reference_provenance':payload['reference_provenance'],
            'anchoring':payload.get('anchoring','unspecified'),'recall_interpretable':not anchored,'ontology':ontology['id'],'qualification':{'status':'inconclusive','reasons':(['foundation evaluator does not implement all release gates','no preregistered joint qualification protocol','synthetic fixtures are not quality evidence'] if payload['reference_provenance']=='synthetic_fixture' else ['challenge strata, paired ablations, calibration and joint gates remain unimplemented'])+(['panel is prediction-assisted: recall and coverage are not interpretable from it'] if anchored else [])},'metrics':metrics,'per_domain':per_domain,'per_image':distributions,'bootstrap':{'unit':'image','stratified_by':'domain','replicates':bootstrap_replicates,'seed':seed,'joint_qualification_bounds':False},'category_scope':sorted(scope) if scope else None,'category_scope_reason':payload.get('category_scope_reason'),'out_of_scope_predictions':sum(d['out_of_scope_predictions'] for d in distributions),'missing_or_failed_images':sum(d['run_status']!='completed' for d in distributions),'unimplemented_metrics':['merge/split rates','boundary errors','broad-parent rates','risk-coverage sweep','challenge strata','operational cost','paired ablations']}
