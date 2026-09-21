"""Diagnostics that stay interpretable on a finite-category reference.

Most measurement against a reference that annotates only some classes is blocked by one fact: an
unmatched box is not evidence of a false box, so precision is a lower bound of unknown tightness
and any change that adds boxes cannot be judged. Two questions survive that blocker, and this
module answers both.

**Which mechanism lost each missed reference.** Misses are fully determined by the reference: a
resolvable reference instance either was localized or was not, and nothing about unannotated
objects enters the question. Attributing each miss to a mechanism is therefore valid here, and it
decides whether a recall gap is a detector limit at all.

**Whether a scorer that removes boxes discriminates.** What it removes among matched boxes is
exactly measurable; what it removes among unmatched boxes is of unknown value. That asymmetry is
enough, because a pruner removing at random removes both classes at the same rate.
"""
import numpy as np
from .evaluation import iou_matrix,match_matrix
from .geometry import iou

# Ordered from cheapest repair to most expensive. Only the last is a case for another detector.
MISS_BUCKETS=('merged_with_neighbour','localized_below_match','below_threshold','merged_away',
              'retained_unmatched','never_proposed')

# evaluation.md item 7: original-resolution equivalent side length, sqrt(visible area).
SIZE_BINS=((16.0,'<16'),(32.0,'16-32'),(96.0,'32-96'),(float('inf'),'>=96'))


def size_bin(box):
    side=(max(0.0,box[2]-box[0])*max(0.0,box[3]-box[1]))**0.5
    return next(name for edge,name in SIZE_BINS if side<edge)


def attribute_misses(references,retained,probe,shipped_threshold,match_iou=0.5,near_iou=0.3):
    """Assign every unmatched eligible reference to the mechanism that lost it.

    `retained` is the shipped run's retained set and decides what counts as a miss. `probe` is the
    candidate store of a second run of the same schedule at a lower score threshold, which is the
    only way to see observations the shipped threshold removed inside the adapter. Each probe
    record carries `model_score` and a `disposition` of retained, merged or rejected.

    The buckets, and the repair each implies:

    `merged_with_neighbour`  a retained box covers this reference at the match criterion but the
        one-to-one assignment gave it to another reference, so two references share one box. A
        separation failure, repaired in instance resolution.
    `localized_below_match`  something reached the reference but no box cleared the criterion. A
        boundary failure, repaired by refinement, not by proposing more.
    `below_threshold`        the object was proposed and every proposal scored under the shipped
        threshold. A ranking failure, repaired for free by moving the threshold -- at a precision
        cost this diagnostic deliberately does not estimate, because on this reference it cannot.
    `merged_away`            a proposal cleared the threshold and was absorbed into a cluster whose
        representative does not cover this reference. A reconciliation failure.
    `retained_unmatched`     a probe proposal cleared the threshold, was retained, and still did
        not reach the shipped set. The two runs disagree; inspect before drawing a conclusion.
    `never_proposed`         nothing reached the reference at any score in any view. The only
        bucket that is evidence of a detector blind spot, and the only one another route can fix.

    Bucket availability is an adapter property. A route decoding a fixed number of queries can
    also lose an object to cap saturation; a route emitting one prediction per patch token above a
    score threshold cannot, so no cap bucket exists here and its absence is not a finding.
    """
    if probe is None:raise ValueError('probe_candidates_required')
    values=iou_matrix(retained,references) if retained and references else np.zeros((len(retained),len(references)))
    hit={b for _,b,_ in match_matrix(values,values>=match_iou)}
    probe_values=iou_matrix(probe,references) if probe and references else np.zeros((len(probe),len(references)))
    misses=[]
    for index,reference in enumerate(references):
        if index in hit:continue
        covering=values[:,index] if values.size else np.zeros(0)
        best_retained=float(covering.max()) if covering.size else 0.0
        column=probe_values[:,index] if probe_values.size else np.zeros(0)
        reached=[i for i in range(len(probe)) if column.size and column[i]>=match_iou]
        above=[i for i in reached if probe[i]['model_score']>=shipped_threshold]
        if best_retained>=match_iou:bucket='merged_with_neighbour'
        elif best_retained>=near_iou:bucket='localized_below_match'
        elif reached and not above:bucket='below_threshold'
        elif any(probe[i].get('disposition')=='merged' for i in above):bucket='merged_away'
        elif above:bucket='retained_unmatched'
        elif column.size and column.max()>=near_iou:bucket='localized_below_match'
        else:bucket='never_proposed'
        misses.append({'reference':index,'reference_id':reference.get('id'),'bucket':bucket,
                       'size_bin':size_bin(reference['bbox_xyxy']),
                       'best_retained_iou':round(best_retained,4),
                       'best_probe_iou':round(float(column.max()),4) if column.size else 0.0,
                       'best_probe_score':round(max((probe[i]['model_score'] for i in reached),default=0.0),4)})
    counts={bucket:sum(m['bucket']==bucket for m in misses) for bucket in MISS_BUCKETS}
    by_size={}
    for miss in misses:
        by_size.setdefault(miss['size_bin'],{b:0 for b in MISS_BUCKETS})[miss['bucket']]+=1
    return {'eligible':len(references),'matched':len(hit),
            'recall':round(len(hit)/len(references),4) if references else None,
            'missed':len(misses),'counts':counts,'by_size':by_size,'misses':misses,
            'shipped_threshold':shipped_threshold,'match_iou':match_iou,'near_iou':near_iou,
            'meaning':'miss mechanisms only; says nothing about whether unmatched boxes are false'}


def boundary_headroom(references,retained,probe,match_iou=0.5,near_iou=0.3):
    """For every reference no retained box matched, the best IoU any observation reached.

    This bounds reconciliation before a reconciliation rule exists. Every observation counts --
    any score, any view, retained, merged or rejected -- because the question is not what the
    pipeline chose but what it had available to choose from.

    `recoverable_by_selection`  some observation already clears the criterion, so the miss is a
        representative-selection or merge failure and a better choice among recorded boxes returns
        it. This is the ceiling of any selection rule.
    `needs_new_geometry`        nothing in any view clears it. No selection rule returns this,
        however clever; it needs magnification, refinement against pixels, or another route.

    The split is the whole point. Score-weighted fusion was written before it was measured and
    cost 18 of 646 matches, which is what guessing at this ceiling buys.
    """
    values=iou_matrix(retained,references) if retained and references else np.zeros((len(retained),len(references)))
    hit={b for _,b,_ in match_matrix(values,values>=match_iou)}
    probe_values=iou_matrix(probe,references) if probe and references else np.zeros((len(probe),len(references)))
    rows=[]
    for index,reference in enumerate(references):
        if index in hit:continue
        column=probe_values[:,index] if probe_values.size else np.zeros(0)
        best=float(column.max()) if column.size else 0.0
        source=int(column.argmax()) if column.size and best>0 else None
        rows.append({'reference':index,'reference_id':reference.get('id'),
                     'size_bin':size_bin(reference['bbox_xyxy']),
                     'best_observed_iou':round(best,4),
                     'verdict':'recoverable_by_selection' if best>=match_iou
                               else ('needs_new_geometry' if best>=near_iou else 'not_localized'),
                     'observation_score':round(probe[source]['model_score'],4) if source is not None else None,
                     'observation_view':probe[source].get('job') if source is not None else None,
                     'observation_disposition':probe[source].get('disposition') if source is not None else None})
    counts={}
    for row in rows:counts[row['verdict']]=counts.get(row['verdict'],0)+1
    by_size={}
    for row in rows:
        by_size.setdefault(row['size_bin'],{}).setdefault(row['verdict'],0)
        by_size[row['size_bin']][row['verdict']]+=1
    return {'eligible':len(references),'matched':len(hit),'missed':len(rows),
            'counts':counts,'by_size':by_size,'rows':rows,
            'meaning':'ceiling of reconciliation over recorded observations; not a claim any rule reaches it'}


def selection_separability(references,records,match_iou=0.5):
    """Could any evidence-only rule have picked the better box?

    `boundary_headroom` says a miss was recoverable by selection. That is a ceiling, not a rule:
    it is stated in terms of the reference, which inference cannot see. Before writing a rule that
    would reach it, check whether anything the pipeline actually has -- the detector score, the
    magnification of the view, the truncation flag -- distinguishes the clearing observation from
    the representative its cluster chose.

    A signal that fails here cannot be used, and a rule written anyway is fitted to the reference
    labels: it would encode the answer rather than derive it, which is the target-data tuning the
    constraints forbid. The honest outcome of this function is often that no rule exists, and that
    the geometry has to come from somewhere else.

    `records` is one run's whole candidate store, retained and merged alike, because the losing
    member and the representative that beat it are both needed.
    """
    retained=[r for r in records if r['disposition']=='retained']
    by_id={r['id']:r for r in records}
    values=iou_matrix(retained,references) if retained and references else np.zeros((len(retained),len(references)))
    hit={b for _,b,_ in match_matrix(values,values>=match_iou)}
    signals=('higher_score','more_magnified','untruncated_over_truncated')
    counts={name:0 for name in signals};counts['no_signal_separates']=0;rows=[]
    for index,reference in enumerate(references):
        if index in hit:continue
        clearing=[r for r in records if iou(r['bbox_xyxy'],reference['bbox_xyxy'])>=match_iou]
        if not clearing:continue
        best=max(clearing,key=lambda r:iou(r['bbox_xyxy'],reference['bbox_xyxy']))
        lead=by_id.get(best.get('merged_into')) if best['disposition']=='merged' else None
        if lead is None:
            # The clearing box was retained and still lost the reference: a one-to-one assignment
            # contention, which no representative rule addresses.
            rows.append({'reference':index,'case':'assignment_contention'});continue
        separates={'higher_score':best['model_score']>lead['model_score'],
                   'more_magnified':best.get('level',0)>lead.get('level',0),
                   'untruncated_over_truncated':not best.get('view_edge') and bool(lead.get('view_edge'))}
        for name,value in separates.items():
            if value:counts[name]+=1
        if not any(separates.values()):counts['no_signal_separates']+=1
        rows.append({'reference':index,'case':'representative_selection',
                     'chosen_iou':round(iou(lead['bbox_xyxy'],reference['bbox_xyxy']),4),
                     'clearing_iou':round(iou(best['bbox_xyxy'],reference['bbox_xyxy']),4),
                     'score_delta':round(best['model_score']-lead['model_score'],4),
                     'level_delta':best.get('level',0)-lead.get('level',0),
                     **separates})
    selection=[r for r in rows if r['case']=='representative_selection']
    return {'cases':len(selection),'contention':len(rows)-len(selection),'counts':counts,'rows':rows,
            'meaning':'a signal that separates no case cannot be used; a rule written anyway encodes the reference'}


def selectivity_curve(retained,references,scores,thresholds=None,match_iou=0.5,points=20):
    """Removal rate among matched boxes against removal rate among unmatched ones.

    A pruner is useful here only if it removes the two classes at different rates. Removing at
    random removes both at the same rate, so the gap between them is the whole signal, and it
    needs no assumption about what the unmatched boxes are.

    `scores` is any per-box score, one per retained box: the detector's own score gives the
    baseline curve that an external pruner has to beat. Thresholds default to score quantiles, so
    the sweep is determined by the data rather than by a hand-picked grid.

    The result is descriptive. A separating curve shows that a scorer discriminates; it cannot
    show that the surviving set became correct, because that is the question this reference
    cannot answer.
    """
    if len(scores)!=len(retained):raise ValueError('scores_must_be_parallel_to_retained')
    values=iou_matrix(retained,references) if retained and references else np.zeros((len(retained),len(references)))
    valid=values>=match_iou
    matched={a for a,_,_ in match_matrix(values,valid)}
    total=len(retained);matched_total=len(matched);unmatched_total=total-matched_total
    scores=np.asarray(scores,dtype=float)
    if thresholds is None:
        thresholds=sorted({float(v) for v in np.quantile(scores,np.linspace(0,1,points))}) if total else []
    rows=[]
    for threshold in thresholds:
        keep=[i for i in range(total) if scores[i]>=threshold]
        kept_matched=len(match_matrix(values[keep],valid[keep])) if keep else 0
        matched_removed=matched_total-kept_matched
        unmatched_removed=(total-len(keep))-matched_removed
        matched_rate=matched_removed/matched_total if matched_total else None
        unmatched_rate=unmatched_removed/unmatched_total if unmatched_total else None
        rows.append({'threshold':round(float(threshold),6),'kept':len(keep),'removed':total-len(keep),
                     'matched_kept':kept_matched,'matched_removed':matched_removed,
                     'unmatched_removed':unmatched_removed,
                     'matched_removed_rate':None if matched_rate is None else round(matched_rate,4),
                     'unmatched_removed_rate':None if unmatched_rate is None else round(unmatched_rate,4),
                     'selectivity':None if matched_rate is None or unmatched_rate is None else round(unmatched_rate-matched_rate,4),
                     'recall':round(kept_matched/len(references),4) if references else None,
                     'precision':round(kept_matched/len(keep),4) if keep else None})
    scored=[r for r in rows if r['selectivity'] is not None and r['removed']]
    return {'boxes':total,'matched':matched_total,'unmatched':unmatched_total,
            'references':len(references),'match_iou':match_iou,'rows':rows,
            'best':max(scored,key=lambda r:r['selectivity']) if scored else None,
            'meaning':'a random pruner scores selectivity 0; a separating curve shows discrimination, not correctness'}
