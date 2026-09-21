"""Auditing unmatched boxes, which is where a finite-category reference leaves precision undecided.

A box matching an eligible reference is a true positive and needs no judgement. A box matching
nothing is either a false positive or a real object the reference never annotated, and those two
are indistinguishable from the data. So the open question is confined to unmatched boxes, and
answering it needs a sample of them rather than an annotated panel.

Judging a sample estimates the proportion that are real, and that proportion turns the precision
bound into an interval. The uncorrected number is the case where that proportion is zero, which is
exactly why it is a bound.

Nothing here establishes recall. It resolves what the retained boxes are, not what is missing.
"""
import numpy as np

from .diagnostics import size_bin
from .evaluation import iou_matrix, match_matrix

VERDICTS=('real','false','undecidable')


def unmatched_boxes(retained,references,match_iou=0.5):
    """The retained boxes no eligible reference claims, which are the only undecided ones."""
    values=iou_matrix(retained,references) if retained and references else np.zeros((len(retained),len(references)))
    matched={a for a,_,_ in match_matrix(values,values>=match_iou)}
    return [i for i in range(len(retained)) if i not in matched],len(matched)


def draw_sample(population,size,seed=0):
    """A stratified draw over size bin and emitted class, under a frozen seed.

    Stratifying on both is not decoration: a rare class or a small-object regime that the draw
    happened to miss would be invisible in the result, and both are exactly where the answer is
    likely to differ. Each stratum contributes in proportion to its share, with at least one box
    wherever a stratum is non-empty, so nothing present goes unlooked-at.

    The draw is returned in full so it can be recorded, re-judged or extended rather than redrawn.
    """
    if size<=0:raise ValueError('invalid_sample_size')
    strata={}
    for entry in population:
        key=(size_bin(entry['bbox_xyxy']),entry.get('label','unlabelled'))
        strata.setdefault(key,[]).append(entry)
    total=len(population)
    if total<=size:return list(population)
    rng=np.random.default_rng(seed)
    chosen=[]
    for key in sorted(strata):
        group=strata[key]
        take=max(1,round(size*len(group)/total))
        take=min(take,len(group))
        index=rng.choice(len(group),take,replace=False)
        chosen.extend(group[int(i)] for i in sorted(index))
    # Proportional rounding overshoots or undershoots; trim or top up deterministically.
    if len(chosen)>size:
        keep=rng.choice(len(chosen),size,replace=False)
        chosen=[chosen[int(i)] for i in sorted(keep)]
    return chosen


def wilson(successes,trials,z=1.96):
    """Wilson interval, which stays inside [0,1] and does not collapse at zero successes.

    A normal approximation would give a zero-width interval when nothing in the sample was real,
    and that is precisely the case where the audit has the least to say.
    """
    if trials<=0:return (0.0,1.0)
    p=successes/trials
    denominator=1+z*z/trials
    centre=(p+z*z/(2*trials))/denominator
    spread=z*((p*(1-p)/trials+z*z/(4*trials*trials))**0.5)/denominator
    return (max(0.0,centre-spread),min(1.0,centre+spread))


def estimate_precision(matched,retained,verdicts,z=1.96):
    """Corrected precision and its interval, from the judged sample of unmatched boxes.

    Undecidable boxes leave both the numerator and the denominator of `p_real`. They are not
    evidence either way, and forcing them into one would buy a tidy number with an invented
    judgement. Their rate is reported alongside, because an audit that could not resolve much of
    its sample has not measured precision however narrow the interval looks.
    """
    counts={v:sum(1 for x in verdicts if x==v) for v in VERDICTS}
    unmatched=retained-matched
    decided=counts['real']+counts['false']
    point,low,high=(None,None,None)
    if decided:
        p=counts['real']/decided
        lo,hi=wilson(counts['real'],decided,z)
        point=(matched+p*unmatched)/retained if retained else None
        low=(matched+lo*unmatched)/retained if retained else None
        high=(matched+hi*unmatched)/retained if retained else None
    return {'retained':retained,'matched':matched,'unmatched':unmatched,
            'sampled':len(verdicts),'verdicts':counts,'decided':decided,
            'undecidable_rate':round(counts['undecidable']/len(verdicts),4) if verdicts else None,
            'uncorrected_precision':round(matched/retained,4) if retained else None,
            'real_rate_among_unmatched':round(counts['real']/decided,4) if decided else None,
            'corrected_precision':None if point is None else round(point,4),
            'corrected_precision_interval':None if point is None else [round(low,4),round(high,4)],
            'meaning':('the uncorrected figure is the case where no unmatched box is real, which is why it '
                       'is a bound; undecidable boxes are evidence for neither side and leave both counts')}


def classification_accuracy(judgements):
    """Class correctness on boxes judged real, which the reference cannot adjudicate at all.

    Class correctness on matched boxes is already known from permitted labels. On an object the
    reference never annotated it is known from nothing, so it is only measurable here.
    """
    scored=[j for j in judgements if j.get('verdict')=='real' and j.get('true_label')]
    if not scored:return {'scored':0,'correct':0,'accuracy':None}
    correct=sum(1 for j in scored if j['true_label']==j.get('label'))
    return {'scored':len(scored),'correct':correct,'accuracy':round(correct/len(scored),4),
            'confusions':sorted({(j.get('label'),j['true_label']) for j in scored if j['true_label']!=j.get('label')})}
