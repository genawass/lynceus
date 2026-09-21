"""Reading an automatic verifier's agreement, and bounding what it can support.

A second model cannot establish that a box is correct. It agrees or it does not, and agreement with
an automatic reference is not truth -- the contract is explicit about that, and shared ancestry
between models makes it worse rather than better.

What rescues this from being a bare vote is that half the population is already labelled. A box
matching an eligible reference is a known true positive, so the verifier's confirmation rate on
those boxes measures its sensitivity directly: on this data, at this object scale, against this
detector's geometry. That number is not assumed, it is observed.

The asymmetry is the whole point and it has to survive into the report. Sensitivity is measurable
because known positives exist. Specificity is not, because nothing here is a known negative -- that
absence is precisely the problem being worked around. So a low confirmation rate among undecided
boxes has two readings, that they are false or that the verifier misses them too, and only the
first is a finding. What can be stated is a bound, and the bound is one-sided.
"""

VERIFIER_READINGS=('confirmed','unconfirmed')


def _rate(hits,total):
    return round(hits/total,4) if total else None


def summarise(rows):
    """Confirmation rates split by what the reference already settled.

    `known` is `true_positive` where a reference matched the box and `undecided` where none did.
    Those are different populations and are never pooled: pooling would let the labelled half carry
    a conclusion about the unlabelled half without saying so.
    """
    known=[r for r in rows if r['known']=='true_positive']
    undecided=[r for r in rows if r['known']=='undecided']
    sensitivity=_rate(sum(1 for r in known if r['confirmed']),len(known))
    undecided_rate=_rate(sum(1 for r in undecided if r['confirmed']),len(undecided))
    report={'retained':len(rows),'known_true_positives':len(known),'undecided':len(undecided),
            'sensitivity_on_known_positives':sensitivity,
            'confirmation_rate_on_undecided':undecided_rate,
            'uncorrected_precision':_rate(len(known),len(rows))}
    # Every confirmed undecided box is a box two independently prompted models place together. That
    # is not proof it is real, but a verifier with measured sensitivity s cannot confirm more than
    # s of whatever is real, so the confirmed count divided by s estimates how many real boxes are
    # there -- an estimate that is only as good as the assumption that the verifier behaves the
    # same on both populations, which is exactly what cannot be checked.
    if sensitivity and undecided:
        confirmed=sum(1 for r in undecided if r['confirmed'])
        implied=min(len(undecided),confirmed/sensitivity)
        report['implied_real_among_undecided']=round(implied)
        report['precision_if_confirmed_are_real']=_rate(len(known)+confirmed,len(rows))
        report['precision_under_equal_sensitivity']=_rate(len(known)+implied,len(rows))
    report['interpretation']=(
        'sensitivity is measured against known positives; the confirmation rate on undecided boxes '
        'is not a precision measurement, because a box the verifier misses and a box that is not '
        'there look identical. The corrected figures assume the verifier behaves the same on both '
        'populations, which nothing here establishes.')
    return report


def by_stratum(rows,key):
    """The same split per size bin or per class, where the assumption is most likely to fail.

    A verifier whose sensitivity collapses on small objects would produce a low confirmation rate
    among undecided boxes that says nothing about those boxes. Reporting per stratum is what makes
    that visible instead of averaging it away.
    """
    groups={}
    for row in rows:groups.setdefault(row[key],[]).append(row)
    out={}
    for name,group in sorted(groups.items()):
        known=[r for r in group if r['known']=='true_positive']
        undecided=[r for r in group if r['known']=='undecided']
        out[name]={'known_positives':len(known),'undecided':len(undecided),
                   'sensitivity':_rate(sum(1 for r in known if r['confirmed']),len(known)),
                   'confirmation_on_undecided':_rate(sum(1 for r in undecided if r['confirmed']),len(undecided))}
    return out


def class_agreement(rows,synonyms=None,ontology=None):
    """Where the two models agree a box exists, do they agree what it is?

    Exact string comparison is the wrong test against an ontology with a hierarchy. Emitting
    `vehicle` where the verifier says `car` is not a contradiction: `vehicle` is the broader class
    and is a permitted label for a car, so the two are compatible and the system was merely less
    specific. Counting that as disagreement would report a naming policy working as designed as if
    it were an error.

    So three outcomes are separated. Exact agreement, compatible but less specific -- the emitted
    label is an ancestor of the verifier's -- and conflict, where neither is an ancestor of the
    other and the two models genuinely disagree about what the object is. Only the third is a
    finding.

    Only boxes both models place are eligible, and neither label is a reference, so this is
    agreement between two emitted labels rather than accuracy against truth. Disagreement localises
    naming problems; it does not adjudicate them.
    """
    synonyms=synonyms or {}
    paired=[r for r in rows if r['confirmed'] and r.get('verifier_label')]
    if not paired:return {'paired':0,'agreement':None}
    def norm(value):return synonyms.get(value,value)

    def ancestors(label):
        seen=[]
        if ontology is None:return seen
        index={c['id']:c for c in ontology['classes']}
        node=index.get(label)
        while node is not None and node.get('parent'):
            seen.append(node['parent']);node=index.get(node['parent'])
        return seen

    exact=broader=conflict=0
    conflicts={};generic={}
    for r in paired:
        emitted,verifier=norm(r['label']),norm(r['verifier_label'])
        if emitted==verifier:exact+=1
        elif emitted in ancestors(verifier):
            broader+=1;generic[(emitted,verifier)]=generic.get((emitted,verifier),0)+1
        else:
            conflict+=1;conflicts[(emitted,verifier)]=conflicts.get((emitted,verifier),0)+1
    return {'paired':len(paired),'exact':exact,'compatible_less_specific':broader,'conflict':conflict,
            'agreement':_rate(exact,len(paired)),
            'compatible_rate':_rate(exact+broader,len(paired)),
            'conflict_rate':_rate(conflict,len(paired)),
            'top_conflicts':sorted(conflicts.items(),key=lambda kv:-kv[1])[:8],
            'top_under_specified':sorted(generic.items(),key=lambda kv:-kv[1])[:8],
            'meaning':('agreement between two emitted labels, not accuracy against a reference; '
                       'a broader emitted label is compatible with a specific one, not a disagreement')}
