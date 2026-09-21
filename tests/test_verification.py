"""What an automatic verifier's agreement supports, and what it must refuse to claim."""
from lynceus.verification import by_stratum, class_agreement, summarise


def row(known, confirmed, label='car', verifier=None, size='16-32'):
    return {'known': known, 'confirmed': confirmed, 'label': label,
            'verifier_label': verifier, 'size_bin': size, 'image': 'i', 'id': 'o'}


def test_sensitivity_is_measured_against_boxes_the_reference_already_settled():
    rows = [row('true_positive', True)] * 9 + [row('true_positive', False)]
    report = summarise(rows)
    # Known positives make sensitivity an observation rather than an assumption.
    assert report['sensitivity_on_known_positives'] == 0.9
    assert report['known_true_positives'] == 10


def test_the_two_populations_are_never_pooled():
    rows = [row('true_positive', True)] * 10 + [row('undecided', False)] * 10
    report = summarise(rows)
    assert report['sensitivity_on_known_positives'] == 1.0
    assert report['confirmation_rate_on_undecided'] == 0.0
    assert report['uncorrected_precision'] == 0.5


def test_a_confirmation_rate_on_undecided_boxes_is_not_called_precision():
    rows = [row('true_positive', True)] * 10 + [row('undecided', True)] * 10
    report = summarise(rows)
    assert 'not a precision measurement' in report['interpretation']
    # The corrected figures exist but are labelled as resting on an uncheckable assumption.
    assert 'assume the verifier behaves the same' in report['interpretation']


def test_imperfect_sensitivity_implies_more_real_boxes_than_were_confirmed():
    rows = ([row('true_positive', True)] * 8 + [row('true_positive', False)] * 2
            + [row('undecided', True)] * 20 + [row('undecided', False)] * 80)
    report = summarise(rows)
    assert report['sensitivity_on_known_positives'] == 0.8
    # 20 confirmed at sensitivity 0.8 implies about 25 real among the undecided.
    assert report['implied_real_among_undecided'] == 25
    assert report['precision_if_confirmed_are_real'] == 0.2727
    assert report['precision_under_equal_sensitivity'] > report['precision_if_confirmed_are_real']


def test_the_implied_count_cannot_exceed_the_population():
    rows = [row('true_positive', True), row('true_positive', False),
            row('undecided', True), row('undecided', True)]
    report = summarise(rows)
    assert report['implied_real_among_undecided'] <= report['undecided']


def test_strata_expose_a_verifier_that_fails_where_the_answer_matters():
    rows = ([row('true_positive', True, size='>=96')] * 10
            + [row('true_positive', False, size='<16')] * 10
            + [row('undecided', False, size='<16')] * 10)
    strata = by_stratum(rows, 'size_bin')
    # Sensitivity collapses on small objects, so their low confirmation rate says nothing.
    assert strata['>=96']['sensitivity'] == 1.0
    assert strata['<16']['sensitivity'] == 0.0
    assert strata['<16']['confirmation_on_undecided'] == 0.0


def test_class_agreement_is_between_two_emitted_labels():
    rows = [row('true_positive', True, label='car', verifier='car'),
            row('true_positive', True, label='car', verifier='van'),
            row('undecided', False, label='bus')]
    report = class_agreement(rows)
    assert report['paired'] == 2 and report['agreement'] == 0.5
    assert report['top_conflicts'][0][0] == ('car', 'van')
    assert 'not accuracy against a reference' in report['meaning']


def test_a_broader_emitted_label_is_compatible_not_a_disagreement():
    """`vehicle` where the verifier says `car` is the naming policy working, not an error.

    Counting it as disagreement would report conservative naming as a fault and bury the genuine
    conflicts underneath it.
    """
    from lynceus.policy import load_ontology
    ontology = load_ontology('aerial-traffic-strict-v1')
    rows = [row('true_positive', True, label='vehicle', verifier='car'),
            row('true_positive', True, label='vehicle', verifier='motorcycle'),
            row('true_positive', True, label='bicycle', verifier='person')]
    report = class_agreement(rows, None, ontology)
    assert report['exact'] == 0
    assert report['compatible_less_specific'] == 2
    assert report['conflict'] == 1
    assert report['compatible_rate'] == 0.6667
    assert report['top_conflicts'][0][0] == ('bicycle', 'person')
    assert report['top_under_specified'][0][1] == 1


def test_without_an_ontology_only_exact_matches_are_compatible():
    rows = [row('true_positive', True, label='vehicle', verifier='car')]
    report = class_agreement(rows)
    # No hierarchy available, so nothing can be shown to be broader.
    assert report['compatible_less_specific'] == 0 and report['conflict'] == 1


def test_synonyms_reconcile_vocabularies_before_disagreement_is_counted():
    rows = [row('true_positive', True, label='motorcycle', verifier='motorbike')]
    assert class_agreement(rows)['agreement'] == 0.0
    assert class_agreement(rows, {'motorbike': 'motorcycle'})['agreement'] == 1.0


def test_nothing_is_claimed_when_nothing_was_confirmed():
    assert class_agreement([row('undecided', False)])['agreement'] is None
