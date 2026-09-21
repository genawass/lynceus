"""The unmatched-box audit: sampling, estimation, and what it refuses to conclude."""
import pytest

from lynceus.audit import (classification_accuracy, draw_sample, estimate_precision, unmatched_boxes, wilson)


def box(x, y, side=20, label='car'):
    return {'bbox_xyxy': [float(x), float(y), float(x + side), float(y + side)], 'label': label}


def test_only_unmatched_boxes_are_undecided():
    retained = [box(0, 0), box(100, 100), box(500, 500)]
    references = [box(0, 0), box(100, 100)]
    unmatched, matched = unmatched_boxes(retained, references)
    # A box matching an eligible reference is a true positive and needs no judgement.
    assert matched == 2
    assert unmatched == [2]


def test_the_sample_covers_every_size_and_class_stratum_present():
    population = ([box(i * 30, 0, side=8, label='car') for i in range(40)]
                  + [box(i * 30, 100, side=200, label='bus') for i in range(4)]
                  + [box(i * 30, 400, side=8, label='tricycle') for i in range(2)])
    sample = draw_sample(population, 12, seed=0)
    strata = {(s['label'], s['bbox_xyxy'][2] - s['bbox_xyxy'][0]) for s in sample}
    # A rare class that the draw happened to miss would be invisible in the result.
    assert {'car', 'bus', 'tricycle'} <= {s['label'] for s in sample}
    assert len(strata) >= 3


def test_the_draw_is_reproducible_under_its_seed():
    population = [box(i * 30, 0, label='car' if i % 2 else 'van') for i in range(50)]
    first = draw_sample(population, 10, seed=7)
    assert first == draw_sample(population, 10, seed=7)
    assert first != draw_sample(population, 10, seed=8)


def test_a_population_smaller_than_the_sample_is_taken_whole():
    population = [box(0, 0), box(50, 50)]
    assert draw_sample(population, 10, seed=0) == population


def test_an_invalid_sample_size_is_refused():
    with pytest.raises(ValueError, match='invalid_sample_size'):
        draw_sample([box(0, 0)], 0)


def test_the_uncorrected_figure_is_the_case_where_nothing_unmatched_is_real():
    report = estimate_precision(matched=100, retained=1000, verdicts=['false'] * 50)
    assert report['uncorrected_precision'] == 0.1
    assert report['real_rate_among_unmatched'] == 0.0
    assert report['corrected_precision'] == 0.1
    # Zero successes must not produce a zero-width interval: that is where the audit knows least.
    assert report['corrected_precision_interval'][1] > 0.1


def test_real_unmatched_boxes_raise_precision_above_the_bound():
    report = estimate_precision(matched=100, retained=1000, verdicts=['real'] * 45 + ['false'] * 45)
    assert report['uncorrected_precision'] == 0.1
    assert report['corrected_precision'] > 0.5
    low, high = report['corrected_precision_interval']
    assert low < report['corrected_precision'] < high


def test_undecidable_boxes_are_evidence_for_neither_side():
    decided = estimate_precision(matched=10, retained=100, verdicts=['real'] * 10 + ['false'] * 10)
    padded = estimate_precision(matched=10, retained=100,
                                verdicts=['real'] * 10 + ['false'] * 10 + ['undecidable'] * 40)
    # Adding boxes nobody could resolve must not move the estimate.
    assert padded['corrected_precision'] == decided['corrected_precision']
    assert padded['decided'] == decided['decided'] == 20
    assert padded['undecidable_rate'] == 0.6667


def test_an_audit_that_decided_nothing_estimates_nothing():
    report = estimate_precision(matched=10, retained=100, verdicts=['undecidable'] * 20)
    assert report['corrected_precision'] is None
    assert report['corrected_precision_interval'] is None
    assert report['undecidable_rate'] == 1.0


def test_wilson_stays_inside_the_unit_interval():
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0
    assert 0.0 <= wilson(3, 7)[0] < wilson(3, 7)[1] <= 1.0


def test_classification_is_scored_only_on_boxes_judged_real():
    judgements = [{'verdict': 'real', 'label': 'car', 'true_label': 'car'},
                  {'verdict': 'real', 'label': 'car', 'true_label': 'van'},
                  {'verdict': 'false', 'label': 'bus', 'true_label': 'bus'},
                  {'verdict': 'undecidable', 'label': 'car'}]
    report = classification_accuracy(judgements)
    assert report['scored'] == 2 and report['correct'] == 1
    assert report['accuracy'] == 0.5
    assert report['confusions'] == [('car', 'van')]


def test_classification_reports_nothing_when_nothing_was_judged_real():
    assert classification_accuracy([{'verdict': 'false', 'label': 'car'}])['accuracy'] is None
