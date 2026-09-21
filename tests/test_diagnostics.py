"""Miss attribution and pruner selectivity on fixtures with known answers."""
import pytest

from lynceus.diagnostics import (attribute_misses, boundary_headroom, selection_separability,
                                 selectivity_curve, size_bin)


def ref(box, identifier='ref-0'):
    return {'id': identifier, 'bbox_xyxy': [float(v) for v in box]}


def obs(box, score, disposition='retained'):
    return {'bbox_xyxy': [float(v) for v in box], 'model_score': score, 'disposition': disposition}


def bucket_of(references, retained, probe, threshold=0.1):
    report = attribute_misses(references, retained, probe, threshold)
    assert report['missed'] == 1
    return report['misses'][0]['bucket']


def test_never_proposed_requires_nothing_reaching_the_reference():
    assert bucket_of([ref([0, 0, 10, 10])], [], [obs([500, 500, 520, 520], 0.9)]) == 'never_proposed'


def test_below_threshold_when_every_proposal_scores_under_the_shipped_threshold():
    assert bucket_of([ref([0, 0, 10, 10])], [], [obs([0, 0, 10, 10], 0.02)]) == 'below_threshold'


def test_merged_away_when_a_scoring_proposal_was_absorbed_elsewhere():
    probe = [obs([0, 0, 10, 10], 0.6, 'merged')]
    assert bucket_of([ref([0, 0, 10, 10])], [], probe) == 'merged_away'


def test_localized_below_match_is_a_boundary_failure_not_a_discovery_failure():
    # IoU 40/100 = 0.40: past the near threshold, short of the match criterion.
    assert bucket_of([ref([0, 0, 10, 10])], [ref([0, 0, 10, 4], 'obj-0')], []) == 'localized_below_match'


def test_merged_with_neighbour_when_two_references_share_one_box():
    references = [ref([0, 0, 10, 10], 'ref-0'), ref([0, 0, 10, 12], 'ref-1')]
    report = attribute_misses(references, [ref([0, 0, 10, 11], 'obj-0')], [], 0.1)
    # One box cannot satisfy both, so exactly one reference is matched and the other is attributed
    # to the merge rather than to discovery.
    assert report['matched'] == 1 and report['missed'] == 1
    assert report['misses'][0]['bucket'] == 'merged_with_neighbour'


def test_a_matched_reference_is_not_a_miss():
    report = attribute_misses([ref([0, 0, 10, 10])], [ref([0, 0, 10, 10], 'obj-0')], [], 0.1)
    assert report['missed'] == 0 and report['recall'] == 1.0
    assert report['counts'] == {bucket: 0 for bucket in report['counts']}


def test_sub_threshold_evidence_does_not_outrank_a_retained_box_that_missed():
    # A retained box already reached the reference loosely; the probe's weak hit must not relabel
    # this as a ranking failure, because raising the threshold would not produce a tighter box.
    references = [ref([0, 0, 10, 10])]
    retained = [ref([0, 0, 10, 4], 'obj-0')]
    assert bucket_of(references, retained, [obs([0, 0, 10, 10], 0.01)]) == 'localized_below_match'


def test_probe_is_required_because_absent_evidence_is_not_absent_proposals():
    with pytest.raises(ValueError, match='probe_candidates_required'):
        attribute_misses([ref([0, 0, 10, 10])], [], None, 0.1)


def test_size_bins_follow_the_frozen_edges():
    assert size_bin([0, 0, 10, 10]) == '<16'
    assert size_bin([0, 0, 20, 20]) == '16-32'
    assert size_bin([0, 0, 50, 50]) == '32-96'
    assert size_bin([0, 0, 200, 200]) == '>=96'


def build_mixed_set():
    references = [ref([0, 0, 10, 10], 'ref-0'), ref([100, 100, 110, 110], 'ref-1')]
    retained = [ref([0, 0, 10, 10], 'obj-0'), ref([100, 100, 110, 110], 'obj-1'),
                ref([500, 500, 510, 510], 'obj-2'), ref([600, 600, 610, 610], 'obj-3')]
    return references, retained


def test_a_perfect_pruner_separates_completely():
    references, retained = build_mixed_set()
    perfect = [1.0, 1.0, 0.0, 0.0]
    curve = selectivity_curve(retained, references, perfect)
    assert curve['matched'] == 2 and curve['unmatched'] == 2
    best = curve['best']
    assert best['matched_removed_rate'] == 0.0
    assert best['unmatched_removed_rate'] == 1.0
    assert best['selectivity'] == 1.0
    assert best['recall'] == 1.0


def test_a_constant_scorer_cannot_discriminate():
    references, retained = build_mixed_set()
    curve = selectivity_curve(retained, references, [0.5] * 4)
    # Every threshold keeps everything or removes everything, so no operating point separates.
    assert all(row['selectivity'] in (0.0, None) for row in curve['rows'])


def test_an_inverted_pruner_reports_negative_selectivity():
    references, retained = build_mixed_set()
    curve = selectivity_curve(retained, references, [0.0, 0.0, 1.0, 1.0])
    assert min(row['selectivity'] for row in curve['rows'] if row['selectivity'] is not None) == -1.0


def test_removing_boxes_can_only_lose_matches():
    references, retained = build_mixed_set()
    curve = selectivity_curve(retained, references, [0.9, 0.2, 0.8, 0.1])
    kept = [row['matched_kept'] for row in curve['rows']]
    assert kept == sorted(kept, reverse=True)


def test_scores_must_be_parallel_to_the_retained_set():
    references, retained = build_mixed_set()
    with pytest.raises(ValueError, match='scores_must_be_parallel_to_retained'):
        selectivity_curve(retained, references, [0.1])


def test_headroom_calls_a_miss_recoverable_when_some_observation_already_clears_it():
    references = [ref([0, 0, 10, 10])]
    retained = [ref([0, 0, 10, 4], 'obj-0')]
    report = boundary_headroom(references, retained, [obs([0, 0, 10, 10], 0.3)])
    assert report['counts'] == {'recoverable_by_selection': 1}
    assert report['rows'][0]['best_observed_iou'] == 1.0


def test_headroom_calls_a_miss_new_geometry_when_nothing_clears_it():
    references = [ref([0, 0, 10, 10])]
    retained = [ref([0, 0, 10, 4], 'obj-0')]
    # Every observation is the same loose box, so no choice among them returns the reference.
    report = boundary_headroom(references, retained, [obs([0, 0, 10, 4], 0.3), obs([0, 0, 10, 4], 0.9)])
    assert report['counts'] == {'needs_new_geometry': 1}


def test_headroom_ignores_references_that_were_matched():
    references = [ref([0, 0, 10, 10])]
    report = boundary_headroom(references, [ref([0, 0, 10, 10], 'obj-0')], [])
    assert report['missed'] == 0 and report['counts'] == {}


def test_headroom_counts_observations_of_every_disposition():
    references = [ref([0, 0, 10, 10])]
    report = boundary_headroom(references, [], [obs([0, 0, 10, 10], 0.9, 'merged')])
    assert report['rows'][0]['observation_disposition'] == 'merged'
    assert report['counts'] == {'recoverable_by_selection': 1}


def record(identifier, box, score, disposition='retained', level=0, view_edge=False, merged_into=None):
    return {'id': identifier, 'bbox_xyxy': [float(v) for v in box], 'model_score': score, 'level': level,
            'disposition': disposition, 'view_edge': view_edge, 'merged_into': merged_into}


def test_separability_reports_a_signal_that_would_have_picked_the_better_box():
    references = [ref([0, 0, 10, 10])]
    records = [record('a', [0, 0, 10, 4], 0.4, level=0, view_edge=True),
               record('b', [0, 0, 10, 10], 0.9, 'merged', level=1, merged_into='a')]
    report = selection_separability(references, records)
    assert report['cases'] == 1
    assert report['counts']['higher_score'] == 1
    assert report['counts']['more_magnified'] == 1
    assert report['counts']['untruncated_over_truncated'] == 1
    assert report['counts']['no_signal_separates'] == 0


def test_separability_reports_when_nothing_distinguishes_the_better_box():
    references = [ref([0, 0, 10, 10])]
    # The clearing member scores lower, is no more magnified, and neither box is truncated: the
    # pipeline had nothing to go on.
    records = [record('a', [0, 0, 10, 4], 0.9, level=1),
               record('b', [0, 0, 10, 10], 0.2, 'merged', level=1, merged_into='a')]
    report = selection_separability(references, records)
    assert report['counts']['no_signal_separates'] == 1
    assert report['rows'][0]['score_delta'] < 0


def test_separability_separates_assignment_contention_from_selection():
    references = [ref([0, 0, 10, 10], 'ref-0'), ref([0, 0, 10, 12], 'ref-1')]
    records = [record('a', [0, 0, 10, 11], 0.9)]
    report = selection_separability(references, records)
    assert report['cases'] == 0 and report['contention'] == 1
