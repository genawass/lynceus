"""Each obligation of the exhaustive-panel protocol, and what happens when one is unmet.

These are the failure modes that would otherwise be invisible in the finished panel: an unvisited
tile looks like empty space, an untagged part looks like an instance, and a reference annotated
against predictions looks exactly like one annotated blind.
"""
import pytest

from lynceus.reference import build_panel, check_image, plan_review
from lynceus.policy import available_ontologies

ONTOLOGY = 'aerial-traffic-strict-v1'


def tiles_for(width, height):
    return [t['id'] for t in plan_review(width, height)]


def annotation(identifier='ref-0', box=(10, 10, 40, 40), **overrides):
    entry = {'id': identifier, 'bbox_xyxy': list(box), 'label': 'car', 'kind': 'instance',
             'scene_layer': 'physical', 'resolution': 'resolved', 'origin': 'added'}
    entry.update(overrides)
    return entry


def image(identifier='img-0', size=(600, 600), annotations=None, **overrides):
    record = {'id': identifier, 'sha256': 'a' * 64, 'scene_id': 'seq-1', 'near_duplicate_group': 'seq-1',
              'size': list(size), 'visited_tiles': tiles_for(*size), 'domain': 'aerial_drone',
              'annotations': annotations if annotations is not None else [annotation()]}
    record.update(overrides)
    return record


def test_the_review_sweep_covers_the_image_with_overlapping_tiles():
    plan = plan_review(600, 600, tile=512, overlap=0.5)
    assert len(plan) > 1
    assert all(t['bbox_xyxy'][2] <= 600 and t['bbox_xyxy'][3] <= 600 for t in plan)
    # Overlap means adjacent tiles share area, so no boundary is seen only once.
    assert plan[0]['bbox_xyxy'][2] > plan[1]['bbox_xyxy'][0]


def test_an_unvisited_tile_makes_the_image_non_exhaustive():
    record = image()
    record['visited_tiles'] = record['visited_tiles'][:-1]
    with pytest.raises(ValueError, match='incomplete_sweep'):
        check_image(record, available_ontologies()[ONTOLOGY])


def test_a_tile_that_was_never_planned_is_refused():
    record = image()
    record['visited_tiles'] = record['visited_tiles'] + ['r9c9']
    with pytest.raises(ValueError, match='unknown_review_tile'):
        check_image(record, available_ontologies()[ONTOLOGY])


def test_a_resolved_physical_instance_is_the_only_eligible_reference():
    record = image(annotations=[annotation('ref-0'),
                                annotation('ref-1', kind='part'),
                                annotation('ref-2', scene_layer='depicted'),
                                annotation('ref-3', resolution='unresolved')])
    checked = check_image(record, available_ontologies()[ONTOLOGY])
    eligible = [r for r in checked['references'] if r.get('eligible', True)]
    assert [r['id'] for r in eligible] == ['ref-0']
    # The rest are kept and counted, so their absence from a denominator is recorded, not hidden.
    assert len(checked['references']) == 4
    reasons = {r['id']: r['exclusion_reason'] for r in checked['references'] if not r.get('eligible', True)}
    assert reasons['ref-3'] == 'unresolved_by_policy'
    assert 'not_a_physical_instance' in reasons['ref-1']


def test_granularity_and_layer_tags_are_required_and_checked():
    for field, value in (('kind', 'vehicle'), ('scene_layer', 'hologram'), ('resolution', 'maybe')):
        record = image(annotations=[annotation(**{field: value})])
        with pytest.raises(ValueError, match=f'invalid_{field}'):
            check_image(record, available_ontologies()[ONTOLOGY])


def test_an_annotation_missing_its_granularity_tag_is_refused():
    entry = annotation()
    del entry['kind']
    with pytest.raises(ValueError, match='missing_annotation_field: kind'):
        check_image(image(annotations=[entry]), available_ontologies()[ONTOLOGY])


def test_a_label_outside_the_declared_ontology_is_refused():
    with pytest.raises(ValueError, match='unknown_label'):
        check_image(image(annotations=[annotation(label='spaceship')]), available_ontologies()[ONTOLOGY])


def test_a_box_outside_its_image_is_refused():
    with pytest.raises(ValueError):
        check_image(image(annotations=[annotation(box=(10, 10, 9000, 9000))]), available_ontologies()[ONTOLOGY])


def test_leakage_identity_is_required_on_every_image():
    for field in ('sha256', 'scene_id', 'near_duplicate_group'):
        record = image()
        record[field] = ''
        with pytest.raises(ValueError, match=f'missing_panel_field: {field}'):
            check_image(record, available_ontologies()[ONTOLOGY])


def test_a_panel_annotated_against_predictions_is_refused():
    with pytest.raises(ValueError, match='reference_annotated_against_predictions'):
        build_panel([image()], ONTOLOGY, domain='aerial_drone', annotator='someone', blind=False)


def test_the_same_image_content_cannot_appear_twice():
    with pytest.raises(ValueError, match='duplicate_image_content'):
        build_panel([image('img-0'), image('img-1')], ONTOLOGY, domain='aerial_drone', annotator='someone')


def test_an_exhaustive_panel_declares_no_category_scope():
    panel = build_panel([image('img-0'), image('img-1', sha256='b' * 64)],
                        ONTOLOGY, domain='aerial_drone', annotator='someone')
    assert panel['category_scope'] is None
    assert panel['exhaustive'] is True and panel['blind_annotation'] is True
    assert panel['reference_provenance'] == 'independent'
    assert panel['counts'] == {'images': 2, 'references': 2, 'eligible_references': 2}


def test_the_panel_is_accepted_by_the_evaluator_it_is_built_for():
    from lynceus.evaluation import evaluate_benchmark
    panel = build_panel([image('img-0'), image('img-1', sha256='b' * 64)],
                        ONTOLOGY, domain='aerial_drone', annotator='someone')
    result = evaluate_benchmark(panel, bootstrap_replicates=10, seed=0)
    assert result['qualification']['status'] == 'inconclusive'
    # Every reference is adjudicable, so nothing leaves a precision denominator as out of scope.
    assert result['out_of_scope_predictions'] == 0
    assert result['metrics']['localized_instance_recall']['denominator'] == 2


def test_an_empty_panel_is_refused():
    with pytest.raises(ValueError, match='empty_panel'):
        build_panel([], ONTOLOGY, domain='aerial_drone', annotator='someone')


def verification_image(identifier='img-0', **overrides):
    record = image(identifier, annotations=[
        annotation('ref-a', origin='accepted'),
        annotation('ref-b', box=(50, 50, 70, 70), origin='adjusted'),
        annotation('ref-c', box=(80, 80, 95, 95), origin='added')])
    record['rejected'] = [{'id': 'obj-9'}]
    record.update(overrides)
    return record


def test_a_verification_panel_declares_that_its_recall_means_nothing():
    panel = build_panel([verification_image()], ONTOLOGY, domain='aerial_drone', annotator='someone',
                        blind=False, anchoring='prediction_assisted', rejected=4)
    assert panel['anchoring'] == 'prediction_assisted'
    assert panel['exhaustive'] is False
    assert panel['recall_interpretable'] is False
    assert panel['blind_annotation'] is False
    assert panel['rejected_predictions'] == 4


def test_origins_are_counted_so_borrowed_geometry_is_visible():
    panel = build_panel([verification_image()], ONTOLOGY, domain='aerial_drone', annotator='someone',
                        blind=False, anchoring='prediction_assisted')
    # An accepted box is still the model's, so localization against it measures the thing under test.
    assert panel['origins'] == {'accepted': 1, 'adjusted': 1, 'added': 1}


def test_a_prediction_assisted_panel_cannot_claim_it_was_blind():
    with pytest.raises(ValueError, match='prediction_assisted_panel_cannot_claim_blind'):
        build_panel([verification_image()], ONTOLOGY, domain='aerial_drone', annotator='someone',
                    blind=True, anchoring='prediction_assisted')


def test_a_blind_panel_still_refuses_to_have_been_annotated_against_predictions():
    with pytest.raises(ValueError, match='reference_annotated_against_predictions'):
        build_panel([image()], ONTOLOGY, domain='aerial_drone', annotator='someone',
                    blind=False, anchoring='blind')


def test_an_unknown_origin_is_refused():
    record = image(annotations=[annotation(origin='invented')])
    with pytest.raises(ValueError, match='invalid_origin'):
        check_image(record, available_ontologies()[ONTOLOGY])


def test_the_evaluator_refuses_to_report_recall_from_an_anchored_panel():
    from lynceus.evaluation import evaluate_benchmark
    panel = build_panel([verification_image('img-0'), verification_image('img-1', sha256='b' * 64)],
                        ONTOLOGY, domain='aerial_drone', annotator='someone',
                        blind=False, anchoring='prediction_assisted')
    result = evaluate_benchmark(panel, bootstrap_replicates=10, seed=0)
    assert result['recall_interpretable'] is False
    recall = result['metrics']['localized_instance_recall']
    assert recall['interval_status'] == 'not_interpretable_anchored_panel'
    assert recall['descriptive_lower_95'] is None
    assert 'construction' in recall['eligibility']
    # Precision is exactly what such a panel does establish, so it stays a normal metric.
    assert result['metrics']['final_instance_precision']['interval_status'] != 'not_interpretable_anchored_panel'
    assert any('prediction-assisted' in r for r in result['qualification']['reasons'])
