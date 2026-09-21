"""Acceptance rule (WP-09): uncertain -> accepted only where a check ran and passed."""
import json

from lynceus.pipeline import annotate, assess, ACCEPTANCE_DEFAULTS


def one(label='cup', score=.95, box=(5, 5, 40, 40), edge=False):
    class Adapter:
        def predict(self, view):
            return [{'bbox_xyxy': list(box), 'label': label, 'model_score': score, 'crop_edge': edge}]
    return Adapter()


def run(staged, adapter, **kw):
    image, bundle, output = staged
    return annotate(image, bundle, output, adapter=adapter, acceptance=kw.pop('acceptance', {}), **kw)


def test_accepts_a_supported_detection(staged):
    obj = run(staged, one())['objects'][0]
    assert obj['status'] == 'accepted'
    assert all(obj['uncertainty'][d] == 'supported'
               for d in ('existence', 'count', 'boundary', 'class', 'granularity'))
    assert 'single_view_schedule' in obj['uncertainty']['reasons']


def test_weak_score_blocks_existence(staged):
    obj = run(staged, one(score=.2))['objects'][0]
    assert obj['status'] == 'uncertain'
    assert obj['uncertainty']['existence'] == 'unresolved'
    assert 'weak_score' in obj['uncertainty']['reasons']


def test_unknown_class_never_accepted_but_keeps_geometry(staged):
    obj = run(staged, one(label='unknown_object'))['objects'][0]
    assert obj['status'] == 'uncertain'
    assert obj['uncertainty']['class'] == 'unresolved'
    assert 'no_useful_class' in obj['uncertainty']['reasons']
    assert obj['bbox_xyxy'] == [5., 5., 40., 40.]


def test_view_edge_truncation_blocks_boundary(staged):
    obj = run(staged, one(edge=True))['objects'][0]
    assert obj['uncertainty']['boundary'] == 'unresolved'
    assert obj['status'] == 'uncertain'


def test_containment_withholds_granularity(staged):
    """A smaller box inside a larger one may be a part, so its granularity is not supported."""
    class Nested:
        def predict(self, view):
            return [{'bbox_xyxy': [2, 2, 95, 55], 'label': 'person', 'model_score': .95, 'crop_edge': False},
                    {'bbox_xyxy': [10, 10, 30, 30], 'label': 'bag', 'model_score': .9, 'crop_edge': False}]
    objects = run(staged, Nested())['objects']
    inner = next(o for o in objects if o['label']['id'] == 'bag')
    outer = next(o for o in objects if o['label']['id'] == 'person')
    assert inner['uncertainty']['granularity'] == 'unresolved'
    assert any(r.startswith('possible_part_of:') for r in inner['uncertainty']['reasons'])
    assert inner['status'] == 'uncertain'
    assert outer['uncertainty']['granularity'] == 'supported' and outer['status'] == 'accepted'


def test_default_run_accepts_nothing(staged):
    image, bundle, output = staged
    result = annotate(image, bundle, output, adapter=one())
    assert {o['status'] for o in result['objects']} == {'uncertain'}
    assert result['objects'][0]['uncertainty']['existence'] == 'not_assessed'


def test_rule_is_recorded_in_the_manifest(staged):
    image, bundle, output = staged
    annotate(image, bundle, output, adapter=one(), acceptance={'score': .7})
    recorded = json.loads((output / 'manifest.json').read_text())['config']['acceptance']
    assert recorded['score'] == .7
    assert recorded['min_views'] == ACCEPTANCE_DEFAULTS['min_views']


def test_view_agreement_is_capped_by_the_schedule():
    """A single-view schedule cannot be asked for agreement it never planned to collect."""
    objects = [{'id': 'obj-0', 'kind': 'instance', 'bbox_xyxy': [0, 0, 10, 10],
                'label': {'id': 'cup'}, 'status': 'uncertain'}]
    evidence = [{'score': .9, 'views': 1, 'view_edge': False, 'alternatives': [], 'useful': True}]
    rules = {**ACCEPTANCE_DEFAULTS, 'min_views': 3}
    assert assess([dict(objects[0])], evidence, 1, rules)[0]['status'] == 'accepted'
    # but when three views were planned and only one saw it, agreement is genuinely missing
    blocked = assess([dict(objects[0])], evidence, 3, rules)[0]
    assert blocked['status'] == 'uncertain'
    assert 'insufficient_view_agreement' in blocked['uncertainty']['reasons']


def test_no_probability_field_is_ever_emitted(staged):
    result = run(staged, one())
    text = json.dumps(result)
    for forbidden in ('probability', 'confidence', 'is_ground_truth', 'probability_correct'):
        assert forbidden not in text
