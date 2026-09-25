"""Independent confirmation as a pipeline signal: evidence, never a deletion."""
import json

import pytest

from lynceus.pipeline import annotate, verify_objects


class Verifier:
    """Confirms boxes whose left edge is under 50, so a run has both outcomes."""

    def __init__(self, label='cup', score=.8):
        self.label, self.score, self.calls = label, score, 0

    def capabilities(self):
        return {'device': 'cpu', 'prompt': 'text', 'independence': 'test double'}

    def detect(self, image, prompts):
        self.calls += 1
        self.prompts = dict(prompts)
        return [{'bbox_xyxy': [0, 0, 45, 45], 'label': self.label, 'score': self.score}]


class TwoBoxes:
    def predict(self, view):
        return [{'bbox_xyxy': [2, 2, 44, 44], 'label': 'cup', 'model_score': .9, 'crop_edge': False},
                {'bbox_xyxy': [60, 5, 95, 40], 'label': 'cup', 'model_score': .9, 'crop_edge': False}]


def run(staged, **kw):
    image, bundle, output = staged
    return annotate(image, bundle, output, adapter=TwoBoxes(), verifier=Verifier(), **kw)


def test_confirmation_is_recorded_per_object_and_nothing_is_removed(staged):
    result = run(staged)
    assert len(result['objects']) == 2, 'verification must not delete a box'
    _, _, output = staged
    records = [json.loads(l) for l in (output / 'evidence/candidates.jsonl').read_text().splitlines()]
    verified = [r['verification'] for r in records if r.get('verification')]
    assert len(verified) == 2
    assert sorted(v['confirmed'] for v in verified) == [False, True]
    confirmed = next(v for v in verified if v['confirmed'])
    assert confirmed['verifier_label'] == 'cup' and confirmed['overlap'] >= 0.5


def test_unconfirmed_box_says_so_in_its_reasons(staged):
    result = run(staged)
    unconfirmed = [o for o in result['objects']
                   if 'unconfirmed_by_independent_model' in o['uncertainty']['reasons']]
    assert len(unconfirmed) == 1


def test_verification_summary_reaches_the_events_log(staged):
    _, _, output = staged
    run(staged)
    events = [json.loads(l) for l in (output / 'events.jsonl').read_text().splitlines()]
    stage = next(e for e in events if e['stage'] == 'verification')
    assert stage['summary']['objects'] == 2 and stage['summary']['confirmed'] == 1
    # the limit of what agreement can support travels with the number
    assert 'reproducibility' in stage['summary']['independence']


def test_acceptance_can_require_confirmation(staged):
    """Requiring it withholds support; it never marks the box wrong."""
    lenient = run(staged, acceptance={'score': .5, 'min_views': 1})
    strict = annotate(*_fresh(staged), adapter=TwoBoxes(), verifier=Verifier(),
                      acceptance={'score': .5, 'min_views': 1, 'require_verification': True})
    assert sum(o['status'] == 'accepted' for o in lenient['objects']) == 2
    accepted = [o for o in strict['objects'] if o['status'] == 'accepted']
    assert len(accepted) == 1
    assert 'confirmed_by_independent_model' in accepted[0]['uncertainty']['reasons']
    withheld = next(o for o in strict['objects'] if o['status'] != 'accepted')
    assert withheld['uncertainty']['existence'] == 'unresolved'
    assert withheld['bbox_xyxy'], 'geometry survives an unconfirmed box'


def test_verifier_is_asked_with_the_class_ids_not_the_phrasing(staged):
    """Agreement must not depend on how the question happened to be worded."""
    image, bundle, output = staged
    verifier = Verifier()
    annotate(image, bundle, output, adapter=TwoBoxes(), verifier=verifier)
    assert verifier.calls == 1, 'one detection pass serves every box'
    assert set(verifier.prompts.values()) <= {c['id'] for c in
                                              __import__('lynceus.policy', fromlist=['x']).load_ontology()['classes']}


def _fresh(staged):
    image, bundle, output = staged
    return image, bundle, output.parent / 'strict-run'
