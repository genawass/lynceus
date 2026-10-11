"""The review order: declared, explainable per box, and never a judgement."""
import json

import pytest

from lynceus.pipeline import annotate
from lynceus.review import queue, review_run, unresolved


def make(identifier, box, boundary='supported', existence='supported'):
    return {'id': identifier, 'bbox_xyxy': list(box), 'status': 'uncertain',
            'label': {'id': 'car', 'name': 'car'},
            'uncertainty': {'existence': existence, 'class': 'supported', 'boundary': boundary,
                            'count': 'supported', 'granularity': 'supported', 'reasons': []}}


def evidence(score=.9, verified=True, overlap=.9):
    return {'score': score, 'views': 1, 'view_edge': False, 'alternatives': [], 'useful': True,
            'verified': verified, 'verifier_score': .5, 'verifier_overlap': overlap}


def test_more_unresolved_dimensions_is_reviewed_first():
    objects = [make('settled', [0, 0, 10, 10]),
               make('shaky', [0, 0, 10, 10], boundary='unresolved', existence='unresolved')]
    order = [row['object'] for row in queue(objects, [evidence(), evidence()])]
    assert order == ['shaky', 'settled']
    assert unresolved(objects[1]) == 2


def test_an_unconfirmed_box_precedes_a_confirmed_one_with_the_same_dimensions():
    objects = [make('confirmed', [0, 0, 10, 10]), make('unconfirmed', [0, 0, 10, 10])]
    order = [row['object'] for row in queue(objects, [evidence(), evidence(verified=False)])]
    assert order == ['unconfirmed', 'confirmed']


def test_among_confirmed_boxes_the_one_whose_extent_is_disputed_comes_first():
    objects = [make('agreed', [0, 0, 10, 10]), make('disputed', [0, 0, 10, 10])]
    order = [row['object'] for row in queue(objects, [evidence(overlap=.95), evidence(overlap=.55)])]
    assert order == ['disputed', 'agreed']


def test_every_row_carries_why_it_sits_where_it_does():
    row = queue([make('a', [0, 0, 10, 10])], [evidence()])[0]
    assert row['rank'] == 1
    assert set(row['difficulty']) >= {'unresolved_dimensions', 'verified', 'verifier_overlap',
                                      'model_score', 'area', 'label'}


class Verifier:
    def capabilities(self):
        return {'device': 'cpu', 'prompt': 'text'}

    def detect(self, image, prompts):
        return [{'bbox_xyxy': [0, 0, 45, 45], 'label': 'cup', 'score': .8}]


class TwoBoxes:
    def predict(self, view):
        return [{'bbox_xyxy': [2, 2, 44, 44], 'label': 'cup', 'model_score': .9, 'crop_edge': False},
                {'bbox_xyxy': [60, 5, 95, 40], 'label': 'cup', 'model_score': .4, 'crop_edge': False}]


def test_a_finished_run_can_be_read_back_into_a_queue(staged):
    image, bundle, output = staged
    annotate(image, bundle, output, adapter=TwoBoxes(), verifier=Verifier())
    report = review_run(output)
    assert report['retained'] == 2 and len(report['order']) == 2
    # The unconfirmed box is the one a reviewer should see first.
    assert report['order'][0]['difficulty']['verified'] is False
    assert review_run(output, limit=1)['order'] == report['order'][:1]


def test_reading_a_run_requires_its_evidence(staged, tmp_path):
    image, bundle, output = staged
    annotate(image, bundle, output, adapter=TwoBoxes(), verifier=Verifier())
    lines = (output / 'evidence/candidates.jsonl').read_text().splitlines()
    (output / 'evidence/candidates.jsonl').write_text('\n'.join(lines[1:]) + '\n')
    with pytest.raises(ValueError, match='retained_candidate_mismatch'):
        review_run(output)


def test_the_cli_emits_the_queue(staged, capsys):
    from lynceus.cli import main
    image, bundle, output = staged
    annotate(image, bundle, output, adapter=TwoBoxes(), verifier=Verifier())
    assert main(['review', str(output), '--limit', '1']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['budget'] == 1 and len(payload['order']) == 1
