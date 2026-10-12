"""Per-stage cost: separated, attributed only where it is measured, and priced per object."""
import json

import pytest

from lynceus.pipeline import annotate, stage_cost


class Verifier:
    """Confirms boxes whose left edge is under 50, and reports its own model passes."""

    def __init__(self):
        # Preloaded, as a verifier reused across images would be: the run must report its own
        # passes rather than every pass the instance has ever made.
        self.stats = {'vision_passes': 20, 'decoder_passes': 40, 'prompt_scorings': 40, 'tiles': 20}

    def capabilities(self):
        return {'device': 'cpu', 'prompt': 'text'}

    def detect(self, image, prompts):
        for key, passes in (('vision_passes', 2), ('decoder_passes', 4),
                            ('prompt_scorings', 4), ('tiles', 2)):
            self.stats[key] += passes
        return [{'bbox_xyxy': [0, 0, 45, 45], 'label': 'cup', 'score': .8}]


class TwoBoxes:
    def predict(self, view):
        return [{'bbox_xyxy': [2, 2, 44, 44], 'label': 'cup', 'model_score': .9, 'crop_edge': False},
                {'bbox_xyxy': [60, 5, 95, 40], 'label': 'cup', 'model_score': .9, 'crop_edge': False}]


def test_a_stage_that_did_not_run_is_absent_rather_than_free():
    cost = stage_cost({'discovery': 2.0, 'verify': 1.0}, 4.0, 2)
    assert 'refine' not in cost['stage_seconds'], 'a stage that never ran must not read as zero cost'
    assert cost['unattributed_seconds'] == 1.0
    assert cost['seconds_per_retained_object'] == 2.0
    assert cost['retained_objects_per_hour'] == 1800.0


def test_cost_per_object_is_undefined_without_objects():
    cost = stage_cost({'discovery': 1.0}, 1.0, 0)
    assert cost['seconds_per_retained_object'] is None


def test_each_stage_reports_its_own_wall_time(staged):
    image, bundle, output = staged
    result = annotate(image, bundle, output, adapter=TwoBoxes(), verifier=Verifier())
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    costs = {e['stage']: e['cost'] for e in events}
    assert costs['discovery']['wall_seconds'] > 0 and costs['verification']['wall_seconds'] > 0
    assert costs['discovery']['model_calls'] == 1
    # The verifier's own pass count travels, because tiles times prompts is the number that
    # explains the wall time and the number a cheaper schedule has to reduce. It counts this
    # run's passes, not the instance's lifetime total.
    assert costs['verification']['model_calls'] == 4
    assert costs['refine']['wall_seconds'] is None, 'refinement did not run in this configuration'
    terminal = costs['terminal']
    assert sum(terminal['stage_seconds'].values()) <= result['summary']['resources']['wall_seconds']


def test_cost_per_object_reaches_the_summary(staged):
    image, bundle, output = staged
    result = annotate(image, bundle, output, adapter=TwoBoxes(), verifier=Verifier())
    resources = result['summary']['resources']
    assert resources['seconds_per_retained_object'] > 0
    assert resources['retained_objects_per_hour'] > 0
    assert set(resources['stage_seconds']) >= {'discovery', 'merge', 'verify'}


def test_dropping_the_verifier_masks_keeps_the_surviving_boxes_identical():
    """The processor filters on score alone, so a box-and-score path must agree with it exactly."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor

    from lynceus.adapters.sam3 import Sam3Verifier

    class Output:
        pred_logits = torch.tensor([[2.0, -3.0, 0.4], [1.5, -0.2, -5.0]])
        pred_boxes = torch.tensor([[[.1, .1, .2, .2], [.3, .3, .4, .4], [.5, .5, .7, .7]],
                                   [[.0, .0, .1, .1], [.2, .2, .5, .5], [.6, .6, .9, .9]]])
        presence_logits = torch.tensor([[1.0], [0.5]])
        pred_masks = torch.zeros(2, 3, 8, 8)

    sizes = [(64, 128), (64, 128)]
    reference = Sam3ImageProcessor().post_process_instance_segmentation(
        Output(), threshold=.3, target_sizes=sizes)
    verifier = Sam3Verifier.__new__(Sam3Verifier)
    verifier.threshold = .3
    ours = verifier._boxes_and_scores(Output(), sizes[0])
    for expected, (boxes, scores) in zip(reference, ours):
        assert scores == expected['scores'].tolist()
        assert boxes == expected['boxes'].tolist()
