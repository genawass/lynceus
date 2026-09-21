"""The refinement stage: lineage, disagreement handling, and what it may not do."""
import json

import pytest

from lynceus.contracts import validate_annotation
from lynceus.geometry import iou
from lynceus.pipeline import refine_objects


class FixedRefiner:
    """A refiner returning declared geometry, so the stage is testable without a checkpoint."""

    def __init__(self, results):
        self.results = results
        self.seen = None

    def capabilities(self):
        return {'device': 'cpu', 'prompt': 'box'}

    def refine(self, image, boxes):
        self.seen = [list(b) for b in boxes]
        return self.results


class Canvas:
    size = (100, 100)


def make(box, identifier='obj-0'):
    obj = {'id': identifier, 'kind': 'instance', 'status': 'uncertain', 'scene_layer': 'physical',
           'bbox_xyxy': [float(v) for v in box], 'mask_ref': None, 'alternative_boxes': [],
           'label': {'id': 'car', 'name': 'car'}, 'label_alternatives': [],
           'uncertainty': {'existence': 'not_assessed', 'count': 'not_assessed', 'boundary': 'not_assessed',
                           'class': 'not_assessed', 'granularity': 'not_assessed', 'reasons': []},
           'evidence_refs': ['evidence/candidates.jsonl'], 'relations': [], 'occluded': None,
           'truncated': None, 'flag_evidence_refs': [], 'count': None, 'calibrated_estimates': []}
    evidence = {'score': 0.9, 'views': 1, 'view_edge': False, 'alternatives': [], 'useful': True}
    return obj, evidence


def test_close_agreement_replaces_geometry_without_manufacturing_uncertainty():
    obj, ev = make([10, 10, 30, 30])
    refined = [{'bbox_xyxy': [11, 11, 30, 30], 'mask_quality': 0.9, 'view': 'refine-0'}]
    candidates = []
    report = refine_objects([obj], [ev], candidates, FixedRefiner(refined), Canvas(), 0.7, 8)
    assert obj['bbox_xyxy'] == [11.0, 11.0, 30.0, 30.0]
    assert obj['alternative_boxes'] == []
    assert obj['uncertainty']['boundary'] == 'not_assessed'
    assert report == {'refined': 1, 'extent_disagreements': 0, 'tolerance': 0.7}


def test_extent_disagreement_keeps_the_proposal_and_unresolves_the_boundary():
    obj, ev = make([10, 10, 30, 30])
    # IoU 0.25: the two models agree where the object is and disagree how far it extends.
    refined = [{'bbox_xyxy': [10, 10, 20, 20], 'mask_quality': 0.4, 'view': 'refine-0'}]
    candidates = []
    report = refine_objects([obj], [ev], candidates, FixedRefiner(refined), Canvas(), 0.7, 8)
    assert obj['bbox_xyxy'] == [10.0, 10.0, 20.0, 20.0]
    assert [10.0, 10.0, 30.0, 30.0] in obj['alternative_boxes']
    assert obj['uncertainty']['boundary'] == 'unresolved'
    assert 'refiner_extent_disagreement' in obj['uncertainty']['reasons']
    assert ev['alternatives'][0] == [10.0, 10.0, 30.0, 30.0]
    assert report['extent_disagreements'] == 1


def test_the_proposal_is_always_recoverable_from_the_candidate_store():
    obj, ev = make([10, 10, 30, 30])
    refined = [{'bbox_xyxy': [12, 12, 28, 28], 'mask_quality': 0.77, 'view': 'refine-3'}]
    candidates = []
    refine_objects([obj], [ev], candidates, FixedRefiner(refined), Canvas(), 0.7, 8)
    record = next(c for c in candidates if c['disposition'] == 'refined')
    assert record['proposed_bbox_xyxy'] == [10.0, 10.0, 30.0, 30.0]
    assert record['bbox_xyxy'] == [12.0, 12.0, 28.0, 28.0]
    assert record['mask_quality'] == 0.77
    assert record['object'] == 'obj-0'
    assert 0 < record['agreement_iou'] < 1


def test_a_refinement_outside_the_image_is_rejected_and_the_proposal_stands():
    obj, ev = make([10, 10, 30, 30])
    refined = [{'bbox_xyxy': [10, 10, 500, 500], 'mask_quality': 0.9, 'view': 'refine-0'}]
    candidates = []
    report = refine_objects([obj], [ev], candidates, FixedRefiner(refined), Canvas(), 0.7, 8)
    assert obj['bbox_xyxy'] == [10.0, 10.0, 30.0, 30.0]
    assert report['refined'] == 0
    assert candidates[0]['reason'] == 'invalid_refined_geometry'


def test_an_absent_refinement_keeps_the_proposed_geometry():
    obj, ev = make([10, 10, 30, 30])
    report = refine_objects([obj], [ev], [], FixedRefiner([None]), Canvas(), 0.7, 8)
    assert obj['bbox_xyxy'] == [10.0, 10.0, 30.0, 30.0]
    assert report['refined'] == 0 and ev['refined'] is False


def test_refinement_never_changes_how_many_objects_there_are():
    objects, evidence = zip(*[make([10, 10, 30, 30], 'obj-0'), make([50, 50, 70, 70], 'obj-1')])
    objects, evidence = list(objects), list(evidence)
    refined = [{'bbox_xyxy': [11, 11, 29, 29], 'mask_quality': 0.8, 'view': 'refine-0'},
               {'bbox_xyxy': [51, 51, 69, 69], 'mask_quality': 0.8, 'view': 'refine-1'}]
    refiner = FixedRefiner(refined)
    refine_objects(objects, evidence, [], refiner, Canvas(), 0.7, 8)
    assert len(objects) == 2
    # The refiner is shown the proposed boxes and nothing else.
    assert refiner.seen == [[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 70.0, 70.0]]


def test_alternatives_stay_within_the_declared_cap():
    obj, ev = make([10, 10, 30, 30])
    obj['alternative_boxes'] = [[float(i), 1.0, float(i + 5), 6.0] for i in range(8)]
    refined = [{'bbox_xyxy': [10, 10, 20, 20], 'mask_quality': 0.4, 'view': 'refine-0'}]
    refine_objects([obj], [ev], [], FixedRefiner(refined), Canvas(), 0.7, 8)
    assert len(obj['alternative_boxes']) == 8
    assert obj['alternative_boxes'][0] == [10.0, 10.0, 30.0, 30.0]
