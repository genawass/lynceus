"""The prompt search: that it optimises the whole vocabulary, and records what it tried."""
import pytest

from lynceus.prompt_search import coordinate_ascent, objective, score_vocabulary


def test_the_objective_rewards_neither_flooding_nor_silence():
    eligible = 100
    balanced = objective({'named': 50, 'retained': 60}, eligible)
    flooding = objective({'named': 60, 'retained': 2000}, eligible)
    silent = objective({'named': 3, 'retained': 3}, eligible)
    assert balanced > flooding
    assert balanced > silent


def test_the_objective_counts_correctly_labelled_finds_not_merely_localized_ones():
    """A class-agnostic objective was measured and rejected: it bought localization by abstaining.

    A vocabulary that localizes everything and names nothing must score zero, or the search will
    find it.
    """
    eligible = 100
    localizes_but_abstains = objective({'matched': 90, 'named': 0, 'retained': 100}, eligible)
    names_fewer_correctly = objective({'matched': 40, 'named': 40, 'retained': 100}, eligible)
    assert localizes_but_abstains == 0.0
    assert names_fewer_correctly > 0.0


def test_the_objective_is_zero_when_nothing_is_found():
    assert objective({'named': 0, 'retained': 500}, 100) == 0.0
    assert objective({'named': 0, 'retained': 0}, 100) == 0.0


def test_the_search_adopts_only_changes_that_improve_the_whole_vocabulary():
    """The emitted label is the class that wins over all prompts, so a per-class score means nothing.

    Here 'pedestrian' helps person but costs more elsewhere, so a search scoring the whole
    vocabulary must reject it.
    """
    def evaluate(vocabulary):
        if vocabulary == {'person': 'pedestrian', 'car': 'car'}:
            return 0.4                      # better for person alone, worse overall
        if vocabulary == {'person': 'person', 'car': 'automobile'}:
            return 0.8
        return 0.5
    result = coordinate_ascent(['person', 'car'],
                               {'person': ['pedestrian'], 'car': ['automobile']},
                               evaluate, baseline={'person': 'person', 'car': 'car'})
    assert result['vocabulary'] == {'person': 'person', 'car': 'automobile'}
    assert result['score'] == 0.8


def test_a_candidate_rejected_early_can_win_after_another_class_changes():
    """Adopting one phrasing changes what the others compete against, which is why rounds exist."""
    scores = {('person', 'car'): 0.5, ('pedestrian', 'car'): 0.4,
              ('person', 'automobile'): 0.6, ('pedestrian', 'automobile'): 0.9}
    result = coordinate_ascent(['person', 'car'],
                               {'person': ['pedestrian'], 'car': ['automobile']},
                               lambda v: scores[(v['person'], v['car'])],
                               baseline={'person': 'person', 'car': 'car'}, rounds=2)
    assert result['vocabulary'] == {'person': 'pedestrian', 'car': 'automobile'}
    assert result['score'] == 0.9


def test_the_search_stops_when_a_round_changes_nothing():
    result = coordinate_ascent(['person'], {'person': ['pedestrian']},
                               lambda v: 0.5, baseline={'person': 'person'}, rounds=5)
    assert result['rounds_used'] == 1
    assert result['vocabulary'] == {'person': 'person'}


def test_every_candidate_tried_is_recorded_not_only_the_winner():
    """A vocabulary that beat two alternatives and one that beat forty are different artifacts."""
    result = coordinate_ascent(['person'], {'person': ['pedestrian', 'walker', 'human']},
                               lambda v: {'person': 0.5, 'pedestrian': 0.6,
                                          'walker': 0.4, 'human': 0.55}[v['person']],
                               baseline={'person': 'person'}, rounds=1)
    tried = {step.get('phrase') for step in result['trace'] if 'phrase' in step}
    assert tried == {'pedestrian', 'walker', 'human'}
    adopted = [s for s in result['trace'] if s.get('adopted')]
    assert [s['phrase'] for s in adopted] == ['pedestrian']


class FakeView:
    """A view whose class head is a fixed lookup, standing in for cached image features."""

    def __init__(self, image, boxes, table):
        self.image, self.boxes, self.table = image, boxes, table

    def as_view(self):
        return {'image': self.image, 'boxes': self.boxes,
                'predict': lambda embeds: self.table[tuple(embeds)]}


def test_scoring_uses_cached_features_and_counts_only_permitted_labels():
    import numpy as np
    boxes = np.array([[0., 0., 10., 10.], [50., 50., 60., 60.]])
    view = FakeView('img', boxes, {('a',): np.array([[0.9, 0.1], [0.2, 0.8]])}).as_view()
    references = {'img': [{'bbox_xyxy': [0., 0., 10., 10.], 'permitted_labels': ['car']},
                          {'bbox_xyxy': [50., 50., 60., 60.], 'permitted_labels': ['person']}]}
    report = score_vocabulary([view], ('a',), ['car', 'person'], references, threshold=0.5)
    assert report['matched'] == 2 and report['retained'] == 2
    assert report['recall'] == 1.0 and report['precision'] == 1.0
    # Both are exact boxes under permitted labels, so both count at the tight threshold too.
    assert report['named'] == 2 and report['useful_label_coverage'] == 1.0
    assert report['per_class']['car']['matched'] == 1
    assert report['per_class']['person']['matched'] == 1


def test_a_box_matching_a_reference_under_the_wrong_class_is_not_a_class_match():
    import numpy as np
    boxes = np.array([[0., 0., 10., 10.]])
    view = FakeView('img', boxes, {('a',): np.array([[0.2, 0.9]])}).as_view()
    references = {'img': [{'bbox_xyxy': [0., 0., 10., 10.], 'permitted_labels': ['car']}]}
    report = score_vocabulary([view], ('a',), ['car', 'person'], references, threshold=0.5)
    # Localized, so it counts as matched overall, but not as a correctly labelled find.
    assert report['matched'] == 1
    assert report['named'] == 0
    assert report['per_class']['person']['matched'] == 0
