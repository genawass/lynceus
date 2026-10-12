"""Conservative naming (WP-08): prefer a supported broader class over a coin-flip between siblings."""
import pytest

from lynceus.pipeline import resolve_label, name_observations, annotate, DEFAULT_NAME_RATIO
from lynceus.policy import load_ontology

AERIAL = load_ontology('aerial-traffic-v1')
BROAD = load_ontology('lynceus-broad-v1')


def test_clear_winner_keeps_the_specific_label():
    label, alternatives, reason = resolve_label({'car': .40, 'van': .05, 'person': .01}, AERIAL, .6)
    assert (label, alternatives) == ('car', [])
    assert reason == 'specific_label_clear_of_runner_up'


def test_competing_siblings_resolve_to_the_broader_class():
    """car vs van is the confusion the aerial ontology introduced; vehicle is permitted for both."""
    label, alternatives, reason = resolve_label({'car': .40, 'van': .36, 'person': .01}, AERIAL, .6)
    assert label == 'vehicle'
    assert sorted(alternatives) == ['car', 'van']
    assert reason == 'broader_class_covers_competing_siblings'


def test_motorcycle_versus_bicycle_resolves_to_vehicle():
    label, _, _ = resolve_label({'motorcycle': .30, 'bicycle': .28}, AERIAL, .6)
    assert label == 'vehicle'


def test_competing_unrelated_classes_become_unknown_not_a_guess():
    """person vs car share only `entity`, which is not a useful class."""
    label, alternatives, reason = resolve_label({'person': .30, 'car': .29}, AERIAL, .6)
    assert label == 'unknown_object'
    assert sorted(alternatives) == ['car', 'person']
    assert reason == 'no_useful_common_ancestor'


def test_nested_pair_takes_the_broader_of_the_two():
    label, _, reason = resolve_label({'car': .30, 'vehicle': .29}, AERIAL, .6)
    assert label == 'vehicle' and reason == 'broader_of_nested_pair'


def test_ratio_is_relative_so_scale_does_not_change_the_decision():
    small = resolve_label({'car': .040, 'van': .036}, AERIAL, .6)[0]
    large = resolve_label({'car': .400, 'van': .360}, AERIAL, .6)[0]
    assert small == large == 'vehicle'
    # a weak runner-up leaves the specific label alone at either scale
    assert resolve_label({'car': .040, 'van': .004}, AERIAL, .6)[0] == 'car'
    assert resolve_label({'car': .400, 'van': .040}, AERIAL, .6)[0] == 'car'


def test_ratio_of_one_disables_the_fallback():
    assert resolve_label({'car': .40, 'van': .399}, AERIAL, 1.0)[0] == 'car'


def test_broad_ontology_falls_back_within_its_own_hierarchy():
    assert resolve_label({'cup': .30, 'bowl': .29}, BROAD, .6)[0] == 'container'


@pytest.mark.parametrize('scores', [{'car': 0.0, 'van': 0.0}, {'car': .5}])
def test_degenerate_score_rows_do_not_crash(scores):
    label, _, _ = resolve_label(scores, AERIAL, .6)
    assert label in {c['id'] for c in AERIAL['classes']}


def test_naming_runs_before_merging_so_disagreeing_views_can_agree(staged):
    """Two views calling the same object car and van must not become two instances."""
    image, bundle, run = staged
    class Disagreeing:
        def __init__(self): self.n = 0
        def predict(self, view):
            self.n += 1
            scores = {'car': .40, 'van': .36} if self.n % 2 else {'van': .40, 'car': .36}
            top = max(scores, key=scores.get)
            return [{'bbox_xyxy': [2, 2, view.width - 2, view.height - 2], 'label': top,
                     'model_score': scores[top], 'class_scores': scores, 'crop_edge': False}]
    result = annotate(image, bundle, run, adapter=Disagreeing(),
                      tile_levels=0, ontology_id='aerial-traffic-v1')
    assert [o['label']['id'] for o in result['objects']] == ['vehicle']
    assert sorted(a['id'] for a in result['objects'][0]['label_alternatives']) == ['car', 'van']


def test_adapter_without_score_distribution_is_left_alone():
    observations = [{'label': 'cup', 'class_scores': None}]
    assert name_observations(observations, BROAD, .6)[0]['label'] == 'cup'


def test_default_ratio_is_declared():
    assert DEFAULT_NAME_RATIO == 0.6


def test_co_occurring_classes_do_not_trigger_the_broader_fallback():
    """A person on a motorcycle is two entities in one box, not one ambiguous entity.

    Falling back here would discard a supported class to resolve a conflict that does not exist,
    which is the same error as letting containment imply identity.
    """
    from lynceus.pipeline import resolve_label
    from lynceus.policy import load_ontology
    ontology = load_ontology('aerial-traffic-strict-v2')
    label, alternatives, reason = resolve_label({'person': 0.6, 'motorcycle': 0.55}, ontology, 0.6)
    assert label == 'person'
    assert sorted(alternatives) == ['motorcycle', 'person']
    assert reason == 'co_occurring_classes_not_competing'


def test_exclusive_classes_still_fall_back_to_the_broader_one():
    from lynceus.pipeline import resolve_label
    from lynceus.policy import load_ontology
    ontology = load_ontology('aerial-traffic-strict-v2')
    # A region cannot be both a car and a van, so the supported claim is what they share.
    label, _, reason = resolve_label({'car': 0.6, 'van': 0.58}, ontology, 0.6)
    assert label == 'vehicle'
    assert reason == 'broader_class_covers_competing_siblings'


def test_the_previous_ontology_is_unchanged_and_still_abstains():
    """Version 1 stays valid for the runs measured under it; this is a new version, not an edit."""
    from lynceus.pipeline import resolve_label
    from lynceus.policy import load_ontology
    label, _, reason = resolve_label({'person': 0.6, 'motorcycle': 0.55},
                                     load_ontology('aerial-traffic-strict-v1'), 0.6)
    assert label == 'unknown_object'
    assert reason == 'no_useful_common_ancestor'


def test_co_occurrence_must_be_declared_not_inferred():
    from lynceus.policy import co_occurring, load_ontology
    ontology = load_ontology('aerial-traffic-strict-v2')
    assert co_occurring(ontology, 'person', 'bicycle')
    assert co_occurring(ontology, 'bicycle', 'person')      # order does not matter
    # Two vehicles are not declared co-occurring: one region is not both a car and a bus.
    assert not co_occurring(ontology, 'car', 'bus')
    assert not co_occurring(ontology, 'car', 'car')
