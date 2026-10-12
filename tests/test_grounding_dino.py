"""Grounding DINO's label path: spans come back, classes must go forward."""
import pytest

from lynceus.adapters.grounding_dino import build_prompt, map_span


PHRASES = ['person', 'vehicle', 'car', 'van', 'truck', 'bus', 'motorcycle', 'bicycle', 'tricycle']


def test_prompt_is_period_separated_lowercase():
    prompt = build_prompt(['Person', 'motorcycle.', ' bus '])
    assert prompt == 'person. motorcycle. bus.'


@pytest.mark.parametrize('span,expected', [
    ('car', 'car'),
    ('Car.', 'car'),
    ('a person', 'person'),          # grounder merged a neighbouring token
    ('motor', 'motorcycle'),         # subword fragment of a prompted phrase
    ('bicycle', 'bicycle'),
])
def test_spans_map_back_to_the_prompted_phrase(span, expected):
    assert map_span(span, PHRASES) == expected


def test_unrelated_span_maps_to_nothing():
    """A span that is not a prompted phrase must not be forced onto the nearest class."""
    assert map_span('traffic light', PHRASES) is None
    assert map_span('', PHRASES) is None
    assert map_span('   ', PHRASES) is None


def test_ambiguous_containment_prefers_the_longer_phrase():
    assert map_span('bus', ['bus', 'minibus']) == 'bus'
    assert map_span('minibus', ['bus', 'minibus']) == 'minibus'


def test_class_ids_travel_not_phrasing():
    """The verifier reports ontology ids, so agreement cannot depend on wording."""
    prompts = {'a person on foot': 'person', 'car': 'car'}
    phrase = map_span('a person on foot', list(prompts))
    assert prompts[phrase] == 'person'


def test_vocabulary_is_registered_for_this_adapter():
    from lynceus.vocabulary import load_vocabulary, check_adapter, resolve_prompts
    from lynceus.policy import load_ontology
    spec = load_vocabulary('grounding-dino-aerial-baseline-v1')
    assert spec['adapter'] == 'grounding-dino'
    resolution = resolve_prompts(load_ontology('aerial-traffic-strict-v2'), spec)
    check_adapter(resolution, 'grounding-dino')
    with pytest.raises(ValueError, match='vocabulary_adapter_mismatch'):
        check_adapter(resolution, 'owlv2')


def test_bundle_knows_the_adapter():
    from lynceus.bundle import ADAPTER_ASSETS
    assert 'grounding-dino' in ADAPTER_ASSETS
    assert 'preprocessor_config.json' in ADAPTER_ASSETS['grounding-dino']
    # a bundle cannot pass by carrying another adapter's file set
    assert ADAPTER_ASSETS['grounding-dino'] != ADAPTER_ASSETS['sam3']
