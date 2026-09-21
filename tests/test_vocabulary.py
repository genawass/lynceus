"""Per-model prompt vocabularies: what they must declare, and what they must not be used for."""
import pytest

from lynceus.policy import load_ontology
from lynceus.vocabulary import (available_vocabularies, check_adapter, load_vocabulary,
                                resolve_prompts, validate_vocabulary)

ONTOLOGY = load_ontology('aerial-traffic-strict-v1')


def spec(**overrides):
    base = {'id': 'test-v1', 'version': '1.0', 'adapter': 'owlv2',
            'ontology': 'aerial-traffic-strict-v1',
            'prompts': {'person': 'pedestrian seen from above', 'car': 'car'}}
    base.update(overrides)
    return base


def test_every_shipped_vocabulary_is_valid():
    found = available_vocabularies()
    assert found, 'no vocabularies shipped'
    for vocabulary in found.values():
        validate_vocabulary(vocabulary)


def test_a_vocabulary_declares_what_it_maps_from_and_what_it_is_for():
    for field in ('id', 'version', 'adapter', 'ontology', 'prompts'):
        incomplete = spec()
        del incomplete[field]
        with pytest.raises(ValueError, match=f'missing_vocabulary_field: {field}'):
            validate_vocabulary(incomplete)


def test_a_vocabulary_cannot_cover_a_class_the_ontology_does_not_have():
    with pytest.raises(ValueError, match='vocabulary_covers_unknown_class'):
        validate_vocabulary(spec(prompts={'spaceship': 'spaceship'}))


def test_an_empty_prompt_is_refused():
    with pytest.raises(ValueError, match='empty_prompt_for_class'):
        validate_vocabulary(spec(prompts={'car': '   '}))


def test_the_prompt_is_the_vocabularys_phrasing_not_the_ontologys_name():
    resolution = resolve_prompts(ONTOLOGY, spec(), classes=['person', 'car'])
    assert dict(zip(resolution['classes'], resolution['prompts'])) == {
        'person': 'pedestrian seen from above', 'car': 'car'}


def test_an_uncovered_class_falls_back_and_the_fallback_is_recorded():
    """A vocabulary covering half the ontology must not look like one covering all of it."""
    resolution = resolve_prompts(ONTOLOGY, spec(), classes=['person', 'car', 'bus'])
    assert resolution['prompts'][resolution['classes'].index('bus')] == 'bus'
    assert resolution['fallbacks'] == ['bus']
    assert resolution['coverage'] == round(2 / 3, 4)


def test_a_run_without_a_vocabulary_is_recorded_as_entirely_uncalibrated():
    resolution = resolve_prompts(ONTOLOGY, None)
    assert resolution['vocabulary'] is None
    assert resolution['coverage'] == 0.0
    assert len(resolution['fallbacks']) == len(resolution['classes'])


def test_a_vocabulary_is_recorded_by_identity_and_hash():
    resolution = resolve_prompts(ONTOLOGY, spec())
    recorded = resolution['vocabulary']
    assert recorded['id'] == 'test-v1' and recorded['version'] == '1.0'
    assert recorded['adapter'] == 'owlv2'
    assert len(recorded['sha256']) == 64
    # A different phrasing is a different vocabulary, so the hash must move.
    other = resolve_prompts(ONTOLOGY, spec(prompts={'person': 'person'}))
    assert other['vocabulary']['sha256'] != recorded['sha256']


def test_one_models_vocabulary_cannot_be_handed_to_another():
    """Phrasing does not transfer, which is the whole reason vocabularies are per model."""
    resolution = resolve_prompts(ONTOLOGY, spec(adapter='owlv2'))
    with pytest.raises(ValueError, match='vocabulary_adapter_mismatch'):
        check_adapter(resolution, 'sam3')
    assert check_adapter(resolution, 'owlv2') is resolution


def test_an_uncalibrated_run_passes_the_adapter_check():
    # No vocabulary means nothing to mismatch; it is a legitimate run, just an uncalibrated one.
    assert check_adapter(resolve_prompts(ONTOLOGY, None), 'sam3')


def test_an_unknown_vocabulary_id_is_refused():
    with pytest.raises(ValueError, match='unknown_vocabulary'):
        load_vocabulary('no-such-vocabulary')


def test_the_shipped_owlv2_and_sam3_vocabularies_are_separate_artifacts():
    found = available_vocabularies()
    owl = found['owlv2-aerial-baseline-v1']
    sam = found['sam3-aerial-baseline-v1']
    assert owl['adapter'] == 'owlv2' and sam['adapter'] == 'sam3'
    # They hold the same strings today; that they are separate is what lets them diverge.
    assert owl['id'] != sam['id']
    assert owl.get('fitted_on') is None and sam.get('fitted_on') is None
