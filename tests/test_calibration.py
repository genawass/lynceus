"""Calibrated estimates: a score becomes a probability only under a recorded external fit."""
import json

import pytest

from lynceus.contracts import validate_annotation
from lynceus.pipeline import annotate, load_calibration, calibrated_probability


def spec(domain='lynceus-broad-v1', **kw):
    base = {'id': 'demo-calibration-v1', 'version': '1.0',
            'event': 'a physical object exists whose visible extent matches this box at IoU >= 0.5',
            'localization_threshold': 0.5,
            'population': 'synthetic fixtures; not a real population',
            'domain': domain, 'feature': 'model_score',
            'bins': [[0.0, 0.5, 0.2], [0.5, 1.0, 0.9]]}
    base.update(kw)
    return base


def write(tmp_path, data, name='calibration.json'):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_absent_calibration_is_recorded_not_silent(staged):
    image, bundle, run = staged
    from conftest import FakeAdapter
    result = annotate(image, bundle, run, adapter=FakeAdapter())
    calibration = result['summary']['calibration']
    assert calibration['status'] == 'inapplicable'
    assert calibration['reason'] == 'no_calibration_artifact_staged'
    assert all(o['calibrated_estimates'] == [] for o in result['objects'])


@pytest.mark.parametrize('bad,expected', [
    ({'domain': 'aerial-traffic-v1'}, 'calibration_domain_mismatch'),
    ({'feature': 'vibes'}, 'unsupported_calibration_feature'),
])
def test_inapplicable_calibration_emits_no_probability(staged, tmp_path, bad, expected):
    """A fit from another domain does not transfer; it yields a reason, never a number."""
    image, bundle, run = staged
    from conftest import FakeAdapter
    result = annotate(image, bundle, run, adapter=FakeAdapter(),
                      calibration=str(write(tmp_path, spec(**bad))))
    assert result['summary']['calibration']['status'] == 'inapplicable'
    assert expected in result['summary']['calibration']['reason']
    assert all(o['calibrated_estimates'] == [] for o in result['objects'])


def test_malformed_calibration_is_rejected(staged, tmp_path):
    image, bundle, run = staged
    from conftest import FakeAdapter
    broken = spec()
    del broken['population'], broken['bins']
    result = annotate(image, bundle, run, adapter=FakeAdapter(),
                      calibration=str(write(tmp_path, broken)))
    assert result['summary']['calibration']['status'] == 'inapplicable'
    assert 'malformed_calibration' in result['summary']['calibration']['reason']


def test_applicable_calibration_emits_a_well_formed_estimate(staged, tmp_path):
    image, bundle, run = staged
    from conftest import FakeAdapter
    result = annotate(image, bundle, run, adapter=FakeAdapter(),
                      calibration=str(write(tmp_path, spec())))
    calibration = result['summary']['calibration']
    assert calibration['status'] == 'applied'
    estimate = result['objects'][0]['calibrated_estimates'][0]
    assert estimate['probability'] == 0.9                      # FakeAdapter scores .99
    assert estimate['calibration_ref'] == 'evidence/calibration.json'
    assert estimate['localization_threshold'] == 0.5
    assert 'IoU' in estimate['event'] and estimate['population']
    # the artifact the estimate points at is in the run and hash-verified
    assert 'calibration' in result['artifacts']
    assert validate_annotation(result, run)['summary']['calibration']['status'] == 'applied'


def test_estimate_without_applied_calibration_is_rejected(staged, tmp_path):
    import copy
    image, bundle, run = staged
    from conftest import FakeAdapter
    result = annotate(image, bundle, run, adapter=FakeAdapter(),
                      calibration=str(write(tmp_path, spec())))
    tampered = copy.deepcopy(result)
    tampered['summary']['calibration'] = {'status': 'inapplicable', 'reason': 'x'}
    with pytest.raises(ValueError, match='calibrated_estimate_while_calibration_inapplicable'):
        validate_annotation(tampered, run)


def test_estimate_must_reference_a_run_artifact(staged, tmp_path):
    import copy
    image, bundle, run = staged
    from conftest import FakeAdapter
    result = annotate(image, bundle, run, adapter=FakeAdapter(),
                      calibration=str(write(tmp_path, spec())))
    tampered = copy.deepcopy(result)
    tampered['objects'][0]['calibrated_estimates'][0]['calibration_ref'] = 'evidence/elsewhere.json'
    with pytest.raises(ValueError, match='unknown_artifact_reference'):
        validate_annotation(tampered, run)


@pytest.mark.parametrize('field,value', [('probability', 1.4), ('probability', -0.1),
                                         ('event', 'short'), ('population', '')])
def test_malformed_estimate_is_rejected(staged, tmp_path, field, value):
    import copy
    image, bundle, run = staged
    from conftest import FakeAdapter
    result = annotate(image, bundle, run, adapter=FakeAdapter(),
                      calibration=str(write(tmp_path, spec())))
    tampered = copy.deepcopy(result)
    tampered['objects'][0]['calibrated_estimates'][0][field] = value
    with pytest.raises(ValueError):
        validate_annotation(tampered, run)


def test_forbidden_completeness_claims_stay_forbidden(staged, tmp_path):
    """v1 permits a probability per box and none about the image being complete."""
    import copy
    image, bundle, run = staged
    from conftest import FakeAdapter
    result = annotate(image, bundle, run, adapter=FakeAdapter(),
                      calibration=str(write(tmp_path, spec())))
    for field in ('probability_all_objects_found', 'is_ground_truth'):
        tampered = copy.deepcopy(result)
        tampered[field] = 0.99
        with pytest.raises(ValueError):
            validate_annotation(tampered, run)


def test_value_outside_every_bin_yields_no_estimate():
    assert calibrated_probability(spec(bins=[[0.0, 0.3, 0.1]]), 0.8) is None
    assert calibrated_probability(spec(), 0.99) == 0.9


def test_loader_reports_applicability_without_a_run():
    assert load_calibration(None, 'lynceus-broad-v1')[1]['status'] == 'inapplicable'
