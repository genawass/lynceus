"""Arm construction: a label policy is declared, applied uniformly, and never silently."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/visdrone'))
from export_students import CLASSES, arm_label, to_yolo  # noqa: E402

MAPPING = {'person': 0, 'car': 1, 'van': 1, 'truck': 2, 'bus': 2}


def obj(label):
    return {'label': {'id': label}}


def test_an_exact_class_ignores_the_policy():
    for policy in ('drop', 'modal', 'verifier_label'):
        assert arm_label(obj('truck'), None, MAPPING, policy) == (CLASSES.index('longvehicle'), None)


def test_a_class_the_reference_leaves_as_background_is_dropped_with_its_name():
    index, reason = arm_label(obj('motorcycle'), None, MAPPING, 'modal')
    assert index is None and reason == 'unmapped_class:motorcycle'


@pytest.mark.parametrize('policy,verifier,expected', [
    ('drop', 'car', (None, 'parent_fallback_dropped')),
    ('modal', None, (1, None)),
    ('verifier_label', 'bus', (2, None)),
    ('verifier_label', None, (None, 'parent_fallback_unresolved_by_verifier')),
    ('verifier_label', 'motorcycle', (None, 'parent_fallback_unresolved_by_verifier')),
])
def test_the_parent_fallback_is_resolved_by_the_declared_policy(policy, verifier, expected):
    verification = {'verifier_label': verifier} if verifier else {}
    assert arm_label(obj('vehicle'), verification, MAPPING, policy) == expected


def test_geometry_is_normalized_to_centre_and_extent():
    assert to_yolo([10, 20, 30, 60], 100, 200, 1) == '1 0.200000 0.200000 0.200000 0.200000'
