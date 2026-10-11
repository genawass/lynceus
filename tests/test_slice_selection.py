"""The train slice: a declared rule, recomputed, with sequence-level leakage excluded."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/visdrone'))
from label_train_slice import select_slice, sequence  # noqa: E402


@pytest.fixture
def dataset(tmp_path):
    """Two sequences in val, five in train, two of which also appear in val."""
    val = ['0000115_01031_d_0000082', '0000115_02000_d_0000083', '0000162_01058_d_0000001']
    train = [f'{prefix}_{index:05d}_d_{index:07d}'
             for prefix in ('0000115', '0000162', '0000300', '0000400', '0000500')
             for index in range(1, 7)]
    (tmp_path / 'val.txt').write_text('\n'.join(f'./images/val/{stem}.jpg' for stem in val))
    (tmp_path / 'train.txt').write_text('\n'.join(f'./images/train/{stem}.jpg' for stem in train))
    return tmp_path


def test_a_sequence_present_in_val_cannot_enter_the_slice(dataset):
    chosen = select_slice(dataset, 10, 11)
    assert len(chosen) == 10
    assert {sequence(stem) for stem in chosen} == {'0000300', '0000400', '0000500'}


def test_the_selection_is_a_rule_and_not_a_draw(dataset):
    assert select_slice(dataset, 8, 11) == select_slice(dataset, 8, 11)
    assert select_slice(dataset, 8, 11) != select_slice(dataset, 8, 12)
    assert select_slice(dataset, 8, 11) == sorted(select_slice(dataset, 8, 11))


def test_a_pool_too_small_to_satisfy_the_count_is_an_error_not_a_short_slice(dataset):
    with pytest.raises(SystemExit, match='pool has 18 images'):
        select_slice(dataset, 50, 11)
