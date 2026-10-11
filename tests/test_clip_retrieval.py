"""Retrieval over a bank: it votes, and it can only emit a class the bank holds."""
import pytest

from lynceus.adapters.clip import ClipEmbedder


def test_a_crop_takes_declared_context_and_stays_inside_the_image():
    from PIL import Image
    embedder = ClipEmbedder('unused', context=2.0)
    image = Image.new('RGB', (100, 100))
    assert embedder.crop(image, [40, 40, 60, 60]).size == (40, 40)
    # Context that would leave the image is clipped rather than padded with invented pixels.
    assert embedder.crop(image, [0, 0, 20, 20]).size == (30, 30)


def test_the_vote_prefers_count_over_a_single_close_neighbour():
    torch = pytest.importorskip('torch')
    embedder = ClipEmbedder('unused')
    bank = torch.nn.functional.normalize(torch.tensor([[1.0, 0.02], [1.0, 0.0], [1.0, 0.01]]), dim=-1)
    labels = ['truck', 'car', 'car']
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.02]]), dim=-1)
    result = embedder.nearest(query, bank, labels, k=3)[0]
    assert result['label'] == 'car' and result['votes'] == 2 and result['runner_up'] == 'truck'


def test_retrieval_cannot_emit_a_class_the_bank_does_not_hold():
    torch = pytest.importorskip('torch')
    embedder = ClipEmbedder('unused')
    bank = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    query = torch.nn.functional.normalize(torch.tensor([[0.7, 0.7]]), dim=-1)
    assert embedder.nearest(query, bank, ['car', 'bus'], k=1)[0]['label'] in ('car', 'bus')


def test_an_empty_bank_returns_nothing_rather_than_guessing():
    torch = pytest.importorskip('torch')
    embedder = ClipEmbedder('unused')
    assert embedder.nearest(torch.zeros(2, 4), torch.zeros(0, 4), [], k=3) == []
