"""The review page: what it must contain, and what it must never contain."""
import json
import re

from lynceus.policy import available_ontologies
from lynceus.reference import plan_review
from lynceus.review_ui import render_review_page

ONTOLOGY = available_ontologies()['aerial-traffic-strict-v1']


def build(size=(600, 600)):
    plan = plan_review(*size)
    record = {'id': 'img-0', 'sha256': 'a' * 64, 'scene_id': 's', 'near_duplicate_group': 's',
              'size': list(size), 'visited_tiles': [], 'annotations': [],
              'how_to_fill': {'x': 'y'}, 'planned_tiles': [t['id'] for t in plan]}
    return render_review_page(record, plan, ONTOLOGY, 'image.jpg').decode(), plan


def embedded_state(page):
    match = re.search(r'const S = (\{.*?\});\n', page, re.S)
    return json.loads(match.group(1))


def test_the_page_loads_nothing_from_off_this_host():
    """The constraint is no external dependency, not no request at all.

    The page saves with one same-origin request back to the process that served it, which is how a
    panel on a remote host is annotated through a tunnel without the file landing on the wrong
    machine. Nothing it references has a destination off the host.
    """
    page, _ = build()
    assert 'http://' not in page and 'https://' not in page
    assert '<script src' not in page and 'link rel="stylesheet"' not in page
    assert '//' not in page.split('<script>')[1].split('fetch(')[1][:40]


def test_the_save_request_is_relative_and_falls_back_to_a_download():
    page, _ = build()
    # Relative target: it can only reach whatever served the page.
    assert "fetch('reference.json'" in page
    assert "method:'PUT'" in page
    # Opened as a plain file there is nothing to write to, so the download path must remain.
    assert "location.protocol !== 'file:'" in page
    assert 'a.download' in page


def test_a_blind_page_shows_no_prediction():
    page, _ = build()
    state = embedded_state(page)
    assert state['mode'] == 'blind'
    assert state['predictions'] == []
    assert 'annotation.json' not in page and 'candidates.jsonl' not in page


def test_a_verification_page_carries_geometry_and_class_and_nothing_else():
    """No score beside a box.

    A confidence number would steer the annotator toward the model's own judgement, which is the
    thing being tested, so only what is needed to find and name the box travels into the page.
    """
    plan = plan_review(600, 600)
    record = {'id': 'img-0', 'sha256': 'a' * 64, 'scene_id': 's', 'near_duplicate_group': 's',
              'size': [600, 600], 'visited_tiles': [], 'annotations': []}
    predictions = [{'id': 'obj-0', 'bbox_xyxy': [1, 2, 3, 4], 'label': 'car',
                    'model_score': 0.91, 'status': 'uncertain'}]
    page = render_review_page(record, plan, ONTOLOGY, 'image.jpg', predictions).decode()
    state = embedded_state(page)
    assert state['mode'] == 'verify'
    assert state['predictions'] == [{'id': 'obj-0', 'bbox_xyxy': [1.0, 2.0, 3.0, 4.0], 'label': 'car'}]
    assert 'model_score' not in page and '0.91' not in page
    assert 'uncertain' not in page


def test_a_kept_proposal_is_recorded_as_the_models_geometry():
    plan = plan_review(600, 600)
    record = {'id': 'img-0', 'sha256': 'a' * 64, 'scene_id': 's', 'near_duplicate_group': 's',
              'size': [600, 600], 'visited_tiles': [], 'annotations': []}
    page = render_review_page(record, plan, ONTOLOGY, 'image.jpg',
                              [{'id': 'obj-0', 'bbox_xyxy': [1, 2, 3, 4], 'label': 'car'}]).decode()
    # Accepting a box unchanged must mark it accepted, not added, or borrowed geometry becomes
    # indistinguishable from the annotator's own.
    assert "origin: 'accepted'" in page
    assert "redrawing || 'added'" in page


def test_every_planned_tile_is_in_the_page_so_the_sweep_can_be_recorded():
    page, plan = build()
    state = embedded_state(page)
    assert [t['id'] for t in state['plan']] == [t['id'] for t in plan]
    assert len(plan) > 1


def test_the_class_list_comes_from_the_ontology_and_drops_unusable_labels():
    page, _ = build()
    ids = {c['id'] for c in embedded_state(page)['classes']}
    assert 'car' in ids and 'person' in ids
    # `entity` and `stuff` are not classes an annotator should assign to an entity here.
    assert 'entity' not in ids and 'stuff' not in ids


def test_the_page_offers_every_field_ingest_requires():
    page, _ = build()
    for field in ('f-label', 'f-kind', 'f-layer', 'f-res'):
        assert f'id="{field}"' in page
    for value in ('instance', 'part', 'group', 'unresolved', 'reflected', 'depicted'):
        assert f'value="{value}"' in page


def test_guidance_text_does_not_leak_into_the_page_state():
    page, _ = build()
    state = embedded_state(page)
    assert 'how_to_fill' not in state['record'] and 'planned_tiles' not in state['record']


def test_the_title_is_escaped():
    plan = plan_review(600, 600)
    record = {'id': '<script>x</script>', 'sha256': 'a' * 64, 'scene_id': 's',
              'near_duplicate_group': 's', 'size': [600, 600], 'visited_tiles': [], 'annotations': []}
    page = render_review_page(record, plan, ONTOLOGY, 'image.jpg').decode()
    assert '<title>Reference review - &lt;script&gt;' in page
