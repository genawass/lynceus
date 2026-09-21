"""The review server: what it will write, and everything it refuses to."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from lynceus.review_server import serve


@pytest.fixture
def panel(tmp_path):
    (tmp_path / 'index.html').write_text('<html></html>')
    folder = tmp_path / 'img-0'
    folder.mkdir()
    (folder / 'reference.json').write_text(json.dumps({'visited_tiles': [], 'annotations': []}))
    return tmp_path


@pytest.fixture
def running(panel):
    server = serve(panel, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_address[1]}', panel
    server.shutdown()
    server.server_close()


def put(base, path, payload):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(f'{base}/{path}', data=body, method='PUT',
                                     headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=5)


def test_a_reference_is_written_back_into_its_image_folder(running):
    base, panel = running
    record = {'id': 'img-0', 'sha256': 'a' * 64, 'scene_id': 's', 'near_duplicate_group': 's',
              'size': [10, 10], 'visited_tiles': ['r0c0'], 'annotations': [{'id': 'ref-0'}]}
    response = put(base, 'img-0/reference.json', record)
    assert response.status == 200
    assert json.loads((panel / 'img-0' / 'reference.json').read_text()) == record
    assert json.loads(response.read())['annotations'] == 1


def test_the_panel_is_served_so_the_page_can_load(running):
    base, _ = running
    assert urllib.request.urlopen(f'{base}/index.html', timeout=5).status == 200


def test_nothing_but_a_reference_file_may_be_written(running):
    base, panel = running
    with pytest.raises(urllib.error.HTTPError) as caught:
        put(base, 'img-0/evil.json', {'id': 'img-0', 'sha256': 'a'*64, 'scene_id': 's', 'near_duplicate_group': 's', 'size': [10,10], 'visited_tiles': [], 'annotations': []})
    assert caught.value.code == 403
    assert not (panel / 'img-0' / 'evil.json').exists()


def test_a_write_cannot_escape_the_panel(running):
    base, panel = running
    with pytest.raises(urllib.error.HTTPError) as caught:
        put(base, '../reference.json', {'id': 'img-0', 'sha256': 'a'*64, 'scene_id': 's', 'near_duplicate_group': 's', 'size': [10,10], 'visited_tiles': [], 'annotations': []})
    assert caught.value.code in (403, 404)
    assert not (panel.parent / 'reference.json').exists()


def test_a_write_cannot_invent_an_image_folder(running):
    base, panel = running
    with pytest.raises(urllib.error.HTTPError) as caught:
        put(base, 'not-an-image/reference.json', {'id': 'img-0', 'sha256': 'a'*64, 'scene_id': 's', 'near_duplicate_group': 's', 'size': [10,10], 'visited_tiles': [], 'annotations': []})
    assert caught.value.code == 403
    assert not (panel / 'not-an-image').exists()


def test_the_panel_root_itself_is_not_a_write_target(running):
    base, panel = running
    with pytest.raises(urllib.error.HTTPError) as caught:
        put(base, 'reference.json', {'id': 'img-0', 'sha256': 'a'*64, 'scene_id': 's', 'near_duplicate_group': 's', 'size': [10,10], 'visited_tiles': [], 'annotations': []})
    assert caught.value.code == 403
    assert not (panel / 'reference.json').exists()


def test_something_that_is_not_a_reference_record_is_refused(running):
    base, panel = running
    original = (panel / 'img-0' / 'reference.json').read_text()
    for payload in (b'not json at all', json.dumps({'nope': 1}).encode()):
        with pytest.raises(urllib.error.HTTPError) as caught:
            put(base, 'img-0/reference.json', payload)
        assert caught.value.code == 400
    assert (panel / 'img-0' / 'reference.json').read_text() == original


def test_a_directory_without_a_prepared_panel_is_refused(tmp_path):
    with pytest.raises(ValueError, match='not_a_prepared_panel'):
        serve(tmp_path)


def test_the_suite_still_forbids_reaching_off_this_host(network_is_still_denied):
    with pytest.raises(AssertionError, match='network forbidden'):
        urllib.request.urlopen('http://example.com', timeout=1)


def test_a_partial_record_cannot_strip_the_panel_identity(running):
    """The identity fields are what the panel is checked on; losing them must fail loudly here.

    Ingestion would catch it, but only long after the annotator moved on, and by then the original
    skeleton is gone.
    """
    base, panel = running
    full = {'id': 'img-0', 'sha256': 'a' * 64, 'scene_id': 's', 'near_duplicate_group': 's',
            'size': [10, 10], 'visited_tiles': [], 'annotations': []}
    (panel / 'img-0' / 'reference.json').write_text(json.dumps(full))
    with pytest.raises(urllib.error.HTTPError) as caught:
        put(base, 'img-0/reference.json', {'visited_tiles': [], 'annotations': []})
    assert caught.value.code == 400
    assert 'sha256' in caught.value.read().decode()
    assert json.loads((panel / 'img-0' / 'reference.json').read_text()) == full


def test_a_record_cannot_be_saved_into_another_images_folder(running):
    base, panel = running
    (panel / 'img-1').mkdir()
    record = {'id': 'img-0', 'sha256': 'a' * 64, 'scene_id': 's', 'near_duplicate_group': 's',
              'size': [10, 10], 'visited_tiles': [], 'annotations': []}
    with pytest.raises(urllib.error.HTTPError) as caught:
        put(base, 'img-1/reference.json', record)
    assert caught.value.code == 409
    assert not (panel / 'img-1' / 'reference.json').exists()
