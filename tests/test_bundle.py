import json
import pytest
from lynceus.bundle import preflight,doctor,BundleError,metadata_version

def test_bundle(staged):
    image,bundle,run=staged;assert preflight(bundle)['verified_assets']==4
    (bundle.parent/'model/config.json').write_text('changed')
    with pytest.raises(ValueError,match='hash_mismatch'):preflight(bundle)

def test_missing():
    with pytest.raises(ValueError,match='missing_model_assets'):preflight('configs/bundle.example.json')
    assert doctor()['model_loaded'] is False

def test_escape(staged):
    _,bundle,_=staged;m=json.loads(bundle.read_text());m['assets'][0]['path']='../escape';bundle.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='unsafe'):preflight(bundle)


def test_runtime_version_divergence(staged,monkeypatch):
    """Metadata can disagree with the module that actually imports; preflight must not trust metadata alone."""
    _,bundle,_=staged;m=json.loads(bundle.read_text());m['dependencies']={'pytest':metadata_version('pytest')};bundle.write_text(json.dumps(m))
    assert preflight(bundle)['runtime_versions']['pytest']
    monkeypatch.setattr('lynceus.bundle.runtime_version',lambda package:'0.0.0-divergent')
    with pytest.raises(BundleError,match='runtime 0.0.0-divergent'):preflight(bundle)
    monkeypatch.setattr('lynceus.bundle.runtime_version',lambda package:None)
    with pytest.raises(BundleError,match='unimportable_dependency'):preflight(bundle)

def test_local_build_suffix_accepted(staged,monkeypatch):
    _,bundle,_=staged;m=json.loads(bundle.read_text());m['dependencies']={'pytest':metadata_version('pytest')};bundle.write_text(json.dumps(m))
    monkeypatch.setattr('lynceus.bundle.runtime_version',lambda package:metadata_version('pytest')+'+cu124')
    assert preflight(bundle)['verified_assets']==4

def test_adapter_requires_declared_dependencies(staged):
    from lynceus.adapters.owlv2 import Owlv2Adapter
    _,bundle,_=staged
    with pytest.raises(BundleError,match='undeclared_dependency'):Owlv2Adapter(bundle)

def test_doctor_reports_version_conflicts():
    report=doctor();assert isinstance(report['version_conflicts'],list);assert 'runtime_versions' in report


def test_a_bundle_names_its_adapter_and_is_checked_against_that_adapters_layout(tmp_path):
    """SAM 3 ships a processor config where OWLv2 ships a preprocessor config.

    The completeness check is per adapter, so a bundle cannot pass by carrying the other
    adapter's files.
    """
    from lynceus.bundle import ADAPTER_ASSETS
    assert ADAPTER_ASSETS['owlv2'] != ADAPTER_ASSETS['sam3']
    assert 'preprocessor_config.json' in ADAPTER_ASSETS['owlv2']
    assert 'processor_config.json' in ADAPTER_ASSETS['sam3']


def test_an_unknown_adapter_is_refused(tmp_path):
    import json
    from lynceus.bundle import BundleError, preflight
    manifest = tmp_path / 'bundle.json'
    manifest.write_text(json.dumps({'schema_version': 'lynceus.bundle/1.0', 'adapter': 'nonesuch',
                                    'qualification': 'unqualified', 'model_dir': 'model', 'assets': []}))
    with pytest.raises(BundleError, match='incompatible_bundle'):
        preflight(manifest)
