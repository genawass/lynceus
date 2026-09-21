import importlib
import importlib.metadata
import platform
from pathlib import Path
import json
from .artifacts import safe_path,digest

class BundleError(ValueError):pass

# Distribution name -> importable module. Metadata alone can disagree with the module that
# actually imports (duplicate or stale dist-info), so both are reported and compared.
RUNTIME_MODULES={'pydantic':'pydantic','Pillow':'PIL','numpy':'numpy','scipy':'scipy','pytest':'pytest','torch':'torch','transformers':'transformers'}
# Imported eagerly by this package anyway, so doctor() can compare them without extra cost.
LIGHT_PACKAGES=['pydantic','Pillow','numpy','scipy']


def metadata_version(package):
    try:return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:return None


def runtime_version(package):
    """Version reported by the module that actually imports, or None when it cannot be imported."""
    module_name=RUNTIME_MODULES.get(package)
    if module_name is None:return None
    try:module=importlib.import_module(module_name)
    except Exception:return None
    return getattr(module,'__version__',None)


def public_version(version):
    """Drop local build suffixes such as torch's '+cu124' before comparison."""
    return version.split('+')[0] if version else version


def doctor(bundle=None):
    versions={name:metadata_version(name) for name in RUNTIME_MODULES}
    runtime={name:runtime_version(name) for name in LIGHT_PACKAGES}
    conflicts=sorted(n for n,v in runtime.items() if v and versions[n] and public_version(v)!=public_version(versions[n]))
    report={'offline':True,'python':platform.python_version(),'platform':platform.platform(),'versions':versions,'runtime_versions':runtime,'version_conflicts':conflicts,'qualification':'unqualified','model_loaded':False,'bundle':None}
    if bundle: report['bundle']=preflight(bundle)
    return report


def check_dependencies(dependencies):
    """Declared dependencies must match both the installed metadata and the imported module."""
    runtime={}
    for package,expected in dependencies.items():
        installed=metadata_version(package)
        if installed is None:raise BundleError('missing_dependency: '+package)
        if installed!=expected:raise BundleError(f'incompatible_dependency: {package} metadata {installed} but manifest {expected}')
        actual=runtime_version(package)
        if actual is None:raise BundleError('unimportable_dependency: '+package)
        if public_version(actual)!=public_version(expected):raise BundleError(f'incompatible_dependency: {package} runtime {actual} but manifest {expected}')
        runtime[package]=actual
    return runtime


# A bundle names the adapter that will load it, and each adapter has its own idea of what a
# complete checkpoint looks like: OWLv2 ships a preprocessor config, SAM 3 a processor config.
# Declaring the set per adapter keeps "complete" checkable rather than assumed.
ADAPTER_ASSETS={'owlv2':{'config.json','preprocessor_config.json','tokenizer_config.json'},
                'sam3':{'config.json','processor_config.json','tokenizer_config.json'}}


def preflight(path):
    path=Path(path)
    if not path.is_file():raise BundleError('missing_bundle_manifest')
    manifest=json.loads(path.read_text());root=path.parent
    if manifest.get('schema_version')!='lynceus.bundle/1.0' or manifest.get('adapter') not in ADAPTER_ASSETS:raise BundleError('incompatible_bundle')
    if manifest.get('qualification')!='unqualified':raise BundleError('unsupported_qualification')
    assets=manifest.get('assets',[])
    if not assets:raise BundleError('missing_model_assets')
    for a in assets:
        p=safe_path(root,a['path'])
        if not p.is_file():raise BundleError('missing_asset: '+a['path'])
        if digest(p.read_bytes())!=a['sha256']:raise BundleError('asset_hash_mismatch: '+a['path'])
    model_dir=safe_path(root,manifest['model_dir'])
    required=ADAPTER_ASSETS[manifest['adapter']]
    listed={safe_path(root,a['path']) for a in assets}
    for name in required:
        if model_dir/name not in listed:raise BundleError('missing_asset: '+name)
    if not any(p.suffix in ['.safetensors','.bin'] for p in listed):raise BundleError('missing_model_weights')
    actual={p.resolve() for p in model_dir.rglob('*') if p.is_file()}
    if actual-listed:raise BundleError('unmanifested_model_assets')
    runtime=check_dependencies(manifest.get('dependencies',{}))
    capabilities=['single_pass_uncertain_proposals'] if manifest['adapter']=='owlv2' else ['box_prompted_boundary_refinement']
    return {'manifest':manifest,'manifest_sha256':digest(path.read_bytes()),'model_dir':str(model_dir),'adapter':manifest['adapter'],'capabilities':capabilities,'verified_assets':len(assets),'runtime_versions':runtime}
