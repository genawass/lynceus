import math
from typing import Literal
from collections import Counter
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from .geometry import validate_box
from .policy import available_ontologies, load_policy
from .artifacts import canonical, digest, verify_artifacts, safe_path

State=Literal['supported','unresolved','not_assessed']
class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
class Artifact(Strict):
    path:str
    format:str
    sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
class Identity(Strict):
    id:str
    version:str
    sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    definition:dict
class ImageInfo(Strict):
    source_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    pixel_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    original_size:tuple[int,int]
    normalized_size:tuple[int,int]
    orientation:int=Field(ge=1,le=8)
    orientation_transform:list[list[float]]
    color_conversion:Literal['RGB']
    alpha_conversion:Literal['none','white_composite']
class Label(Strict):
    id:str
    name:str
class Uncertainty(Strict):
    existence:State
    count:State
    boundary:State
    class_:State=Field(alias='class')
    granularity:State
    reasons:list[str]
class Relation(Strict):
    type:Literal['part_of','worn_by','carried_by','supported_by','depicted_on']
    target_id:str
    evidence_refs:list[str]=Field(min_length=1)
class CalibratedEstimate(Strict):
    """A probability for one exactly defined event, under a recorded external calibration.

    The event text must name something that can be wrong, and the estimate is only meaningful
    under the population it was fitted on -- so both travel with the number, along with the
    artifact that produced it.
    """
    event:str=Field(min_length=8)
    probability:float=Field(ge=0.0,le=1.0)
    calibration_ref:str
    population:str=Field(min_length=1)
    localization_threshold:float|None=Field(default=None,ge=0.0,le=1.0)


class Object(Strict):
    id:str
    kind:Literal['instance','part','group','stuff']
    status:Literal['accepted','uncertain']
    scene_layer:Literal['physical','reflected','depicted']
    bbox_xyxy:tuple[float,float,float,float]
    mask_ref:str|None=None
    alternative_boxes:list[tuple[float,float,float,float]]=Field(default_factory=list)
    label:Label
    label_alternatives:list[Label]=Field(default_factory=list)
    uncertainty:Uncertainty
    evidence_refs:list[str]=Field(min_length=1)
    relations:list[Relation]=Field(default_factory=list)
    occluded:bool|None=None
    truncated:bool|None=None
    flag_evidence_refs:list[str]=Field(default_factory=list)
    count:int|tuple[int,int]|None=None
    calibrated_estimates:list[CalibratedEstimate]=Field(default_factory=list)
class Region(Strict):
    kind:Literal['unsearched','unresolved','policy_excluded']
    bbox_xyxy:tuple[float,float,float,float]
    reason:str=Field(min_length=1)
    evidence_refs:list[str]=Field(min_length=1)
class Qualification(Strict):
    id:str|None=None
    domain:str
    status:Literal['unqualified','eligible','ineligible']
class Execution(Strict):
    status:Literal['completed','interrupted','failed']
    reason:str
    failure_refs:list[str]
class Search(Strict):
    profile_id:str
    status:Literal['saturated','profile_complete','budget_exhausted','interrupted','failed']
    mandatory_planned:int=Field(ge=0)
    mandatory_completed:int=Field(ge=0)
    mandatory_failed:int=Field(ge=0)
    stable_rounds:int=Field(ge=0)
    coverage_ref:str
    stop_rule:dict
class Annotation(Strict):
    schema_version:Literal['lynceus.annotation/1.0']
    run_id:str
    image:ImageInfo
    policy:Identity
    ontology:Identity
    manifest_ref:str
    qualification:Qualification
    execution:Execution
    search:Search
    objects:list[Object]
    regions:list[Region]
    summary:dict
    artifacts:dict[str,Artifact]


def identity(definition):
    return {'id':definition['id'],'version':definition['version'],'sha256':digest(canonical(definition)),'definition':definition}


# Recomputable from objects/regions alone, so the validator can reject a doctored summary.
COVERAGE_FIELDS=('planned_jobs','completed_jobs','failed_jobs','route_covered_area_fraction')
RESOURCE_FIELDS=('wall_seconds','peak_rss_bytes','artifact_bytes','model_calls')
CALIBRATION_FIELDS=('status','reason')
# A score becomes a probability only under a calibration fitted elsewhere. Absent that, the run
# must say so rather than quietly emit nothing, so consumers can tell "no estimate" from
# "estimate of zero".
CALIBRATION_STATES=('applied','inapplicable')


def summary_counts(objects,regions):
    return {'by_status':dict(Counter(o['status'] for o in objects)),'by_kind':dict(Counter(o['kind'] for o in objects)),'by_class':dict(Counter(o['label']['id'] for o in objects)),'unresolved_objects':sum(o['status']=='uncertain' for o in objects),'regions':len(regions),'unsearched_regions':sum(r['kind']=='unsearched' for r in regions)}


def summarize(objects,regions,coverage,resources,calibration):
    return {**summary_counts(objects,regions),'coverage':dict(coverage),'resources':dict(resources),
            'calibration':dict(calibration)}


def annotation_schema():
    return Annotation.model_json_schema(by_alias=True)


def validate_annotation(payload,artifact_root=None):
    data=Annotation.model_validate(payload).model_dump(by_alias=True,mode='json')
    w,h=data['image']['normalized_size']
    if min(w,h,*data['image']['original_size'])<=0: raise ValueError('invalid_image_size')
    matrix=np.asarray(data['image']['orientation_transform'])
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12: raise ValueError('invalid_orientation_transform')
    if data['policy']!=identity(load_policy()): raise ValueError('incompatible_policy')
    # An annotation names its ontology; it must be one this build actually ships, byte for byte.
    registry=available_ontologies(); declared=data['ontology']['id']
    if declared not in registry: raise ValueError(f'unknown_ontology: {declared}')
    if data['ontology']!=identity(registry[declared]): raise ValueError('incompatible_ontology')
    calibration=data['summary'].get('calibration') or {}
    if calibration.get('status') not in CALIBRATION_STATES: raise ValueError('invalid_calibration_status')
    if calibration['status']=='inapplicable' and any(o['calibrated_estimates'] for o in data['objects']):
        raise ValueError('calibrated_estimate_while_calibration_inapplicable')
    labels={x['id']:x for x in data['ontology']['definition']['classes']}
    ids=[o['id'] for o in data['objects']]
    if len(set(ids))!=len(ids): raise ValueError('duplicate_object_id')
    artifacts=data['artifacts']; paths=[a['path'] for a in artifacts.values()]
    if len(paths)!=len(set(paths)): raise ValueError('duplicate_artifact_path')
    for path in paths: safe_path(artifact_root or '.',path)
    def ref(value):
        if value not in paths: raise ValueError(f'unknown_artifact_reference: {value}')
    ref(data['manifest_ref']); ref(data['search']['coverage_ref'])
    edges={i:[] for i in ids}
    for o in data['objects']:
        for box in [o['bbox_xyxy']]+o['alternative_boxes']: validate_box(box,w,h)
        if o['mask_ref']: raise ValueError('mask_validation_not_implemented')
        for label in [o['label']]+o['label_alternatives']:
            if label['id'] not in labels or labels[label['id']]['name']!=label['name']: raise ValueError('invalid_ontology_label')
        states=[o['uncertainty'][k] for k in ['existence','count','boundary','class','granularity']]
        if o['status']=='accepted' and (any(s!='supported' for s in states) or not labels[o['label']['id']]['useful']): raise ValueError('invalid_acceptance')
        if o['label']['id']=='unknown_object' and o['uncertainty']['class']=='supported': raise ValueError('unknown_class_supported')
        for estimate in o['calibrated_estimates']:
            ref(estimate['calibration_ref'])
            if not math.isfinite(estimate['probability']): raise ValueError('invalid_calibrated_probability')
            if calibration.get('status')!='applied': raise ValueError('calibrated_estimate_without_applied_calibration')
        if (o['occluded'] is not None or o['truncated'] is not None) and not o['flag_evidence_refs']: raise ValueError('unsupported_flag')
        if o['kind']!='group' and o['count'] is not None: raise ValueError('count_only_for_groups')
        if o['count'] is not None:
            counts=o['count'] if isinstance(o['count'],list) else [o['count']]
            if min(counts)<1 or counts!=sorted(counts) or o['uncertainty']['count']!='supported': raise ValueError('unsupported_group_count')
        if o['kind']=='group' and o['count'] is None and o['uncertainty']['count']=='supported': raise ValueError('unresolved_group_count')
        for ev in o['evidence_refs']+o['flag_evidence_refs']: ref(ev)
        for rel in o['relations']:
            if rel['target_id'] not in ids or rel['target_id']==o['id']: raise ValueError('invalid_relation_target')
            for ev in rel['evidence_refs']: ref(ev)
            if rel['type']=='part_of': edges[o['id']].append(rel['target_id'])
        if o['kind']=='part' and not edges[o['id']]: raise ValueError('missing_part_of')
    def visit(node,active,done):
        if node in active: raise ValueError('cyclic_part_of')
        if node in done:return
        active.add(node)
        for child in edges[node]: visit(child,active,done)
        active.remove(node);done.add(node)
    done=set()
    for node in ids: visit(node,set(),done)
    for r in data['regions']:
        validate_box(r['bbox_xyxy'],w,h)
        for ev in r['evidence_refs']: ref(ev)
    s=data['search']; e=data['execution']
    for ev in e['failure_refs']: ref(ev)
    if s['mandatory_completed']+s['mandatory_failed']>s['mandatory_planned']: raise ValueError('invalid_job_counts')
    if s['status'] in ['saturated','profile_complete'] and (e['status']!='completed' or s['mandatory_completed']!=s['mandatory_planned'] or s['mandatory_failed']): raise ValueError('incomplete_search')
    if s['status']=='saturated':
        if s['profile_id']=='owlv2-single-pass-v1' or s['stable_rounds']<s['stop_rule'].get('required_stable_rounds',1) or any(r['kind']=='unsearched' for r in data['regions']): raise ValueError('false_saturation')
        raise ValueError('saturation_profile_not_implemented')
    if e['status'] in ['failed','interrupted'] and (s['status']!=e['status'] or not e['failure_refs']): raise ValueError('invalid_failure_state')
    if data['qualification']['status']=='eligible' or data['qualification']['id'] is not None: raise ValueError('qualification_not_implemented')
    summary=data['summary']
    # Coverage and resource measurements cannot be recomputed here, so require them explicitly
    # and validate everything that can be recomputed.
    for key,fields in [('coverage',COVERAGE_FIELDS),('resources',RESOURCE_FIELDS),('calibration',CALIBRATION_FIELDS)]:
        if not isinstance(summary.get(key),dict): raise ValueError(f'summary_missing_{key}')
        missing=[f for f in fields if f not in summary[key]]
        if missing: raise ValueError(f'summary_missing_{key}_field: '+','.join(missing))
    if {k:v for k,v in summary.items() if k not in ('coverage','resources','calibration')}!=summary_counts(data['objects'],data['regions']): raise ValueError('summary_mismatch')
    if artifact_root is not None: verify_artifacts(artifact_root,artifacts)
    return data
