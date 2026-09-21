import copy
import pytest
from lynceus.contracts import validate_annotation,annotation_schema

def test_valid(completed):
    result,root=completed;assert validate_annotation(result,root)['qualification']['status']=='unqualified';assert '$defs' in annotation_schema()

@pytest.mark.parametrize('change',['accept','unknown','nan','outside','relation','saturation','summary','escape','truth','flags','calibration','no_coverage','no_resources','partial_resources'])
def test_reject(completed,change):
    result,root=completed;result=copy.deepcopy(result);o=result['objects'][0]
    if change=='accept':o['status']='accepted'
    if change=='unknown':o['label']['id']='unregistered'
    if change=='nan':o['bbox_xyxy'][0]=float('nan')
    if change=='outside':o['bbox_xyxy'][2]=1000
    if change=='relation':o['relations']=[{'type':'part_of','target_id':'missing','evidence_refs':['events.jsonl']}]
    if change=='saturation':result['search']['status']='saturated'
    if change=='summary':result['summary']['regions']=42
    if change=='escape':result['artifacts']['report']['path']='../outside'
    if change=='truth':result['is_ground_truth']=True
    if change=='flags':o['occluded']=False
    if change=='calibration':o['calibrated_estimates']=[{'probability_correct':.99}]
    if change=='no_coverage':result['summary'].pop('coverage')
    if change=='no_resources':result['summary'].pop('resources')
    if change=='partial_resources':result['summary']['resources'].pop('peak_rss_bytes')
    with pytest.raises(ValueError):validate_annotation(result,root)

def test_tamper(completed):
    result,root=completed;(root/'events.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='hash_mismatch'):validate_annotation(result,root)


def test_summary_records_coverage_and_resources(completed):
    result,_=completed;summary=result['summary']
    assert summary['coverage']['planned_jobs']==1 and summary['coverage']['completed_jobs']==1
    assert summary['resources']['model_calls']==1 and summary['resources']['peak_rss_bytes']>0
    assert summary['resources']['artifact_bytes']>0 and summary['resources']['wall_seconds']>=0


def test_annotation_must_name_a_shipped_ontology(completed):
    import copy as _copy
    result, root = completed
    result = _copy.deepcopy(result)
    result['ontology']['id'] = 'invented-ontology-v9'
    with pytest.raises(ValueError, match='unknown_ontology'):
        validate_annotation(result, root)

def test_ontology_definition_must_match_the_shipped_bytes(completed):
    import copy as _copy
    result, root = completed
    result = _copy.deepcopy(result)
    result['ontology']['definition']['classes'].append(
        {'id': 'smuggled', 'name': 'smuggled', 'synonyms': [], 'parent': 'entity', 'useful': True})
    with pytest.raises(ValueError, match='incompatible_ontology'):
        validate_annotation(result, root)
