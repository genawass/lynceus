import json
import copy
import pytest
from lynceus.evaluation import evaluate_benchmark,match

@pytest.fixture
def benchmark():return json.load(open('examples/benchmark.json'))

def test_manual(benchmark):
    r=evaluate_benchmark(benchmark,100);m=r['metrics']
    for key,n,d in [('localized_instance_recall',2,3),('final_instance_precision',2,3),('accepted_annotation_precision',1,2),('useful_label_coverage',1,3),('classification_accuracy',1,2),('duplicate_rate',1,3)]:
        assert (m[key]['numerator'],m[key]['denominator'])==(n,d)
    assert r['qualification']['status']=='inconclusive';assert r['missing_or_failed_images']==1

def test_leakage(benchmark):
    extra=copy.deepcopy(benchmark['images'][0]);extra['id']='leak';extra['split']='development';benchmark['images'].append(extra)
    with pytest.raises(ValueError,match='split_leakage'):evaluate_benchmark(benchmark,10)

def test_empty_precision(benchmark):
    for im in benchmark['images']:im['predictions']=[]
    m=evaluate_benchmark(benchmark,10)['metrics'];assert m['accepted_annotation_precision']['value'] is None;assert m['useful_label_coverage']['value']==0

def test_invalid_prediction_not_dropped(benchmark):
    benchmark['images'][0]['predictions'][0]['bbox_xyxy']=[0,0,200,200]
    with pytest.raises(ValueError):evaluate_benchmark(benchmark,10)

def test_cardinality(monkeypatch):
    import lynceus.evaluation as module
    values={(0,0):1,(0,1):.5,(1,0):.5,(1,1):.49}
    monkeypatch.setattr(module,'iou',lambda a,b:values[(a,b)])
    assert len(match([{'bbox_xyxy':0},{'bbox_xyxy':1}],[{'bbox_xyxy':0},{'bbox_xyxy':1}],.5))==2

def scoped(benchmark,scope,reason='finite-category reference'):
    benchmark['category_scope']=scope;benchmark['category_scope_reason']=reason;return benchmark

def test_category_scope_requires_reason(benchmark):
    benchmark['category_scope']=['cup']
    with pytest.raises(ValueError,match='missing_category_scope_reason'):evaluate_benchmark(benchmark,10)
    benchmark['category_scope']=['not_an_ontology_id'];benchmark['category_scope_reason']='x'
    with pytest.raises(ValueError,match='invalid_category_scope'):evaluate_benchmark(benchmark,10)

def counts(metric):return metric['numerator'],metric['denominator']

def test_absent_scope_scores_every_instance(benchmark):
    from lynceus.policy import load_ontology
    base=evaluate_benchmark(copy.deepcopy(benchmark),10)
    wide=evaluate_benchmark(scoped(copy.deepcopy(benchmark),sorted(c['id'] for c in load_ontology()['classes']),'all classes'),10)
    assert counts(base['metrics']['final_instance_precision'])==counts(wide['metrics']['final_instance_precision'])
    assert base['category_scope'] is None and base['out_of_scope_predictions']==0

def test_out_of_scope_label_leaves_precision_denominator(benchmark):
    """A correct prediction of a class the reference never annotates is not a scoreable error."""
    image=benchmark['images'][0]
    image['predictions'].append({'id':'oos','bbox_xyxy':[0,0,20,20],'label':'building','status':'uncertain','kind':'instance','scene_layer':'physical'})
    unscoped=evaluate_benchmark(copy.deepcopy(benchmark),10)['metrics']['final_instance_precision']
    result=evaluate_benchmark(scoped(copy.deepcopy(benchmark),['cup','person'],'reference annotates cups and people only'),10)
    assert unscoped['denominator']==result['metrics']['final_instance_precision']['denominator']+1
    assert result['out_of_scope_predictions']==1
    assert 'category scope' in result['metrics']['final_instance_precision']['eligibility']

def test_unknown_object_stays_in_precision_denominator(benchmark):
    """Scope must not let abstention buy precision; gate G6 depends on it."""
    image=benchmark['images'][0]
    image['predictions'].append({'id':'spam','bbox_xyxy':[0,0,20,20],'label':'unknown_object','status':'uncertain','kind':'instance','scene_layer':'physical'})
    result=evaluate_benchmark(scoped(benchmark,['cup'],'cups only'),10)
    assert result['out_of_scope_predictions']==0
    assert result['metrics']['final_instance_precision']['denominator']==4

def test_scope_does_not_change_recall(benchmark):
    image=benchmark['images'][0]
    image['predictions'].append({'id':'oos','bbox_xyxy':[0,0,20,20],'label':'building','status':'uncertain','kind':'instance','scene_layer':'physical'})
    base=evaluate_benchmark(copy.deepcopy(benchmark),10)['metrics']['localized_instance_recall']
    result=evaluate_benchmark(scoped(copy.deepcopy(benchmark),['cup'],'cups only'),10)['metrics']['localized_instance_recall']
    assert counts(base)==counts(result)
