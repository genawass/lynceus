import pytest
from lynceus.pipeline import annotate,resume
from conftest import FakeAdapter

def test_resume(completed):
    result,root=completed;assert resume(root)['run_id']==result['run_id']
    (root/'annotation.json').write_text('{}')
    with pytest.raises(ValueError,match='annotation_hash'):resume(root)

def test_interrupted(staged):
    class Interrupted:
        def predict(self,image):raise KeyboardInterrupt()
    image,bundle,run=staged;r=annotate(image,bundle,run,adapter=Interrupted());assert r['execution']['status']=='interrupted'
    new=resume(run,run.parent/'retry',adapter=FakeAdapter());assert new['execution']['status']=='completed';assert new['run_id']!=r['run_id']

def test_changed_input(completed,staged):
    result,root=completed;image,_,_=staged;image.write_bytes(b'changed')
    with pytest.raises((ValueError,OSError)):resume(root)

def test_missing_weights(staged):
    image,_,run=staged
    with pytest.raises(ValueError,match='missing_model_assets'):annotate(image,'configs/bundle.example.json',run)
    assert not run.exists()


def test_overlay_is_not_the_source_image(completed,staged):
    result,root=completed;image,_,_=staged
    assert (root/'overlay.png').read_bytes()!=image.read_bytes()
    from PIL import Image
    with Image.open(root/'overlay.png') as rendered, Image.open(image) as source:
        assert rendered.size==source.size
        assert rendered.convert('RGB').tobytes()!=source.convert('RGB').tobytes()

def test_report_explains_the_run(completed):
    result,root=completed;html=(root/'report.html').read_text()
    for fragment in ['Stop reason','Unknown classes','Unresolved boundaries','Excluded policy layers','obj-0']:
        assert fragment in html
    assert 'http://' not in html and 'https://' not in html

def test_events_record_resources(completed):
    import json as _json
    result,root=completed
    events=[_json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    assert all('resources' in e and e['resources']['peak_rss_bytes']>0 for e in events)


def test_tile_plan_covers_image_at_every_level():
    from lynceus.geometry import plan_tiles,covered_fraction
    for size in [(1360,765),(1024,1024),(765,1360),(4000,3000),(100,60)]:
        for levels in (0,1,2,3):
            jobs=plan_tiles(*size,levels=levels)
            assert jobs[0]['id']=='full' and jobs[0]['bbox_xyxy']==[0,0,float(size[0]),float(size[1])]
            assert covered_fraction(jobs,*size)==1.0, (size,levels)

def test_grid_follows_image_aspect_not_a_square_grid():
    """A square grid inherits the image aspect, and the processor pads every view to a square."""
    from lynceus.geometry import grid_shape,plan_tiles
    assert grid_shape(1024,1024,1)==(2,2) and grid_shape(1024,1024,2)==(3,3)
    assert grid_shape(1360,765,1)==(3,2)          # 16:9 gets 3x2, not 3x3
    assert grid_shape(765,1360,1)==(2,3)          # portrait is the transpose
    def waste(w,h,level):
        tile=[j for j in plan_tiles(w,h,levels=level) if j['level']==level][0]['bbox_xyxy']
        tw,th=tile[2]-tile[0],tile[3]-tile[1]
        return 1-(tw*th)/max(tw,th)**2
    assert waste(1360,765,1)<0.25                 # was 0.437 under a square grid
    assert waste(1024,1024,1)==0                  # square images unaffected

def test_overlap_parameter_is_the_actual_overlap_fraction():
    from lynceus.geometry import plan_tiles
    for requested in (0.1,0.25,0.4):
        row=[j for j in plan_tiles(1360,765,levels=1,overlap=requested) if j['level']==1][:2]
        (ax1,_,ax2,_),(bx1,_,_,_)=row[0]['bbox_xyxy'],row[1]['bbox_xyxy']
        assert abs((ax2-bx1)/(ax2-ax1)-requested)<0.02, requested

def test_tiled_run_merges_repeated_views(staged):
    """One physical object visible from several views becomes one instance, not one per view."""
    image,bundle,run=staged
    OBJECT=(10,10,34,34)   # fixed source-pixel extent, seen by whichever views contain it
    class SeesOneObject:
        def __init__(self):self.origin=(0,0)
        def predict(self,view):
            # The pipeline crops views in planned order, so recover this view's origin by size.
            x1,y1=self.origin
            box=[OBJECT[0]-x1,OBJECT[1]-y1,OBJECT[2]-x1,OBJECT[3]-y1]
            clipped=[max(0,min(view.width,box[0])),max(0,min(view.height,box[1])),
                     max(0,min(view.width,box[2])),max(0,min(view.height,box[3]))]
            if clipped[2]-clipped[0]<2 or clipped[3]-clipped[1]<2:return []
            return [{'bbox_xyxy':clipped,'label':'cup','model_score':.9,'crop_edge':clipped!=box}]
    from lynceus.geometry import plan_tiles
    adapter=SeesOneObject()
    jobs=iter(plan_tiles(100,60,levels=1))
    original=adapter.predict
    def tracked(view):
        adapter.origin=tuple(int(v) for v in next(jobs)['bbox_xyxy'][:2])
        return original(view)
    adapter.predict=tracked
    result=annotate(image,bundle,run,adapter=adapter,tile_levels=1)
    assert result['search']['profile_id']=='owlv2-tiled-L1-v1'
    assert result['search']['mandatory_planned']==7 and result['search']['mandatory_completed']==7
    assert result['summary']['coverage']['route_covered_area_fraction']['owlv2']==1.0
    # several views saw it; one instance survives and the rest remain inspectable as evidence
    import json as _json
    lines=[_json.loads(l) for l in (run/'evidence/candidates.jsonl').read_text().splitlines()]
    assert len(lines)>1, 'more than one view should have seen the object'
    assert len(result['objects'])==1
    assert {c['disposition'] for c in lines}=={'retained','merged'}

def test_cross_label_overlap_is_never_suppressed(staged):
    """A bag inside a person's box is a separate instance; geometry alone must not remove it."""
    image,bundle,run=staged
    class TwoLabels:
        def predict(self,view):
            return [{'bbox_xyxy':[5,5,90,55],'label':'person','model_score':.9,'crop_edge':False},
                    {'bbox_xyxy':[10,10,40,40],'label':'backpack','model_score':.8,'crop_edge':False}]
    result=annotate(image,bundle,run,adapter=TwoLabels(),tile_levels=0)
    assert sorted(o['label']['id'] for o in result['objects'])==['backpack','person']

def test_failed_view_blocks_complete_scan_and_records_unsearched(staged):
    image,bundle,run=staged
    class FlakyTiles:
        def predict(self,view):
            if view.size!=(100,60): raise RuntimeError('tile route failed')
            return [{'bbox_xyxy':[1,1,20,20],'label':'cup','model_score':.5,'crop_edge':False}]
    result=annotate(image,bundle,run,adapter=FlakyTiles(),tile_levels=1)
    assert result['execution']['status']=='completed'
    assert result['search']['status']=='failed' and result['search']['mandatory_failed']==6
    assert result['summary']['unsearched_regions']==6
    assert result['summary']['coverage']['route_covered_area_fraction']['owlv2']==1.0

def test_boundary_unresolved_when_views_disagree(staged):
    image,bundle,run=staged
    class Drifting:
        def __init__(self):self.n=0
        def predict(self,view):
            self.n+=1
            return [{'bbox_xyxy':[1,1,30+self.n,30],'label':'cup','model_score':1.0/self.n,'crop_edge':False}]
    result=annotate(image,bundle,run,adapter=Drifting(),tile_levels=1)
    merged=[o for o in result['objects'] if o['alternative_boxes']]
    assert merged and merged[0]['uncertainty']['boundary']=='unresolved'
    assert 'multi_view_boundary_disagreement' in merged[0]['uncertainty']['reasons']


def test_crowd_neighbours_are_not_merged_by_containment(staged):
    """IoS alone collapses a small box inside a larger one, which is ordinary crowd geometry."""
    from lynceus.pipeline import merge_observations
    big={'id':'a','label':'person','model_score':.9,'view_edge':False,'bbox_xyxy':[0,0,40,80]}
    inside={'id':'b','label':'person','model_score':.8,'view_edge':False,'bbox_xyxy':[10,10,30,40]}
    groups,_=merge_observations([big,inside],ios_threshold=.6,iou_threshold=.5)
    assert len(groups)==2, 'unflagged containment must not merge'
    truncated={**inside,'view_edge':True}
    groups,reasons=merge_observations([big,truncated],ios_threshold=.6,iou_threshold=.5)
    assert len(groups)==1 and 'view_edge_truncation' in reasons.values()

def test_plain_duplicates_still_merge_without_edge_flag(staged):
    from lynceus.pipeline import merge_observations
    a={'id':'a','label':'car','model_score':.9,'view_edge':False,'bbox_xyxy':[0,0,50,50]}
    b={'id':'b','label':'car','model_score':.8,'view_edge':False,'bbox_xyxy':[3,3,52,52]}
    groups,reasons=merge_observations([a,b],ios_threshold=.6,iou_threshold=.5)
    assert len(groups)==1 and 'iou_agreement' in reasons.values()


def test_acceptance_is_off_by_default(staged):
    """Without the rule nothing may be accepted, and every dimension stays unassessed."""
    image, bundle, run = staged
    result = annotate(image, bundle, run, adapter=FakeAdapter())
    assert {o['status'] for o in result['objects']} == {'uncertain'}
    assert result['objects'][0]['uncertainty']['existence'] == 'not_assessed'

def test_strong_single_view_detection_remains_uncertain(staged):
    class Strong:
        def predict(self, view):
            return [{'bbox_xyxy': [5, 5, 40, 40], 'label': 'cup', 'model_score': .95, 'crop_edge': False}]
    image, bundle, run = staged
    result = annotate(image, bundle, run, adapter=Strong())
    obj = result['objects'][0]
    assert obj['status'] == 'uncertain'
    assert all(obj['uncertainty'][d] == 'not_assessed'
               for d in ('existence', 'count', 'boundary', 'class', 'granularity'))
    assert 'unverified_model_proposal' in obj['uncertainty']['reasons']

def test_weak_score_stays_uncertain(staged):
    class Weak:
        def predict(self, view):
            return [{'bbox_xyxy': [5, 5, 40, 40], 'label': 'cup', 'model_score': .2, 'crop_edge': False}]
    image, bundle, run = staged
    obj = annotate(image, bundle, run, adapter=Weak())['objects'][0]
    assert obj['status'] == 'uncertain' and obj['uncertainty']['existence'] == 'not_assessed'

def test_unknown_class_can_never_be_accepted(staged):
    class Unknown:
        def predict(self, view):
            return [{'bbox_xyxy': [5, 5, 40, 40], 'label': 'unknown_object', 'model_score': .99, 'crop_edge': False}]
    image, bundle, run = staged
    obj = annotate(image, bundle, run, adapter=Unknown())['objects'][0]
    assert obj['status'] == 'uncertain' and obj['uncertainty']['class'] == 'not_assessed'
    assert obj['bbox_xyxy'] == [5, 5, 40, 40], 'geometry must survive an unresolved class'

def test_containment_withholds_granularity_support(staged):
    """A small box inside a larger one may be a part, so its granularity is not supported."""
    class Nested:
        def predict(self, view):
            return [{'bbox_xyxy': [2, 2, 95, 55], 'label': 'person', 'model_score': .95, 'crop_edge': False},
                    {'bbox_xyxy': [10, 10, 30, 30], 'label': 'bag', 'model_score': .9, 'crop_edge': False}]
    image, bundle, run = staged
    objects = annotate(image, bundle, run, adapter=Nested())['objects']
    inner = next(o for o in objects if o['label']['id'] == 'bag')
    outer = next(o for o in objects if o['label']['id'] == 'person')
    assert inner['uncertainty']['granularity'] == 'not_assessed' and inner['status'] == 'uncertain'
    assert outer['uncertainty']['granularity'] == 'not_assessed' and outer['status'] == 'uncertain'

def test_view_edge_truncation_blocks_acceptance(staged):
    class Truncated:
        def predict(self, view):
            return [{'bbox_xyxy': [0, 5, 40, 40], 'label': 'cup', 'model_score': .95, 'crop_edge': True}]
    image, bundle, run = staged
    obj = annotate(image, bundle, run, adapter=Truncated())['objects'][0]
    assert obj['uncertainty']['boundary'] == 'unresolved' and obj['status'] == 'uncertain'


def test_merging_is_idempotent_without_fusion(staged):
    """Re-merging cannot help unless geometry changes: the same boxes produce the same groups."""
    from lynceus.pipeline import consolidate
    obs = [{'id': f'o{i}', 'label': 'car', 'model_score': .9 - i * .1, 'view_edge': False,
            'bbox_xyxy': [i * 40, 0, i * 40 + 30, 30]} for i in range(4)]
    first = consolidate(obs, .6, .5, rounds=1, fuse=False)
    for rounds in (2, 5):
        later = consolidate(obs, .6, .5, rounds=rounds, fuse=False)
        assert [r['bbox_xyxy'] for r in later[1]] == [r['bbox_xyxy'] for r in first[1]]

def test_fusion_interpolates_and_never_extends_beyond_observations(staged):
    """A union would assert extent no view supported; a weighted mean stays inside."""
    from lynceus.pipeline import consolidate
    obs = [{'id': 'a', 'label': 'car', 'model_score': .9, 'view_edge': False, 'bbox_xyxy': [0, 0, 40, 40]},
           {'id': 'b', 'label': 'car', 'model_score': .1, 'view_edge': False, 'bbox_xyxy': [4, 4, 44, 44]}]
    fused = consolidate(obs, .6, .5, rounds=1, fuse=True)[1][0]['bbox_xyxy']
    for k in range(4):
        low, high = sorted((obs[0]['bbox_xyxy'][k], obs[1]['bbox_xyxy'][k]))
        assert low <= fused[k] <= high
    assert fused[0] < 2, 'the higher-scoring observation should dominate'


def test_two_views_that_disagree_on_class_merge_under_what_they_jointly_support():
    """Neither view is uncertain, so per-observation naming cannot reach this case.

    One view says `car`, another says `van`, and they describe the same instance. The claim both
    support is `vehicle`, and the surviving record makes that claim while keeping both readings.
    """
    from lynceus.pipeline import merge_observations
    from lynceus.policy import load_ontology
    ontology = load_ontology('aerial-traffic-strict-v1')
    observations = [
        {'id': 'a', 'label': 'car', 'bbox_xyxy': [10, 10, 30, 30], 'model_score': 0.9, 'view_edge': False},
        {'id': 'b', 'label': 'van', 'bbox_xyxy': [11, 11, 31, 31], 'model_score': 0.5, 'view_edge': False}]
    groups, reasons = merge_observations(observations, 0.6, 0.5, ontology)
    assert len(groups) == 1
    assert observations[0]['label'] == 'vehicle'
    assert sorted(observations[0]['label_alternatives']) == ['van']
    assert 'broader_class_across_views' in reasons[1]


def test_views_whose_only_common_ancestor_is_useless_stay_separate():
    from lynceus.pipeline import merge_observations
    from lynceus.policy import load_ontology
    ontology = load_ontology('aerial-traffic-strict-v1')
    # `bicycle` and `person` share only `entity`, which is not a useful class, so one box cannot
    # speak for both without asserting something neither view supports.
    observations = [
        {'id': 'a', 'label': 'bicycle', 'bbox_xyxy': [10, 10, 30, 30], 'model_score': 0.9, 'view_edge': False},
        {'id': 'b', 'label': 'person', 'bbox_xyxy': [11, 11, 31, 31], 'model_score': 0.5, 'view_edge': False}]
    groups, _ = merge_observations(observations, 0.6, 0.5, ontology)
    assert len(groups) == 2


def test_unknown_object_never_merges_into_a_class():
    from lynceus.pipeline import merge_observations
    from lynceus.policy import load_ontology
    ontology = load_ontology('aerial-traffic-strict-v1')
    # It asserts that no useful class is supported; folding it into one manufactures a claim.
    observations = [
        {'id': 'a', 'label': 'vehicle', 'bbox_xyxy': [10, 10, 30, 30], 'model_score': 0.9, 'view_edge': False},
        {'id': 'b', 'label': 'unknown_object', 'bbox_xyxy': [11, 11, 31, 31], 'model_score': 0.5, 'view_edge': False}]
    groups, _ = merge_observations(observations, 0.6, 0.5, ontology)
    assert len(groups) == 2


def test_without_an_ontology_only_identical_labels_merge():
    from lynceus.pipeline import merge_observations
    observations = [
        {'id': 'a', 'label': 'car', 'bbox_xyxy': [10, 10, 30, 30], 'model_score': 0.9, 'view_edge': False},
        {'id': 'b', 'label': 'van', 'bbox_xyxy': [11, 11, 31, 31], 'model_score': 0.5, 'view_edge': False}]
    groups, _ = merge_observations(observations, 0.6, 0.5, None)
    assert len(groups) == 2
