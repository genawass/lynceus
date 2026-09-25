import argparse
import json
import sys
from pathlib import Path
from .bundle import doctor,BundleError
from .policy import load_policy,load_ontology,available_ontologies
from .vocabulary import available_vocabularies
from .contracts import annotation_schema,validate_annotation
from .evaluation import evaluate_benchmark
from .pipeline import annotate,resume


def acceptance_rules(args):
    """Frozen acceptance thresholds, or None to leave every object uncertain."""
    if not args.accept:return None
    given={'score':args.accept_score,'min_views':args.accept_min_views,
           'boundary_iou':args.accept_boundary_iou,'granularity_ios':args.accept_granularity_ios,
           'require_verification':args.require_verification or None}
    return {k:v for k,v in given.items() if v is not None}


def build_parser():
    parser=argparse.ArgumentParser(prog='lynceus')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('doctor');p.add_argument('--offline',action='store_true');p.add_argument('--bundle')
    p=sub.add_parser('policy');p.add_argument('--ontology')
    sub.add_parser('schema')
    p=sub.add_parser('validate');p.add_argument('annotation');p.add_argument('--artifact-root')
    p=sub.add_parser('annotate');p.add_argument('image');p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tile-levels',type=int,default=0,help='0 = single full-image pass; k adds aspect-matched overlapping source-resolution tile grids')
    p.add_argument('--tile-overlap',type=float,default=0.25,help='linear overlap fraction shared by adjacent tiles')
    p.add_argument('--merge-ios',type=float,default=0.6,help='intersection-over-smaller threshold, trusted only across a view border')
    p.add_argument('--merge-iou',type=float,default=0.5)
    p.add_argument('--device');p.add_argument('--threshold',type=float,default=0.1)
    p.add_argument('--ontology',help='ontology id; see `lynceus policy`')
    p.add_argument('--name-ratio',type=float,default=None,
                   help='runner-up counts as competing at this fraction of the winner; 1.0 disables broader-class fallback')
    p.add_argument('--vocabulary',help='per-model prompt vocabulary id; see `lynceus policy`')
    p.add_argument('--merge-labels',choices=['exact','common-ancestor'],default='common-ancestor',help='common-ancestor merges two views that disagree on class under the broader class they jointly support')
    p.add_argument('--merge-rounds',type=int,default=1,help='bounded merge/fuse rounds; only useful with --fuse, since merging is idempotent')
    p.add_argument('--fuse',action='store_true',help='score-weighted consensus box per cluster; measured harmful, off by default')
    p.add_argument('--refine-bundle',help='bundle holding a SAM 3 checkpoint; enables boundary refinement against pixels')
    p.add_argument('--refine-iou',type=float,default=0.7,help='below this agreement the proposed box is kept as an alternative and the boundary is unresolved')
    p.add_argument('--refine-tile-side',type=int,default=512,help='source-resolution tile side for refinement')
    p.add_argument('--calibration',help='calibration artifact; without it no probability is emitted and the reason is recorded')
    p.add_argument('--verify-bundle',help='bundle for an independent verifier; records per-box confirmation')
    p.add_argument('--verify-iou',type=float,default=0.5,help='overlap at which a verifier detection confirms a box')
    p.add_argument('--verify-threshold',type=float,default=0.3,help='verifier detection threshold')
    p.add_argument('--require-verification',action='store_true',help='acceptance additionally requires independent confirmation')
    p.add_argument('--accept',action='store_true',help='run the acceptance rule; without it every object stays uncertain')
    p.add_argument('--accept-score',type=float);p.add_argument('--accept-min-views',type=int)
    p.add_argument('--accept-boundary-iou',type=float);p.add_argument('--accept-granularity-ios',type=float)
    p=sub.add_parser('resume');p.add_argument('run');p.add_argument('--output')
    p=sub.add_parser('evaluate');p.add_argument('--benchmark',required=True);p.add_argument('--bootstrap-replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=0)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        if args.command=='doctor':result=doctor(args.bundle)
        elif args.command=='policy':result={'policy':load_policy(),'ontology':load_ontology(args.ontology),'available_ontologies':sorted(available_ontologies()),'available_vocabularies':{k:{'adapter':v['adapter'],'ontology':v['ontology'],'version':v['version'],'fitted_on':v.get('fitted_on')} for k,v in available_vocabularies().items()}}
        elif args.command=='schema':result=annotation_schema()
        elif args.command=='validate':result=validate_annotation(json.loads(Path(args.annotation).read_text()),args.artifact_root or Path(args.annotation).parent)
        elif args.command=='annotate':result=annotate(args.image,args.bundle,args.output,
            tile_levels=args.tile_levels,tile_overlap=args.tile_overlap,merge_ios=args.merge_ios,
            merge_iou=args.merge_iou,device=args.device,threshold=args.threshold,
            ontology_id=args.ontology,vocabulary_id=args.vocabulary,acceptance=acceptance_rules(args),
            merge_rounds=args.merge_rounds,merge_labels=args.merge_labels,fuse_boxes_enabled=args.fuse,calibration=args.calibration,
            verify_bundle=args.verify_bundle,verify_iou=args.verify_iou,verify_threshold=args.verify_threshold,
            refine_bundle=args.refine_bundle,refine_iou=args.refine_iou,refine_tile_side=args.refine_tile_side,
            **({} if args.name_ratio is None else {'name_ratio':args.name_ratio}))
        elif args.command=='resume':result=resume(args.run,args.output)
        else:result=evaluate_benchmark(json.loads(Path(args.benchmark).read_text()),args.bootstrap_replicates,args.seed)
        print(json.dumps(result,indent=2,allow_nan=False))
        if result.get('execution',{}).get('status') in ['failed','interrupted']:return 4
        if result.get('search',{}).get('status')=='budget_exhausted':return 3
        return 0
    except BundleError as exc:
        print(json.dumps({'error':{'code':'bundle_error','message':str(exc)}}),file=sys.stderr);return 5
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'error':{'code':'invalid_input','message':str(exc)}}),file=sys.stderr);return 2
