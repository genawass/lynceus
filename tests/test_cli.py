import json
from lynceus.cli import main

def test_cli(capsys,staged):
    assert main(['doctor','--offline'])==0;assert json.loads(capsys.readouterr().out)['offline']
    assert main(['schema'])==0;assert '$defs' in json.loads(capsys.readouterr().out)
    assert main(['evaluate','--benchmark','examples/benchmark.json','--bootstrap-replicates','10'])==0;assert json.loads(capsys.readouterr().out)['qualification']['status']=='inconclusive'
    image,_,run=staged;assert main(['annotate',str(image),'--bundle','configs/bundle.example.json','--output',str(run)])==5;assert json.loads(capsys.readouterr().err)['error']['code']=='bundle_error'


def test_every_annotate_option_is_wired(staged):
    """The library API is not proof the CLI reaches it; parse and map each flag explicitly."""
    from lynceus.cli import build_parser, acceptance_rules
    image, _, run = staged
    args = build_parser().parse_args([
        'annotate', str(image), '--bundle', 'b', '--output', str(run),
        '--tile-levels', '2', '--tile-overlap', '0.3', '--merge-ios', '0.7', '--merge-iou', '0.4',
        '--device', 'cpu', '--threshold', '0.05', '--ontology', 'aerial-traffic-v1',
        '--accept', '--accept-score', '0.8', '--accept-min-views', '3',
        '--accept-boundary-iou', '0.9', '--accept-granularity-ios', '0.6'])
    assert (args.tile_levels, args.tile_overlap, args.merge_ios, args.merge_iou) == (2, 0.3, 0.7, 0.4)
    assert (args.device, args.threshold, args.ontology) == ('cpu', 0.05, 'aerial-traffic-v1')
    assert acceptance_rules(args) == {'score': 0.8, 'min_views': 3,
                                      'boundary_iou': 0.9, 'granularity_ios': 0.6}
    assert acceptance_rules(build_parser().parse_args(
        ['annotate', 'i', '--bundle', 'b', '--output', 'o'])) is None

def test_policy_command_lists_ontologies(capsys):
    assert main(['policy']) == 0
    out = json.loads(capsys.readouterr().out)
    assert 'aerial-traffic-v1' in out['available_ontologies']
    assert out['ontology']['id'] == 'lynceus-broad-v1'
    assert main(['policy', '--ontology', 'aerial-traffic-v1']) == 0
    assert json.loads(capsys.readouterr().out)['ontology']['id'] == 'aerial-traffic-v1'


def test_dedup_and_calibration_flags_are_wired(staged):
    from lynceus.cli import build_parser
    image, _, run = staged
    args = build_parser().parse_args(['annotate', str(image), '--bundle', 'b', '--output', str(run),
                                      '--merge-rounds', '3', '--fuse', '--calibration', 'c.json'])
    assert (args.merge_rounds, args.fuse, args.calibration) == (3, True, 'c.json')
    default = build_parser().parse_args(['annotate', 'i', '--bundle', 'b', '--output', 'o'])
    # both measured harmful or inert, so both are off unless asked for
    assert (default.merge_rounds, default.fuse, default.calibration) == (1, False, None)
