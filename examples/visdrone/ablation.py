"""Compare lynceus run sets on the frozen VisDrone mapping (the WP-06 / gate G7 ablation shape).

Reports, for each run set, the evaluator's metrics under the declared category scope and the same
metrics with no scope declared. The two are bounds: scoped precision excludes predictions the
reference cannot adjudicate, unscoped precision charges every one of them as an error. The truth
for in-scope objects lies between. Neither number qualifies anything.
"""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from lynceus.evaluation import evaluate_benchmark  # noqa: E402

METRICS = ['localized_instance_recall', 'final_instance_precision', 'accepted_annotation_precision',
           'useful_label_coverage', 'classification_accuracy', 'duplicate_rate', 'unknown_rate']


def build(runs, output, count, seed):
    script = Path(__file__).parent / 'build_benchmark.py'
    subprocess.run([sys.executable, str(script), '--runs', str(runs), '--output', str(output),
                    '--count', str(count), '--seed', str(seed)], check=True, capture_output=True)
    return json.loads(Path(output).read_text())


def row(label, metrics, compute):
    cells = ' '.join(f'{metrics[m]["numerator"]:>5d}/{metrics[m]["denominator"]:<5d}' for m in METRICS)
    return f'{label:<22} {cells}  {compute}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', action='append', required=True, metavar='NAME=DIR')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--out')
    args = parser.parse_args()

    report = {'metrics_order': METRICS, 'run_sets': {}}
    with tempfile.TemporaryDirectory() as tmp:
        print(f'{"":<22} ' + ' '.join(f'{m[:11]:^11}' for m in METRICS) + '  compute')
        for spec in args.runs:
            name, _, directory = spec.partition('=')
            payload = build(Path(directory), Path(tmp) / f'{name}.json', args.count, args.seed)
            results = [json.loads(path.read_text())
                       for path in (Path(directory) / im['id'] / 'annotation.json' for im in payload['images'])
                       if path.is_file()]
            views = sum(r['summary']['resources']['model_calls'] for r in results)
            seconds = sum(r['summary']['resources']['wall_seconds'] for r in results)
            missing = len(payload['images']) - len(results)
            compute = f'{views} views, {seconds:.0f}s, {len(results)}/{len(payload["images"])} runs'
            if missing:
                compute += f' ({missing} MISSING, still counted as misses)'
            scoped = evaluate_benchmark(copy.deepcopy(payload), args.replicates)
            unscoped_payload = copy.deepcopy(payload)
            unscoped_payload.pop('category_scope'), unscoped_payload.pop('category_scope_reason')
            unscoped = evaluate_benchmark(unscoped_payload, args.replicates)
            print(row(f'{name} (scoped)', scoped['metrics'], compute))
            print(row(f'{name} (unscoped)', unscoped['metrics'], f'out-of-scope preds: {scoped["out_of_scope_predictions"]}'))
            report['run_sets'][name] = {'compute': {'views': views, 'wall_seconds': seconds},
                                        'scoped': scoped, 'unscoped': unscoped}

    names = list(report['run_sets'])
    if len(names) == 2:
        a, b = (report['run_sets'][n]['scoped']['metrics']['localized_instance_recall'] for n in names)
        gained = b['numerator'] - a['numerator']
        misses = a['denominator'] - a['numerator']
        print(f'\n{names[1]} vs {names[0]}: {gained:+d} additional eligible references localized '
              f'({gained / misses * 100:.1f}% of the {misses} {names[0]} misses recovered)' if misses else '')
        print('Gate G7 proposes >=20% of baseline misses recovered. This panel is far too small to '
              'support that claim; the number is descriptive only.')
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f'\nreport -> {args.out}')


if __name__ == '__main__':
    main()
