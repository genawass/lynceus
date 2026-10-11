"""Train one student per arm and score every one against human labels on the full val split.

This is the measurement the finite-category reference cannot give directly. Precision against
VisDrone is a bound, because an unmatched box is either a real object the dataset omits or a false
positive and nothing here separates them. A student does not need them separated: a real omitted
object raises its score and a false box lowers it, so training prices the undecided population by
what it does to a consumer.

Every arm is identical but for its labels -- same images, architecture, weights, hyperparameters and
seed -- so the spread between arms is attributable to labels and to nothing else. The result is a
direction on 200 images and one architecture, not a gate.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def train_arm(arm, data, weights, settings, project, seed, device=None):
    from ultralytics import YOLO, settings as ultralytics_settings
    # Integrations write outside this project's directories, which an offline experiment should not.
    ultralytics_settings.update({'mlflow': False, 'comet': False, 'dvc': False, 'neptune': False,
                                 'clearml': False, 'wandb': False, 'tensorboard': False})
    model = YOLO(weights)
    started = time.perf_counter()
    model.train(data=str(data), imgsz=settings['imgsz'], epochs=settings['epochs'],
                batch=settings['batch'], seed=seed, deterministic=True, pretrained=True,
                project=str(project), name=f'{arm}-seed{seed}', exist_ok=True, verbose=False,
                val=False, plots=False, save_period=-1,
                **({'device': device} if device else {}))
    trained = time.perf_counter() - started
    metrics = model.val(data=str(data), imgsz=settings['imgsz'], split='val', verbose=False,
                        project=str(project), name=f'{arm}-seed{seed}-val', exist_ok=True, plots=False,
                        **({'device': device} if device else {}))
    box = metrics.box
    return {'arm': arm, 'seed': seed, 'train_seconds': round(trained, 1),
            'mAP50': round(float(box.map50), 6), 'mAP50_95': round(float(box.map), 6),
            'precision': round(float(box.mp), 6), 'recall': round(float(box.mr), 6),
            'per_class_mAP50': {name: round(float(value), 6)
                                for name, value in zip(metrics.names.values(), box.ap50)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(Path(__file__).parent / 'student-protocol.json'))
    parser.add_argument('--arms', required=True, help='directory written by export_students.py')
    parser.add_argument('--out', required=True, help='results json')
    parser.add_argument('--weights', required=True, help='local initial weights; nothing is downloaded')
    parser.add_argument('--only', action='append', help='train only these arm ids')
    parser.add_argument('--seeds', type=int, action='append')
    parser.add_argument('--device')
    parser.add_argument('--imgsz', type=int, help='override the protocol, for a wiring smoke test only')
    parser.add_argument('--epochs', type=int, help='override the protocol, for a wiring smoke test only')
    args = parser.parse_args()

    os.environ.setdefault('YOLO_OFFLINE', '1')
    protocol = json.loads(Path(args.protocol).read_text())
    settings = dict(protocol['student'])
    for key in ('imgsz', 'epochs'):
        if getattr(args, key):
            settings[key] = getattr(args, key)
            print(f'WARNING: {key}={settings[key]} overrides the declared protocol; '
                  'this is a wiring check and not a result', flush=True)
    seeds = args.seeds or settings['seeds']
    arms = [a['id'] for a in protocol['arms'] if not args.only or a['id'] in args.only]
    results = []
    out = Path(args.out)
    if out.is_file():
        results = json.loads(out.read_text())['runs']
    done = {(r['arm'], r['seed']) for r in results}
    for seed in seeds:
        for arm in arms:
            if (arm, seed) in done:
                print(f'{arm} seed {seed} already measured, skipped', flush=True)
                continue
            data = Path(args.arms) / arm / 'dataset.yaml'
            row = train_arm(arm, data, args.weights, settings, Path(args.arms) / 'students', seed,
                            device=args.device)
            results.append(row)
            print(json.dumps(row), flush=True)
            out.write_text(json.dumps({'protocol': protocol['id'], 'student': settings,
                                       'runs': results}, indent=1))
    print(f'{len(results)} arm-seed results -> {out}')


if __name__ == '__main__':
    main()
