"""Report what the automatic verifier's agreement supports, and label what it does not."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from lynceus.policy import load_ontology  # noqa: E402
from lynceus.verification import by_stratum, class_agreement, summarise  # noqa: E402

# The two vocabularies are not the same list: the verifier is prompted with ontology names, and a
# name it returns is reconciled to the ontology id before disagreement is counted.
SYNONYMS = {'motorcycle': 'motorcycle', 'motorbike': 'motorcycle', 'person': 'person',
            'car': 'car', 'van': 'van', 'truck': 'truck', 'bus': 'bus',
            'bicycle': 'bicycle', 'tricycle': 'tricycle', 'vehicle': 'vehicle'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()

    data = json.loads(Path(args.verification).read_text())
    rows = data['rows']
    overall = summarise(rows)
    sizes = by_stratum(rows, 'size_bin')
    classes = by_stratum(rows, 'label')
    agreement = class_agreement(rows, SYNONYMS, load_ontology('aerial-traffic-strict-v1'))

    print(f"verifier: {data['verifier']}, prompts {data['prompts']}, threshold {data['threshold']}\n")
    print(f"{overall['retained']} retained boxes: {overall['known_true_positives']} already settled by the "
          f"reference, {overall['undecided']} undecided")
    print(f"  precision lower bound (nothing undecided is real) : {overall['uncorrected_precision']}")
    print(f"  verifier sensitivity, measured on known positives : {overall['sensitivity_on_known_positives']}")
    print(f"  verifier confirmation rate on undecided boxes     : {overall['confirmation_rate_on_undecided']}")
    if 'implied_real_among_undecided' in overall:
        print(f"  precision if every confirmed undecided box is real: {overall['precision_if_confirmed_are_real']}")
        print(f"  precision under equal sensitivity (assumption)   : {overall['precision_under_equal_sensitivity']}")
        print(f"  implied real among undecided                     : {overall['implied_real_among_undecided']}"
              f" of {overall['undecided']}")

    for title, table in (('by size bin', sizes), ('by emitted class', classes)):
        print(f"\n{title}")
        print(f"  {'':12s} {'known+':>7s} {'sens':>6s} {'undec':>7s} {'conf':>6s}")
        for name, v in table.items():
            sens = '-' if v['sensitivity'] is None else f"{v['sensitivity']:.3f}"
            conf = '-' if v['confirmation_on_undecided'] is None else f"{v['confirmation_on_undecided']:.3f}"
            print(f"  {name:12s} {v['known_positives']:7d} {sens:>6s} {v['undecided']:7d} {conf:>6s}")

    print(f"\nclass comparison where both models place a box ({agreement['paired']} boxes)")
    print(f"  exact agreement                : {agreement['exact']:5d}  {agreement['agreement']}")
    print(f"  compatible, emitted is broader : {agreement['compatible_less_specific']:5d}")
    print(f"  genuine conflict               : {agreement['conflict']:5d}  {agreement['conflict_rate']}")
    print(f"  compatible overall             : {agreement['compatible_rate']}")
    print("  where the system was less specific than it could have been:")
    for (a, b), n in agreement.get('top_under_specified', []):
        print(f"    emitted {a:14s} verifier says {b:12s} {n}")
    print("  genuine conflicts:")
    for (a, b), n in agreement.get('top_conflicts', []):
        print(f"    emitted {a:14s} verifier says {b:12s} {n}")
    print(f"\n{overall['interpretation']}")

    if args.out:
        Path(args.out).write_text(json.dumps({'overall': overall, 'by_size': sizes,
                                              'by_class': classes, 'class_agreement': agreement}, indent=1))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
