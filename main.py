"""Unified ShopWiser CLI: normalise, cluster, audit, similarity tests.

Examples:
    uv run python main.py normalise --sample
    uv run python main.py cluster --sample
    uv run python main.py audit --sample
    uv run python main.py test-similarity

Module entrypoints remain available, e.g. ``python -m shopwiser.clustering.main --sample``.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_normalise(args: argparse.Namespace) -> None:
    from shopwiser.preprocess.normalise import main

    main(sample=args.sample)


def _cmd_cluster(args: argparse.Namespace) -> None:
    from shopwiser.clustering.main import main

    main(sample=args.sample)


def _cmd_audit(args: argparse.Namespace) -> None:
    from shopwiser.audit import run_audit

    run_audit(sample=args.sample)


def _cmd_test_similarity(_args: argparse.Namespace) -> None:
    from shopwiser.clustering.config import (
        UNIT_TOLERANCE_BRANDED,
        UNIT_TOLERANCE_OWN_BRAND,
        UNIT_TOLERANCE_UNBRANDED,
    )
    from shopwiser.clustering.similarity import compute_similarity
    from shopwiser.paths import ensure_repo_on_syspath

    ensure_repo_on_syspath()
    from tests.test_similarity import run_tests

    print('\nRunning ALL similarity self-tests...')
    ok = run_tests(
        compute_similarity,
        tol_branded=UNIT_TOLERANCE_BRANDED,
        tol_own=UNIT_TOLERANCE_OWN_BRAND,
        tol_unbranded=UNIT_TOLERANCE_UNBRANDED,
    )
    print(f'\nAll tests passed: {ok}')
    sys.exit(0 if ok else 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main',
        description='ShopWiser pipeline: normalise → cluster → optional LLM audit.',
    )
    sub = p.add_subparsers(dest='command', required=True)

    def add_sample(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            '--sample',
            action='store_true',
            help='Use raw_1000 / normalized_products_sample / clusters_sample paths',
        )

    pn = sub.add_parser(
        'normalise',
        aliases=('norm',),
        help='Raw CSV → processed features (normalized_products*.csv)',
    )
    add_sample(pn)
    pn.set_defaults(func=_cmd_normalise)

    pc = sub.add_parser('cluster', help='Cluster normalized CSV → clusters/ + audit_sample_50.csv')
    add_sample(pc)
    pc.set_defaults(func=_cmd_cluster)

    pa = sub.add_parser(
        'audit',
        help='LLM audit of audit_sample_50.csv (needs ANTHROPIC_API_KEY)',
    )
    add_sample(pa)
    pa.set_defaults(func=_cmd_audit)

    pt = sub.add_parser(
        'test-similarity',
        aliases=('test',),
        help='Run compute_similarity unit checks (no large CSV)',
    )
    pt.set_defaults(func=_cmd_test_similarity)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    func(args)


if __name__ == '__main__':
    main()
