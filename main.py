"""Unified ShopWiser CLI: normalise, cluster, ml-match, audit, similarity tests.

Examples:
    uv run python main.py normalise --sample
    uv run python main.py cluster --sample
    uv run python main.py ml-match --sample
    uv run python main.py export-demo
    uv run python main.py audit --sample
    uv run python main.py test-similarity

Module entrypoints remain available, e.g. ``python -m shopwiser.rule_matcher.main --sample`` or
``python -m shopwiser.ml_matcher.main --sample``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_normalise(args: argparse.Namespace) -> None:
    from shopwiser.preprocess.normalise import main

    main(sample=args.sample)


def _cmd_cluster(args: argparse.Namespace) -> None:
    from shopwiser.rule_matcher.main import main

    main(sample=args.sample)


def _cmd_ml_match(args: argparse.Namespace) -> None:
    from shopwiser.ml_matcher.main import run_ml_matching

    run_ml_matching(sample=args.sample)


def _cmd_ensemble(args: argparse.Namespace) -> None:
    from shopwiser.ensemble.main import run_ensemble

    run_ensemble(sample=args.sample)


def _cmd_export_demo(args: argparse.Namespace) -> None:
    from shopwiser.utils.cofounder_demo_export import export_cofounder_demo

    long_p, wide_p, html_p = export_cofounder_demo(
        ml_clusters_csv=Path(args.input) if args.input else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        use_stratified_fallback=args.stratified,
        n_clusters=args.n_clusters,
        seed=args.seed,
        write_html=args.html,
    )
    print('Wrote:')
    print(f'  {long_p}')
    print(f'  {wide_p}')
    if html_p:
        print(f'  {html_p}')


def _cmd_audit(args: argparse.Namespace) -> None:
    from shopwiser.audit import run_audit

    run_audit(sample=args.sample)


def _cmd_test_similarity(_args: argparse.Namespace) -> None:
    from shopwiser.rule_matcher.config import (
        UNIT_TOLERANCE_BRANDED,
        UNIT_TOLERANCE_OWN_BRAND,
        UNIT_TOLERANCE_UNBRANDED,
    )
    from shopwiser.rule_matcher.similarity import compute_similarity
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

    pm = sub.add_parser(
        'ml-match',
        aliases=('match-ml',),
        help='FAISS + LightGBM matching → data/outputs/ml_clusters/',
    )
    add_sample(pm)
    pm.set_defaults(func=_cmd_ml_match)

    pe = sub.add_parser(
        'ensemble',
        help='Union ML-matching + rule-based clusters → data/outputs/ensemble/',
    )
    add_sample(pe)
    pe.set_defaults(func=_cmd_ensemble)

    pdemo = sub.add_parser(
        'export-demo',
        aliases=('demo',),
        help='Build cofounder demo CSV/HTML from ml_clusters.csv → data/outputs/demo/',
    )
    pdemo.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to ml_clusters.csv (default: data/outputs/ml_clusters/ml_clusters.csv)',
    )
    pdemo.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output directory (default: data/outputs/demo)',
    )
    pdemo.add_argument('--n-clusters', type=int, default=25)
    pdemo.add_argument('--seed', type=int, default=42)
    pdemo.add_argument(
        '--stratified',
        action='store_true',
        help='Auto-sample clusters instead of the hand-validated list',
    )
    pdemo.add_argument('--html', action='store_true', help='Also write cofounder_demo.html')
    pdemo.set_defaults(func=_cmd_export_demo)

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
