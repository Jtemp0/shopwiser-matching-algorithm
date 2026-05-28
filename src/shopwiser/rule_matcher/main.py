"""CLI entry: load data, optional self-tests, run clustering passes."""

import sys

import pandas as pd


def run_similarity_self_tests() -> None:
    from shopwiser.paths import ensure_repo_on_syspath

    ensure_repo_on_syspath()
    from tests.test_similarity import run_tests as _run_tests

    from .config import (
        UNIT_TOLERANCE_BRANDED,
        UNIT_TOLERANCE_OWN_BRAND,
        UNIT_TOLERANCE_UNBRANDED,
    )
    from .similarity import compute_similarity

    print('\nRunning similarity self-tests...')
    _all_pass = _run_tests(
        compute_similarity,
        tol_branded=UNIT_TOLERANCE_BRANDED,
        tol_own=UNIT_TOLERANCE_OWN_BRAND,
        tol_unbranded=UNIT_TOLERANCE_UNBRANDED,
    )
    print(f'\nAll tests passed: {_all_pass}')
    if not _all_pass:
        print('WARNING: some tests failed — review thresholds before proceeding')
        sys.exit(1)


def main(*, sample: bool = False) -> None:
    from shopwiser.paths import cluster_outputs_path

    from . import config as cluster_config
    from .config import print_config_banner
    from .data_prep import load_prepared_dataframe
    from .pipeline import run_clustering

    cluster_config.OUTPUT_DIR = cluster_outputs_path(sample=sample)
    print_config_banner()
    df: pd.DataFrame = load_prepared_dataframe(sample=sample)
    run_similarity_self_tests()
    run_clustering(df)


if __name__ == '__main__':
    import argparse

    _p = argparse.ArgumentParser(description='Run ShopWiser clustering on normalized products CSV.')
    _p.add_argument(
        '--sample',
        action='store_true',
        help='Use normalized_products_sample.csv and write outputs under data/intermediate/sample/',
    )
    _args = _p.parse_args()
    main(sample=_args.sample)
