import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.fuzzy_design_quantile import MF_TYPES, generate_quantile_fuzzification_outputs
from src.full_sugeno_model import DATASET_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Generate quantile-based fuzzy membership artifacts.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG.keys()),
        choices=list(DATASET_CONFIG.keys()),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--mf-types",
        nargs="+",
        default=list(MF_TYPES),
        choices=list(MF_TYPES),
        help="Membership function types to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_quantile_fuzzification_outputs(args.datasets, args.mf_types, verbose=True)
    print("\n[OK] Quantile fuzzification artifacts generated.")


if __name__ == "__main__":
    main()
