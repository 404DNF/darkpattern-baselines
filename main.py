import os
import argparse
import pandas as pd

from src.data_utils import load_and_validate_data
from src.classical_models import run_classical_models
from src.plm_models import run_plm_models, PLM_SPECS
from src.utils import ensure_dir, pretty_view


def parse_args():
    parser = argparse.ArgumentParser(description="Dark Pattern Detection Baselines (Classical NLP + Transformer PLMs)")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument(
        "--plm_models",
        nargs="*",
        default=list(PLM_SPECS.keys()),
        help="PLM model keys to run. Default: all"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print("==== [1] Load & Validate Data ====")
    df = load_and_validate_data(args.csv_path)

    print("\n==== [2] Run Classical NLP Models ====")
    classical_results = run_classical_models(df)
    classical_out_path = os.path.join(args.output_dir, "classical_5fold.csv")
    classical_results.to_csv(classical_out_path, index=False)
    print(f"saved: {classical_out_path}")
    print(classical_results)

    print("\n==== [3] Run Transformer PLM Models ====")
    plm_save_dir = os.path.join(args.output_dir, "plm_cv")
    plm_results = run_plm_models(df, args.plm_models, save_dir=plm_save_dir)

    plm_out_path = os.path.join(args.output_dir, "plm_5fold.csv")
    plm_results.to_csv(plm_out_path, index=False)
    print(f"saved: {plm_out_path}")
    print(plm_results)

    print("\n==== [4] Merge Final Results ====")
    all_results = pd.concat([classical_results, plm_results], ignore_index=True)
    all_results = all_results.sort_values("accuracy_mean", ascending=False)

    final_out_path = os.path.join(args.output_dir, "darkpattern_baselines_5fold.csv")
    all_results.to_csv(final_out_path, index=False)
    print(f"saved: {final_out_path}")

    print("\n--- Transformers ---")
    print(pretty_view(plm_results).to_string(index=False))

    print("\n--- Classical + Transformers ---")
    print(pretty_view(all_results).to_string(index=False))


if __name__ == "__main__":
    main()