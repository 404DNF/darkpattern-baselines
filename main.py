import os
import argparse
import pandas as pd

from src.config import (
    RUN_BINARY,
    RUN_HIERARCHICAL,
    RUN_TASKS,
    SAVE_CLASSWISE,
)
from src.data_utils import load_binary_data, load_hierarchical_data
from src.classical_models import run_classical_models
from src.plm_models import (
    PLM_SPECS,
    run_binary_plm_models,
    run_hierarchical_plm_models,
    pretty_view_hierarchical,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dark Pattern Classification Experiments"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--plm_models",
        nargs="*",
        default=list(PLM_SPECS.keys()),
        help="PLM model keys to run. Default: all",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only locally cached Hugging Face files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================
    # STEP 1. Binary Classification (NDP vs DP)
    # =========================================================
    if RUN_BINARY:
        print("\n" + "=" * 60)
        print("STEP 1. Binary Classification (NDP vs DP)")
        print("=" * 60)

        df_bin = load_binary_data(args.csv_path)

        # 1-1) Classical models
        print("\n==== [1-1] Run Classical NLP Models ====")
        classical_results = run_classical_models(df_bin)

        classical_out = os.path.join(args.output_dir, "classical_5fold.csv")
        classical_results.to_csv(classical_out, index=False)
        print(f"✅ saved: {classical_out}")
        print(classical_results)

        # 1-2) Binary PLM models
        print("\n==== [1-2] Run Binary Transformer PLM Models ====")
        binary_plm_dir = os.path.join(args.output_dir, "plm_cv")
        plm_results = run_binary_plm_models(
            df=df_bin,
            model_keys=args.plm_models,
            save_dir=binary_plm_dir,
            local_files_only=args.local_files_only,
        )

        if len(plm_results) > 0:
            plm_out = os.path.join(args.output_dir, "plm_5fold.csv")
            plm_results.to_csv(plm_out, index=False)
            print(f"✅ saved: {plm_out}")
            print(plm_results)

            # 1-3) Merge binary results
            print("\n==== [1-3] Merge Binary Results ====")
            all_results = pd.concat([classical_results, plm_results], ignore_index=True)
            all_results = all_results.sort_values("accuracy_mean", ascending=False)

            final_binary_out = os.path.join(
                args.output_dir,
                "darkpattern_baselines_5fold.csv",
            )
            all_results.to_csv(final_binary_out, index=False)
            print(f"✅ saved: {final_binary_out}")
        else:
            print("⚠️ No binary PLM results were produced.")

    # =========================================================
    # STEP 2-3. Hierarchical Multiclass
    #  - Predicate classification
    #  - Type classification
    # =========================================================
    if RUN_HIERARCHICAL:
        print("\n" + "=" * 60)
        print("STEP 2-3. Hierarchical Multiclass Classification")
        print("=" * 60)

        df_hier, meta = load_hierarchical_data(args.csv_path)

        hier_dir = os.path.join(args.output_dir, "hierarchical")
        os.makedirs(hier_dir, exist_ok=True)

        hier_results = run_hierarchical_plm_models(
            df=df_hier,
            meta=meta,
            run_tasks=RUN_TASKS,          # ["predicate", "type"]
            model_keys=args.plm_models,
            save_dir=hier_dir,
            save_classwise=SAVE_CLASSWISE,
            local_files_only=args.local_files_only,
        )

        if len(hier_results) > 0:
            print("\n==== [2-3] Hierarchical Results ====")
            print(pretty_view_hierarchical(hier_results).to_string(index=False))
        else:
            print("⚠️ No hierarchical results were produced.")

    print("\n🎉 All requested experiments finished.")


if __name__ == "__main__":
    main()