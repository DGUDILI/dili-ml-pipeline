"""
DGUDILI training entry point

Usage:
    python src/train_dgudili.py [options]

Examples:
    python src/train_dgudili.py --env env1
    python src/train_dgudili.py --env env2
    python src/train_dgudili.py --env env1 --shap

Via run.sh:
    ./run.sh dgudili --env env1
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROJECT_ROOT = os.path.dirname(SRC_DIR)


def _build_save_dir(env: str, clean: bool) -> str:
    name = f"dgudili_{env}"
    if clean:
        name += "_clean"
    return os.path.join(PROJECT_ROOT, "src", "models", "stackdili_fixed", "Model", name)


def main():
    parser = argparse.ArgumentParser(description="DGUDILI training")
    parser.add_argument("--env",        choices=["env1", "env2"], default="env1",
                        help="env1=external validation, env2=10-Fold CV")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--d_model",    type=int,   default=64)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--clean",      action="store_true",
                        help="Remove train-test overlap before training")
    parser.add_argument("--shap",       action="store_true",
                        help="Run SHAP interpretation after training")
    args = parser.parse_args()

    if args.clean:
        import subprocess
        clean_script = os.path.join(PROJECT_ROOT, "src", "preprocessing", "make_clean_data.py")
        subprocess.run(["python", clean_script], check=True, text=True)
        features_path = os.path.join(PROJECT_ROOT, "src", "features", "dataset_features_cleaned.csv")
    else:
        features_path = os.path.join(PROJECT_ROOT, "src", "features", "dataset_features.csv")

    from models.stackdili_fixed.dgudili.dgudili_pipeline import DGUDILIPipeline

    save_dir = _build_save_dir(args.env, args.clean)
    pipeline = DGUDILIPipeline(
        env=args.env,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        lr=args.lr,
        seed=args.seed,
    )
    pipeline.run(features_path, save_dir)

    if args.shap and args.env == "env1":
        import pandas as pd
        from models.stackdili_fixed.ga.shap_interpreter import SHAPInterpreter

        df    = pd.read_csv(features_path)
        X_all = df.drop(["SMILES", "Label", "ref"], axis=1)
        y_all = df["Label"].values
        train_mask = df["ref"] != "DILIrank"

        print("\n[SHAP] FP feature importance...")
        interp = SHAPInterpreter(random_seed=args.seed)
        interp.fit(X_all[train_mask], y_all[train_mask])

        shap_dir = os.path.join(save_dir, "shap")
        interp.plot(save_path=os.path.join(shap_dir, "shap_bar.png"), top_n=30)
        interp.save_csv(os.path.join(shap_dir, "shap_importance.csv"))

        top16 = interp.top_k(k=16)
        print(f"\n[SHAP] Top-16 FP features: {top16}")


if __name__ == "__main__":
    main()
