import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.evaluation import predict_with_baselines, regression_metrics
from src.full_sugeno_model import DATASET_CONFIG, FullRuleSugenoModel
from src.fuzzy_design_quantile import (
    MF_TYPES,
    generate_quantile_fuzzification_for_dataset,
    load_quantile_membership_function,
)


DATASETS = ["albrecht", "desharnais"]
LLMS = ["gemini", "gpt", "claude"]
RESULTS_DIR = "reports/results/quantile"
PREDICTIONS_DIR = "reports/predictions/quantile"
RULE_ANALYSIS_DIR = "reports/rule_analysis/quantile"
FIGURES_DIR = "reports/figures/quantile"
EQUATIONS_DIR = "models/sugeno_equations_quantile"


def split_dataset(df):
    test_size = 0.30 if len(df) < 50 else 0.20
    return train_test_split(df, test_size=test_size, random_state=42, shuffle=True)


def cross_validate_full_sugeno_quantile(dataset_name, llm_name, df, mf_type, n_splits=5):
    n_splits = min(n_splits, len(df))
    if n_splits < 2:
        return {}

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df), start=1):
        membership_function = load_quantile_membership_function(dataset_name, mf_type)
        model = FullRuleSugenoModel(
            dataset_name,
            llm_name,
            membership_function=membership_function,
            fuzzification_name="quantile",
            mf_type=mf_type,
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(train_df)
        metrics = model.evaluate(test_df)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "CV RMSE": float(metrics_df["RMSE"].mean()),
        "CV MAE": float(metrics_df["MAE"].mean()),
        "CV MAPE (%)": float(metrics_df["MAPE (%)"].mean()),
        "CV R2": float(metrics_df["R2"].mean()),
    }


def _slug(value):
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
    )


def row_from_metrics(model_name, train_metrics, test_metrics, cv_metrics=None):
    cv_metrics = cv_metrics or {}
    return {
        "Model": model_name,
        "Train RMSE": train_metrics.get("RMSE", np.nan),
        "Train MAE": train_metrics.get("MAE", np.nan),
        "Train MAPE (%)": train_metrics.get("MAPE (%)", np.nan),
        "Train R2": train_metrics.get("R2", np.nan),
        "Test RMSE": test_metrics.get("RMSE", np.nan),
        "Test MAE": test_metrics.get("MAE", np.nan),
        "Test MAPE (%)": test_metrics.get("MAPE (%)", np.nan),
        "Test R2": test_metrics.get("R2", np.nan),
        "CV RMSE": cv_metrics.get("CV RMSE", np.nan),
        "CV MAE": cv_metrics.get("CV MAE", np.nan),
        "CV MAPE (%)": cv_metrics.get("CV MAPE (%)", np.nan),
        "CV R2": cv_metrics.get("CV R2", np.nan),
    }


def save_model_comparison(model, train_df, test_df, baseline_predictions, mf_type):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_actual = train_df[model.target_col].to_numpy(dtype=float)
    test_actual = test_df[model.target_col].to_numpy(dtype=float)
    model_predictions = {
        "Sugeno V2 Full Rule-Level Quantile": {
            "train": model.predict(train_df),
            "test": model.predict(test_df),
        },
        **baseline_predictions,
    }

    rows = []
    for model_name, predictions in model_predictions.items():
        rows.append(row_from_metrics(
            model_name,
            regression_metrics(train_actual, predictions["train"]),
            regression_metrics(test_actual, predictions["test"]),
        ))
    comparison_df = pd.DataFrame(rows)

    comparison_path = os.path.join(
        RESULTS_DIR,
        f"{model.dataset_name}_{model.output_llm_name}_{mf_type}_v2_model_comparison.csv",
    )
    comparison_df.to_csv(comparison_path, index=False)

    for split_name, split_df, actual in [
        ("train", train_df, train_actual),
        ("test", test_df, test_actual),
    ]:
        out = split_df[model.input_vars].copy()
        out["split"] = split_name
        out["actual"] = actual
        for model_name, predictions in model_predictions.items():
            slug = _slug(model_name)
            out[f"{slug}_predicted"] = predictions[split_name]
            out[f"{slug}_residual"] = actual - predictions[split_name]
        out.to_csv(
            os.path.join(
                PREDICTIONS_DIR,
                f"{model.dataset_name}_{model.output_llm_name}_{mf_type}_v2_model_comparison_{split_name}.csv",
            ),
            index=False,
        )

    metric_names = ["RMSE", "MAE", "MAPE (%)", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric_name in zip(axes.ravel(), metric_names):
        column = f"Test {metric_name}"
        ax.bar(comparison_df["Model"], comparison_df[column])
        if metric_name == "R2":
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_title("Test R2 (higher is better)")
        else:
            ax.set_title(f"Test {metric_name} (lower is better)")
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle(f"Quantile V2 Model Comparison - {model.dataset_name} {model.output_llm_name} {mf_type}")
    fig.tight_layout()
    metrics_path = os.path.join(
        FIGURES_DIR,
        f"{model.dataset_name}_{model.output_llm_name}_{mf_type}_v2_model_comparison_metrics.png",
    )
    fig.savefig(metrics_path, bbox_inches="tight")
    plt.close(fig)

    return comparison_path, metrics_path, comparison_df


def run_single(dataset_name, llm_name, mf_type):
    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset_name.capitalize()} | Fuzzification: Quantile-based | MF: {mf_type}")
    print(f"Model: Sugeno V2 Full Rule-Level | LLM: {llm_name.capitalize()}")
    print(f"{'=' * 72}")

    generate_quantile_fuzzification_for_dataset(dataset_name, mf_types=MF_TYPES, verbose=True)
    membership_function = load_quantile_membership_function(dataset_name, mf_type)
    model = FullRuleSugenoModel(
        dataset_name,
        llm_name,
        membership_function=membership_function,
        fuzzification_name="quantile",
        mf_type=mf_type,
    )

    df = model.load_training_frame()
    train_df, test_df = split_dataset(df)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    _, baseline_predictions = predict_with_baselines(train_df, test_df, model.input_vars, model.target_col)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_df)
        for warning in caught:
            print(f"[WARNING] {warning.message}")

    train_metrics = model.evaluate(train_df)
    test_metrics = model.evaluate(test_df)
    cv_metrics = cross_validate_full_sugeno_quantile(dataset_name, llm_name, df, mf_type)

    equations_json, equations_txt = model.save_equations(EQUATIONS_DIR)
    train_predictions = model.save_predictions(train_df, "train", PREDICTIONS_DIR)
    test_predictions = model.save_predictions(test_df, "test", PREDICTIONS_DIR)
    rule_contrib_path, dominant_rules_path = model.save_rule_analysis(df, split_name="all", output_dir=RULE_ANALYSIS_DIR)
    plot_paths = model.save_plots(train_df, test_df, output_dir=FIGURES_DIR)
    comparison_path, comparison_fig_path, _ = save_model_comparison(model, train_df, test_df, baseline_predictions, mf_type)

    print("\nSUMMARY")
    print(f"Dataset: {dataset_name}")
    print("Fuzzification: Quantile-based")
    print(f"MF Type: {mf_type}")
    print("Model: Sugeno V2 Full Rule-Level")
    print(f"Train/Test metrics: Train RMSE={train_metrics['RMSE']:.4f}, Test RMSE={test_metrics['RMSE']:.4f}, Test MAE={test_metrics['MAE']:.4f}, Test MAPE={test_metrics['MAPE (%)']:.2f}, Test R2={test_metrics['R2']:.4f}")
    if cv_metrics:
        print(f"CV metrics: RMSE={cv_metrics['CV RMSE']:.4f}, MAE={cv_metrics['CV MAE']:.4f}, MAPE={cv_metrics['CV MAPE (%)']:.2f}, R2={cv_metrics['CV R2']:.4f}")
    print(f"Saved results: {comparison_path}")

    return {
        "Dataset": dataset_name,
        "LLM": model.output_llm_name,
        "Model": "Sugeno V2 Full Rule-Level",
        "Fuzzification": "Quantile",
        "MF Type": mf_type,
        "Rules": len(model.rules),
        "Inputs": len(model.input_vars),
        "Parameters": model.total_params,
        "Train RMSE": train_metrics["RMSE"],
        "Train MAE": train_metrics["MAE"],
        "Train MAPE (%)": train_metrics["MAPE (%)"],
        "Train R2": train_metrics["R2"],
        "Test RMSE": test_metrics["RMSE"],
        "Test MAE": test_metrics["MAE"],
        "Test MAPE (%)": test_metrics["MAPE (%)"],
        "Test R2": test_metrics["R2"],
        **cv_metrics,
        "Equations JSON": equations_json,
        "Equations TXT": equations_txt,
        "Train Predictions": train_predictions,
        "Test Predictions": test_predictions,
        "Rule Contributions": rule_contrib_path,
        "Dominant Rules": dominant_rules_path,
        "Model Comparison": comparison_path,
        "Comparison Metrics Figure": comparison_fig_path,
        "Surface Figure": plot_paths.get("surface", ""),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run quantile-based Sugeno V2 full rule-level models.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--llms", nargs="+", default=LLMS, choices=LLMS)
    parser.add_argument("--mf-type", default="triangular", choices=MF_TYPES)
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = []
    for dataset_name in args.datasets:
        for llm_name in args.llms:
            all_rows.append(run_single(dataset_name, llm_name, args.mf_type))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "full_sugeno_quantile_summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\n[OK] Quantile Sugeno V2 run completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
