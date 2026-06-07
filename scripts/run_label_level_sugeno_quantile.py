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
from src.fuzzy_design_quantile import (
    MF_TYPES,
    generate_quantile_fuzzification_for_dataset,
    load_quantile_membership_function,
)
from src.full_sugeno_model import DATASET_CONFIG
from src.label_level_sugeno_model import LabelLevelSugenoModel


DATASETS = ["albrecht", "desharnais"]
LLMS = ["gemini", "gpt", "claude"]
RESULTS_ROOT = "reports/results/quantile"
PREDICTIONS_ROOT = "reports/predictions/quantile"
RULE_ANALYSIS_ROOT = "reports/rule_analysis/quantile"
FIGURES_ROOT = "reports/figures/quantile"
EQUATIONS_ROOT = "models/sugeno_label_equations_quantile"
RESULTS_DIR = os.path.join(RESULTS_ROOT, "triangular")
PREDICTIONS_DIR = os.path.join(PREDICTIONS_ROOT, "triangular")
RULE_ANALYSIS_DIR = os.path.join(RULE_ANALYSIS_ROOT, "triangular")
FIGURES_DIR = os.path.join(FIGURES_ROOT, "triangular")
EQUATIONS_DIR = os.path.join(EQUATIONS_ROOT, "triangular")
COMPARISON_COLUMNS = [
    "Dataset",
    "LLM",
    "Model",
    "Fuzzification",
    "MF Type",
    "Rules",
    "Inputs",
    "Parameters",
    "Train RMSE",
    "Test RMSE",
    "Test MAE",
    "Test MAPE (%)",
    "Test R2",
    "CV RMSE",
    "CV MAE",
    "CV MAPE (%)",
    "CV R2",
]


def set_output_dirs(mf_type):
    global RESULTS_DIR, PREDICTIONS_DIR, RULE_ANALYSIS_DIR, FIGURES_DIR, EQUATIONS_DIR
    RESULTS_DIR = os.path.join(RESULTS_ROOT, mf_type)
    PREDICTIONS_DIR = os.path.join(PREDICTIONS_ROOT, mf_type)
    RULE_ANALYSIS_DIR = os.path.join(RULE_ANALYSIS_ROOT, mf_type)
    FIGURES_DIR = os.path.join(FIGURES_ROOT, mf_type)
    EQUATIONS_DIR = os.path.join(EQUATIONS_ROOT, mf_type)


def resolve_mf_types(mf_type):
    if mf_type == "all":
        return list(MF_TYPES)
    return [mf_type]


def split_dataset(df):
    test_size = 0.30 if len(df) < 50 else 0.20
    return train_test_split(df, test_size=test_size, random_state=42, shuffle=True)


def cross_validate_label_sugeno_quantile(dataset_name, llm_name, df, mf_type, n_splits=5):
    n_splits = min(n_splits, len(df))
    if n_splits < 2:
        return {}

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df), start=1):
        membership_function = load_quantile_membership_function(dataset_name, mf_type)
        model = LabelLevelSugenoModel(
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


def build_baseline_rows_and_predictions(train_df, test_df, input_vars, target_col):
    _, predictions = predict_with_baselines(train_df, test_df, input_vars, target_col)
    train_actual = train_df[target_col].to_numpy(dtype=float)
    test_actual = test_df[target_col].to_numpy(dtype=float)

    rows = []
    for model_name, preds in predictions.items():
        rows.append(row_from_metrics(
            model_name,
            regression_metrics(train_actual, preds["train"]),
            regression_metrics(test_actual, preds["test"]),
        ))

    return rows, predictions


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


def load_v2_quantile_row(dataset_name, llm_name, mf_type):
    summary_path = os.path.join(RESULTS_DIR, "full_sugeno_quantile_summary.csv")
    if not os.path.exists(summary_path):
        return None

    summary_df = pd.read_csv(summary_path)
    matching = summary_df[
        (summary_df["Dataset"] == dataset_name)
        & (summary_df["LLM"] == llm_name)
        & (summary_df["MF Type"] == mf_type)
        & (summary_df["Model"] == "Sugeno V2 Full Rule-Level")
    ]
    if matching.empty:
        return None

    row = matching.iloc[0]
    return {col: row.get(col, np.nan) for col in [
        "Model",
        "Train RMSE",
        "Train MAE",
        "Train MAPE (%)",
        "Train R2",
        "Test RMSE",
        "Test MAE",
        "Test MAPE (%)",
        "Test R2",
        "CV RMSE",
        "CV MAE",
        "CV MAPE (%)",
        "CV R2",
    ]}


def save_final_prediction_comparison(model, test_df, baseline_predictions, label_predictions, mf_type):
    actual = test_df[model.target_col].to_numpy(dtype=float)
    prediction_items = {
        "Sugeno V1 Label-Level Quantile": label_predictions,
        "Linear Regression": baseline_predictions["Linear Regression"]["test"],
        "Decision Tree": baseline_predictions["Decision Tree"]["test"],
    }

    v2_prediction_path = os.path.join(
        PREDICTIONS_DIR,
        f"{model.dataset_name}_{model.output_llm_name}_test_predictions.csv",
    )
    if os.path.exists(v2_prediction_path):
        v2_df = pd.read_csv(v2_prediction_path)
        if "predicted" in v2_df.columns and len(v2_df) == len(test_df):
            prediction_items["Sugeno V2 Full Rule-Level Quantile"] = v2_df["predicted"].to_numpy(dtype=float)

    out = test_df[model.input_vars].copy()
    out["actual"] = actual
    for model_name, predicted in prediction_items.items():
        slug = _slug(model_name)
        out[f"{slug}_predicted"] = predicted
        out[f"{slug}_residual"] = actual - predicted

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    path = os.path.join(
        PREDICTIONS_DIR,
        f"{model.dataset_name}_{model.output_llm_name}_{mf_type}_final_quantile_model_comparison_test.csv",
    )
    out.to_csv(path, index=False)
    return path, {"actual": actual, "predictions": prediction_items}


def save_final_comparison_plots(model, comparison_df, prediction_payload, mf_type):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = {}

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
    fig.suptitle(f"Quantile Final Model Comparison - {model.dataset_name} {model.output_llm_name} {mf_type}")
    fig.tight_layout()
    metrics_path = os.path.join(
        FIGURES_DIR,
        f"{model.dataset_name}_{model.output_llm_name}_{mf_type}_final_quantile_model_comparison_metrics.png",
    )
    fig.savefig(metrics_path, bbox_inches="tight")
    plt.close(fig)
    paths["metrics"] = metrics_path

    test_actual = prediction_payload["actual"]
    prediction_items = prediction_payload["predictions"]
    fig, axes = plt.subplots(1, len(prediction_items), figsize=(5 * len(prediction_items), 4), squeeze=False)
    values = [test_actual, *prediction_items.values()]
    min_val = min(np.min(value) for value in values)
    max_val = max(np.max(value) for value in values)
    for ax, (model_name, predicted) in zip(axes.ravel(), prediction_items.items()):
        ax.scatter(test_actual, predicted, alpha=0.8)
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_title(model_name)
        ax.set_xlabel("Actual Effort")
        ax.set_ylabel("Predicted Effort")
    fig.tight_layout()
    pred_path = os.path.join(
        FIGURES_DIR,
        f"{model.dataset_name}_{model.output_llm_name}_{mf_type}_final_quantile_predicted_vs_actual.png",
    )
    fig.savefig(pred_path, bbox_inches="tight")
    plt.close(fig)
    paths["predicted_vs_actual"] = pred_path

    return paths


def summary_row(dataset_name, llm_name, model_name, row, fuzzification, mf_type, rules, inputs, parameters):
    return {
        "Dataset": dataset_name,
        "LLM": llm_name,
        "Model": model_name,
        "Fuzzification": fuzzification,
        "MF Type": mf_type,
        "Rules": rules,
        "Inputs": inputs,
        "Parameters": parameters,
        "Train RMSE": row.get("Train RMSE", np.nan),
        "Train MAE": row.get("Train MAE", np.nan),
        "Train MAPE (%)": row.get("Train MAPE (%)", np.nan),
        "Train R2": row.get("Train R2", np.nan),
        "Test RMSE": row.get("Test RMSE", np.nan),
        "Test MAE": row.get("Test MAE", np.nan),
        "Test MAPE (%)": row.get("Test MAPE (%)", np.nan),
        "Test R2": row.get("Test R2", np.nan),
        "CV RMSE": row.get("CV RMSE", np.nan),
        "CV MAE": row.get("CV MAE", np.nan),
        "CV MAPE (%)": row.get("CV MAPE (%)", np.nan),
        "CV R2": row.get("CV R2", np.nan),
    }


def run_single(dataset_name, llm_name, mf_type):
    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset_name.capitalize()} | Fuzzification: Quantile-based | MF: {mf_type}")
    print(f"Model: Sugeno V1 Label-Level | LLM: {llm_name.capitalize()}")
    print(f"{'=' * 72}")

    generate_quantile_fuzzification_for_dataset(dataset_name, mf_types=MF_TYPES, verbose=True)
    membership_function = load_quantile_membership_function(dataset_name, mf_type)
    model = LabelLevelSugenoModel(
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

    baseline_rows, baseline_predictions = build_baseline_rows_and_predictions(
        train_df,
        test_df,
        model.input_vars,
        model.target_col,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_df)
        for warning in caught:
            print(f"[WARNING] {warning.message}")

    train_metrics = model.evaluate(train_df)
    test_metrics = model.evaluate(test_df)
    cv_metrics = cross_validate_label_sugeno_quantile(dataset_name, llm_name, df, mf_type)

    equations_json, equations_txt = model.save_equations(EQUATIONS_DIR)
    train_predictions = model.save_predictions(train_df, "train", PREDICTIONS_DIR)
    test_predictions = model.save_predictions(test_df, "test", PREDICTIONS_DIR)
    rule_contrib_path, dominant_rules_path = model.save_rule_analysis(df, split_name="all", output_dir=RULE_ANALYSIS_DIR)
    label_plot_paths = model.save_plots(train_df, test_df, output_dir=FIGURES_DIR)

    label_row = row_from_metrics(
        "Sugeno V1 Label-Level",
        train_metrics,
        test_metrics,
        cv_metrics,
    )
    comparison_rows = [*baseline_rows, label_row]
    v2_row = load_v2_quantile_row(dataset_name, model.output_llm_name, mf_type)
    if v2_row:
        comparison_rows.append(v2_row)

    comparison_df = pd.DataFrame(comparison_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison_path = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_{model.output_llm_name}_{mf_type}_final_quantile_model_comparison.csv",
    )
    comparison_df.to_csv(comparison_path, index=False)

    final_prediction_path, prediction_payload = save_final_prediction_comparison(
        model,
        test_df,
        baseline_predictions,
        model.predict(test_df),
        mf_type,
    )
    final_plot_paths = save_final_comparison_plots(model, comparison_df, prediction_payload, mf_type)

    print("\nSUMMARY")
    print(f"Dataset: {dataset_name}")
    print("Fuzzification: Quantile-based")
    print(f"MF Type: {mf_type}")
    print("Model: Sugeno V1 Label-Level")
    print(f"Train/Test metrics: Train RMSE={train_metrics['RMSE']:.4f}, Test RMSE={test_metrics['RMSE']:.4f}, Test MAE={test_metrics['MAE']:.4f}, Test MAPE={test_metrics['MAPE (%)']:.2f}, Test R2={test_metrics['R2']:.4f}")
    if cv_metrics:
        print(f"CV metrics: RMSE={cv_metrics['CV RMSE']:.4f}, MAE={cv_metrics['CV MAE']:.4f}, MAPE={cv_metrics['CV MAPE (%)']:.2f}, R2={cv_metrics['CV R2']:.4f}")
    print(f"Saved results: {comparison_path}")

    summary_rows = []
    for base_row in baseline_rows:
        parameters = len(model.input_vars) + 1 if base_row["Model"] == "Linear Regression" else np.nan
        summary_rows.append(summary_row(
            dataset_name,
            model.output_llm_name,
            base_row["Model"],
            base_row,
            "Baseline",
            "",
            0,
            len(model.input_vars),
            parameters,
        ))

    summary_rows.append({
        **summary_row(
            dataset_name,
            model.output_llm_name,
            "Sugeno V1 Label-Level",
            label_row,
            "Quantile",
            mf_type,
            len(model.rules),
            len(model.input_vars),
            model.total_params,
        ),
        "Equations JSON": equations_json,
        "Equations TXT": equations_txt,
        "Train Predictions": train_predictions,
        "Test Predictions": test_predictions,
        "Rule Contributions": rule_contrib_path,
        "Dominant Rules": dominant_rules_path,
        "Final Model Comparison": comparison_path,
        "Final Prediction Comparison": final_prediction_path,
        "Label Predicted Figure": label_plot_paths["predicted_vs_actual"],
        "Final Metrics Figure": final_plot_paths["metrics"],
    })

    if v2_row:
        summary_rows.append(summary_row(
            dataset_name,
            model.output_llm_name,
            "Sugeno V2 Full Rule-Level",
            v2_row,
            "Quantile",
            mf_type,
            len(model.rules),
            len(model.input_vars),
            len(model.rules) * (len(model.input_vars) + 1),
        ))

    return summary_rows


def _comparison_label(row):
    mf_type = row.get("MF Type", "")
    mf_suffix = f" {mf_type}" if isinstance(mf_type, str) and mf_type else ""
    return f"{row['LLM']} {row['Model']} {row['Fuzzification']}{mf_suffix}"


def save_uniform_vs_quantile_plots(comparison_df):
    os.makedirs(FIGURES_ROOT, exist_ok=True)
    paths = []
    sugeno_df = comparison_df[comparison_df["Model"].str.contains("Sugeno", na=False)].copy()
    for dataset_name in DATASETS:
        dataset_df = sugeno_df[sugeno_df["Dataset"] == dataset_name].copy()
        if dataset_df.empty:
            continue
        dataset_df["Plot Label"] = dataset_df.apply(_comparison_label, axis=1)
        for metric_name, file_part in [("Test RMSE", "rmse"), ("Test MAPE (%)", "mape")]:
            plot_df = dataset_df.dropna(subset=[metric_name]).sort_values(metric_name)
            if plot_df.empty:
                continue
            plt.figure(figsize=(10, max(4, 0.45 * len(plot_df))))
            colors = plot_df["Fuzzification"].map({"Uniform": "#4C78A8", "Quantile": "#F58518"}).fillna("#777777")
            plt.barh(plot_df["Plot Label"], plot_df[metric_name], color=colors)
            plt.xlabel(metric_name)
            plt.title(f"{dataset_name.capitalize()} Uniform vs Quantile - {metric_name}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = os.path.join(FIGURES_ROOT, f"{dataset_name}_uniform_vs_quantile_{file_part}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            paths.append(path)
    return paths


def save_uniform_vs_quantile_comparison(mf_type):
    quantile_summary_path = os.path.join(RESULTS_ROOT, "final_quantile_all_models_summary.csv")
    if not os.path.exists(quantile_summary_path):
        return None, []

    rows = []
    quantile_df = pd.read_csv(quantile_summary_path)
    if mf_type != "all" and "MF Type" in quantile_df.columns:
        quantile_df = quantile_df[(quantile_df["MF Type"].fillna("") == mf_type) | (quantile_df["MF Type"].fillna("") == "")]
    for _, row in quantile_df.iterrows():
        fuzzification = row.get("Fuzzification", "Baseline")
        if pd.isna(fuzzification):
            fuzzification = "Baseline"
        rows.append({
            col: row.get(col, np.nan)
            for col in COMPARISON_COLUMNS
        })
        rows[-1]["Fuzzification"] = fuzzification

    uniform_path = "reports/results/final_all_models_summary.csv"
    if os.path.exists(uniform_path):
        uniform_df = pd.read_csv(uniform_path)
        uniform_df = uniform_df[uniform_df["Model"].isin(["Sugeno V1 Label-Level", "Sugeno V2 Full Rule-Level"])]
        for _, row in uniform_df.iterrows():
            rows.append({
                "Dataset": row.get("Dataset", ""),
                "LLM": row.get("LLM", ""),
                "Model": row.get("Model", ""),
                "Fuzzification": "Uniform",
                "MF Type": "uniform",
                "Rules": row.get("Rules", np.nan),
                "Inputs": row.get("Inputs", np.nan),
                "Parameters": row.get("Parameters", np.nan),
                "Train RMSE": row.get("Train RMSE", np.nan),
                "Test RMSE": row.get("Test RMSE", np.nan),
                "Test MAE": row.get("Test MAE", np.nan),
                "Test MAPE (%)": row.get("Test MAPE (%)", np.nan),
                "Test R2": row.get("Test R2", np.nan),
                "CV RMSE": row.get("CV RMSE", np.nan),
                "CV MAE": row.get("CV MAE", np.nan),
                "CV MAPE (%)": row.get("CV MAPE (%)", np.nan),
                "CV R2": row.get("CV R2", np.nan),
            })

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return None, []

    comparison_df = comparison_df.reindex(columns=COMPARISON_COLUMNS)
    output_path = "reports/results/final_uniform_vs_quantile_comparison.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    figure_paths = save_uniform_vs_quantile_plots(comparison_df)
    return output_path, figure_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run quantile-based Sugeno V1 label-level models.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--llms", nargs="+", default=LLMS, choices=LLMS)
    parser.add_argument("--mf-type", default="all", choices=[*MF_TYPES, "all"])
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = []
    for mf_type in resolve_mf_types(args.mf_type):
        set_output_dirs(mf_type)
        mf_rows = []
        for dataset_name in args.datasets:
            for llm_name in args.llms:
                rows = run_single(dataset_name, llm_name, mf_type)
                mf_rows.extend(rows)
                all_rows.extend(rows)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        mf_summary_path = os.path.join(RESULTS_DIR, "final_quantile_all_models_summary.csv")
        pd.DataFrame(mf_rows).to_csv(mf_summary_path, index=False)
        print(f"[OK] {mf_type} V1 summary saved to {mf_summary_path}")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    summary_path = os.path.join(RESULTS_ROOT, "final_quantile_all_models_summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    comparison_path, figure_paths = save_uniform_vs_quantile_comparison(args.mf_type)

    print(f"\n[OK] Quantile Sugeno V1 run completed. Summary saved to {summary_path}")
    if comparison_path:
        print(f"[OK] Uniform vs quantile comparison saved to {comparison_path}")
        print(f"[OK] Uniform vs quantile plots saved to: {', '.join(figure_paths)}")
    else:
        print("[INFO] Uniform vs quantile comparison was not created because no quantile summary was available.")


if __name__ == "__main__":
    main()
