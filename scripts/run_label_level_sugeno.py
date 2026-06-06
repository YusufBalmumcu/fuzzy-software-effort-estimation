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
from src.full_sugeno_model import DATASET_CONFIG
from src.label_level_sugeno_model import LabelLevelSugenoModel


DATASETS = ["albrecht", "desharnais"]
LLMS = ["gemini", "gpt", "claude"]


def ensure_processed_data():
    missing = []
    for config in DATASET_CONFIG.values():
        for path_key in ["normalized_path", "original_path"]:
            if not os.path.exists(config[path_key]):
                missing.append(config[path_key])

    if not missing:
        return

    print("[INFO] Islenmis veri dosyalari eksik. main.py ile preprocessing calistiriliyor.")
    from main import main as run_preprocessing

    run_preprocessing()

    still_missing = [path for path in missing if not os.path.exists(path)]
    if still_missing:
        raise FileNotFoundError(f"Preprocessing sonrasi hala eksik dosyalar var: {still_missing}")


def split_dataset(df):
    test_size = 0.30 if len(df) < 50 else 0.20
    return train_test_split(df, test_size=test_size, random_state=42, shuffle=True)


def cross_validate_label_sugeno(dataset_name, llm_name, df, n_splits=5):
    n_splits = min(n_splits, len(df))
    if n_splits < 2:
        return {}

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df), start=1):
        model = LabelLevelSugenoModel(dataset_name, llm_name)
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


def load_v2_full_rule_row(dataset_name, llm_name):
    comparison_path = f"reports/results/{dataset_name}_{llm_name}_model_comparison.csv"
    summary_path = "reports/results/full_sugeno_summary.csv"

    if not os.path.exists(comparison_path):
        return None

    comparison_df = pd.read_csv(comparison_path)
    v2_rows = comparison_df[comparison_df["Model"].str.startswith("Full Rule-Level Sugeno")]
    if v2_rows.empty:
        return None

    row = v2_rows.iloc[0]
    output = {
        "Model": "Sugeno V2 Full Rule-Level",
        "Train RMSE": row.get("Train RMSE", np.nan),
        "Train MAE": row.get("Train MAE", np.nan),
        "Train MAPE (%)": row.get("Train MAPE (%)", np.nan),
        "Train R2": row.get("Train R2", np.nan),
        "Test RMSE": row.get("Test RMSE", np.nan),
        "Test MAE": row.get("Test MAE", np.nan),
        "Test MAPE (%)": row.get("Test MAPE (%)", np.nan),
        "Test R2": row.get("Test R2", np.nan),
        "CV RMSE": np.nan,
        "CV MAE": np.nan,
        "CV MAPE (%)": np.nan,
        "CV R2": np.nan,
    }

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        matching = summary_df[(summary_df["Dataset"] == dataset_name) & (summary_df["LLM"] == llm_name)]
        if not matching.empty:
            summary_row = matching.iloc[0]
            output["CV RMSE"] = summary_row.get("CV RMSE", np.nan)
            output["CV MAE"] = summary_row.get("CV MAE", np.nan)
            output["CV MAPE (%)"] = summary_row.get("CV MAPE (%)", np.nan)
            output["CV R2"] = summary_row.get("CV R2", np.nan)

    return output


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


def save_final_comparison_plots(model, comparison_df, prediction_payload):
    os.makedirs("reports/figures", exist_ok=True)
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
    fig.suptitle(f"Final Model Comparison - {model.dataset_name} {model.output_llm_name}")
    fig.tight_layout()
    metrics_path = f"reports/figures/{model.dataset_name}_{model.output_llm_name}_final_model_comparison_metrics.png"
    fig.savefig(metrics_path, bbox_inches="tight")
    plt.close(fig)
    paths["metrics"] = metrics_path

    test_actual = prediction_payload["actual"]
    prediction_items = prediction_payload["predictions"]

    fig, axes = plt.subplots(1, len(prediction_items), figsize=(5 * len(prediction_items), 4), squeeze=False)
    values = [test_actual]
    values.extend(pred for pred in prediction_items.values())
    min_val = min(np.min(value) for value in values)
    max_val = max(np.max(value) for value in values)
    for ax, (model_name, predicted) in zip(axes.ravel(), prediction_items.items()):
        ax.scatter(test_actual, predicted, alpha=0.8)
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_title(model_name)
        ax.set_xlabel("Actual Effort")
        ax.set_ylabel("Predicted Effort")
    fig.suptitle(f"Final Predicted vs Actual - {model.dataset_name} {model.output_llm_name}")
    fig.tight_layout()
    pred_path = f"reports/figures/{model.dataset_name}_{model.output_llm_name}_final_model_comparison_predicted_vs_actual.png"
    fig.savefig(pred_path, bbox_inches="tight")
    plt.close(fig)
    paths["predicted_vs_actual"] = pred_path

    fig, axes = plt.subplots(1, len(prediction_items), figsize=(5 * len(prediction_items), 4), squeeze=False)
    for ax, (model_name, predicted) in zip(axes.ravel(), prediction_items.items()):
        residuals = test_actual - predicted
        ax.scatter(predicted, residuals, alpha=0.8)
        ax.axhline(0.0, color="r", linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("Predicted Effort")
        ax.set_ylabel("Residual")
    fig.suptitle(f"Final Residual Comparison - {model.dataset_name} {model.output_llm_name}")
    fig.tight_layout()
    residual_path = f"reports/figures/{model.dataset_name}_{model.output_llm_name}_final_model_comparison_residuals.png"
    fig.savefig(residual_path, bbox_inches="tight")
    plt.close(fig)
    paths["residuals"] = residual_path

    return paths


def save_final_prediction_comparison(model, test_df, baseline_predictions, label_predictions):
    actual = test_df[model.target_col].to_numpy(dtype=float)
    prediction_items = {
        "Sugeno V1 Label-Level": label_predictions,
        "Linear Regression": baseline_predictions["Linear Regression"]["test"],
        "Decision Tree": baseline_predictions["Decision Tree"]["test"],
    }

    v2_prediction_path = f"reports/predictions/{model.dataset_name}_{model.output_llm_name}_model_comparison_test.csv"
    if os.path.exists(v2_prediction_path):
        v2_df = pd.read_csv(v2_prediction_path)
        v2_cols = [col for col in v2_df.columns if col.startswith("full_rule_level_sugeno") and col.endswith("_predicted")]
        if v2_cols and len(v2_df) == len(test_df):
            prediction_items["Sugeno V2 Full Rule-Level"] = v2_df[v2_cols[0]].to_numpy(dtype=float)

    out = test_df[model.input_vars].copy()
    out["actual"] = actual
    for model_name, predicted in prediction_items.items():
        slug = _slug(model_name)
        out[f"{slug}_predicted"] = predicted
        out[f"{slug}_residual"] = actual - predicted

    os.makedirs("reports/predictions", exist_ok=True)
    path = f"reports/predictions/{model.dataset_name}_{model.output_llm_name}_final_model_comparison_test.csv"
    out.to_csv(path, index=False)
    return path, {"actual": actual, "predictions": prediction_items}


def run_single(dataset_name, llm_name):
    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset_name.capitalize()} | LLM: {llm_name.capitalize()} | Sugeno V1 Label-Level")
    print(f"{'=' * 72}")

    model = LabelLevelSugenoModel(dataset_name, llm_name)
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
            print(f"[UYARI] {warning.message}")

    train_metrics = model.evaluate(train_df)
    test_metrics = model.evaluate(test_df)
    cv_metrics = cross_validate_label_sugeno(dataset_name, llm_name, df)

    equations_json, equations_txt = model.save_equations()
    train_predictions = model.save_predictions(train_df, "train")
    test_predictions = model.save_predictions(test_df, "test")
    rule_contrib_path, dominant_rules_path = model.save_rule_analysis(df, split_name="all")
    label_plot_paths = model.save_plots(train_df, test_df)

    label_row = row_from_metrics(
        "Sugeno V1 Label-Level",
        train_metrics,
        test_metrics,
        cv_metrics,
    )

    comparison_rows = [*baseline_rows, label_row]
    v2_row = load_v2_full_rule_row(dataset_name, model.output_llm_name)
    if v2_row:
        comparison_rows.append(v2_row)

    comparison_df = pd.DataFrame(comparison_rows)
    os.makedirs("reports/results", exist_ok=True)
    comparison_path = f"reports/results/{dataset_name}_{model.output_llm_name}_final_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    final_prediction_path, prediction_payload = save_final_prediction_comparison(
        model,
        test_df,
        baseline_predictions,
        model.predict(test_df),
    )
    final_plot_paths = save_final_comparison_plots(model, comparison_df, prediction_payload)

    print("\nOZET")
    print(f"Dataset: {dataset_name.capitalize()}")
    print(f"LLM: {model.output_llm_name.capitalize()}")
    print("Model: Sugeno V1 Label-Level")
    print(f"Rules: {len(model.rules)}")
    print(f"Output labels: {len(model.output_labels)}")
    print(f"Inputs: {len(model.input_vars)}")
    print(f"Parameters: {model.total_params}")
    print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test MAE: {test_metrics['MAE']:.4f}")
    print(f"Test MAPE: {test_metrics['MAPE (%)']:.2f}")
    print(f"Test R2: {test_metrics['R2']:.4f}")
    if cv_metrics:
        print(f"CV RMSE: {cv_metrics['CV RMSE']:.4f}")
    print(f"Saved equations: {equations_json}, {equations_txt}")
    print(f"Saved predictions: {train_predictions}, {test_predictions}, {final_prediction_path}")
    print(f"Saved rule analysis: {rule_contrib_path}, {dominant_rules_path}")
    print(f"Saved plots: {', '.join([*label_plot_paths.values(), *final_plot_paths.values()])}")
    print("\nFINAL MODEL KARSILASTIRMASI:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_rows = []
    for _, comparison_row in comparison_df.iterrows():
        model_name = comparison_row["Model"]
        if model_name == "Sugeno V1 Label-Level":
            rules = len(model.rules)
            parameters = model.total_params
            equations_json_value = equations_json
            equations_txt_value = equations_txt
            train_predictions_value = train_predictions
            test_predictions_value = test_predictions
            rule_contrib_value = rule_contrib_path
            dominant_rules_value = dominant_rules_path
        elif model_name == "Sugeno V2 Full Rule-Level":
            rules = len(model.rules)
            parameters = len(model.rules) * (len(model.input_vars) + 1)
            equations_json_value = f"models/sugeno_equations/{dataset_name}_{model.output_llm_name}_equations.json"
            equations_txt_value = f"models/sugeno_equations/{dataset_name}_{model.output_llm_name}_equations.txt"
            train_predictions_value = f"reports/predictions/{dataset_name}_{model.output_llm_name}_train_predictions.csv"
            test_predictions_value = f"reports/predictions/{dataset_name}_{model.output_llm_name}_test_predictions.csv"
            rule_contrib_value = f"reports/rule_analysis/{dataset_name}_{model.output_llm_name}_rule_contributions.csv"
            dominant_rules_value = f"reports/rule_analysis/{dataset_name}_{model.output_llm_name}_dominant_rules.csv"
        elif model_name == "Linear Regression":
            rules = 0
            parameters = len(model.input_vars) + 1
            equations_json_value = ""
            equations_txt_value = ""
            train_predictions_value = ""
            test_predictions_value = ""
            rule_contrib_value = ""
            dominant_rules_value = ""
        else:
            rules = 0
            parameters = np.nan
            equations_json_value = ""
            equations_txt_value = ""
            train_predictions_value = ""
            test_predictions_value = ""
            rule_contrib_value = ""
            dominant_rules_value = ""

        summary_rows.append({
            "Dataset": dataset_name,
            "LLM": model.output_llm_name,
            "Model": model_name,
            "Rules": rules,
            "Inputs": len(model.input_vars),
            "Parameters": parameters,
            "Train RMSE": comparison_row.get("Train RMSE", np.nan),
            "Test RMSE": comparison_row.get("Test RMSE", np.nan),
            "Test MAE": comparison_row.get("Test MAE", np.nan),
            "Test MAPE (%)": comparison_row.get("Test MAPE (%)", np.nan),
            "Test R2": comparison_row.get("Test R2", np.nan),
            "CV RMSE": comparison_row.get("CV RMSE", np.nan),
            "CV MAE": comparison_row.get("CV MAE", np.nan),
            "CV MAPE (%)": comparison_row.get("CV MAPE (%)", np.nan),
            "CV R2": comparison_row.get("CV R2", np.nan),
            "Equations JSON": equations_json_value,
            "Equations TXT": equations_txt_value,
            "Train Predictions": train_predictions_value,
            "Test Predictions": test_predictions_value,
            "Rule Contributions": rule_contrib_value,
            "Dominant Rules": dominant_rules_value,
            "Final Model Comparison": comparison_path,
        })

    return summary_rows


def main():
    ensure_processed_data()

    all_results = []
    for dataset_name in DATASETS:
        for llm_name in LLMS:
            all_results.extend(run_single(dataset_name, llm_name))

    summary_df = pd.DataFrame(all_results)
    summary_path = "reports/results/final_all_models_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Sugeno V1 label-level run completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
