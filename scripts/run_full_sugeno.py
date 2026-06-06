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

    print("[INFO] İşlenmiş veri dosyaları eksik. main.py ile preprocessing çalıştırılıyor.")
    from main import main as run_preprocessing

    run_preprocessing()

    still_missing = [path for path in missing if not os.path.exists(path)]
    if still_missing:
        raise FileNotFoundError(f"Preprocessing sonrasi hala eksik dosyalar var: {still_missing}")


def split_dataset(df, target_col):
    test_size = 0.30 if len(df) < 50 else 0.20
    return train_test_split(df, test_size=test_size, random_state=42, shuffle=True)


def cross_validate_full_sugeno(dataset_name, llm_name, df, n_splits=5):
    """
    Albrecht az satirli oldugu icin CV sonucu yuksek varyansli olabilir.
    Yine de egitim/test ayrimina ek olarak modelin genel davranisini gormek icin kaydedilir.
    """
    n_splits = min(n_splits, len(df))
    if n_splits < 2:
        return {}

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df), start=1):
        model = FullRuleSugenoModel(dataset_name, llm_name)
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


def save_combined_predictions(model, train_df, test_df):
    train_pred_path = model.save_predictions(train_df, "train")
    test_pred_path = model.save_predictions(test_df, "test")
    return train_pred_path, test_pred_path


def _metric_column(prefix, metric_name):
    return f"{prefix} {metric_name}"


def _slug_model_name(model_name):
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
    )


def _parse_llm_from_old_model_name(model_name):
    if "(" not in model_name or ")" not in model_name:
        return ""

    llm = model_name.split("(", 1)[1].split(")", 1)[0].strip().lower()
    return "gpt" if llm == "chatgpt" else llm


def save_model_comparison_outputs(model, train_df, test_df, baseline_predictions):
    """
    Full Sugeno, Linear Regression ve Decision Tree modellerini ayni train/test ayriminda karsilastirir.
    Metrikler orijinal Effort olceginde hesaplanir.
    """
    os.makedirs("reports/results", exist_ok=True)
    os.makedirs("reports/predictions", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    train_actual = train_df[model.target_col].to_numpy(dtype=float)
    test_actual = test_df[model.target_col].to_numpy(dtype=float)

    model_predictions = {
        f"Full Rule-Level Sugeno ({model.output_llm_name.capitalize()})": {
            "train": model.predict(train_df),
            "test": model.predict(test_df),
        },
        **baseline_predictions,
    }

    comparison_rows = []
    for model_name, preds in model_predictions.items():
        train_metrics = regression_metrics(train_actual, preds["train"])
        test_metrics = regression_metrics(test_actual, preds["test"])
        row = {"Model": model_name}
        for metric_name, value in train_metrics.items():
            row[_metric_column("Train", metric_name)] = value
        for metric_name, value in test_metrics.items():
            row[_metric_column("Test", metric_name)] = value
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = f"reports/results/{model.dataset_name}_{model.output_llm_name}_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    prediction_paths = {}
    for split_name, split_df, actual in [
        ("train", train_df, train_actual),
        ("test", test_df, test_actual),
    ]:
        prediction_df = split_df[model.input_vars].copy()
        prediction_df["split"] = split_name
        prediction_df["actual"] = actual

        for model_name, preds in model_predictions.items():
            slug = _slug_model_name(model_name)
            prediction_df[f"{slug}_predicted"] = preds[split_name]
            prediction_df[f"{slug}_residual"] = actual - preds[split_name]

        path = f"reports/predictions/{model.dataset_name}_{model.output_llm_name}_model_comparison_{split_name}.csv"
        prediction_df.to_csv(path, index=False)
        prediction_paths[split_name] = path

    figure_paths = {}
    metric_names = ["RMSE", "MAE", "MAPE (%)", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric_name in zip(axes.ravel(), metric_names):
        column = _metric_column("Test", metric_name)
        ax.bar(comparison_df["Model"], comparison_df[column])
        if metric_name == "R2":
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_title("Test R2 (higher is better)")
        else:
            ax.set_title(f"Test {metric_name} (lower is better)")
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle(f"Model Comparison - {model.dataset_name} {model.output_llm_name}")
    fig.tight_layout()
    metrics_fig_path = f"reports/figures/{model.dataset_name}_{model.output_llm_name}_model_comparison_metrics.png"
    fig.savefig(metrics_fig_path, bbox_inches="tight")
    plt.close(fig)
    figure_paths["metrics"] = metrics_fig_path

    fig, axes = plt.subplots(1, len(model_predictions), figsize=(5 * len(model_predictions), 4))
    if len(model_predictions) == 1:
        axes = [axes]
    min_val = min([test_actual.min()] + [preds["test"].min() for preds in model_predictions.values()])
    max_val = max([test_actual.max()] + [preds["test"].max() for preds in model_predictions.values()])
    for ax, (model_name, preds) in zip(axes, model_predictions.items()):
        ax.scatter(test_actual, preds["test"], alpha=0.8)
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_title(model_name)
        ax.set_xlabel("Actual Effort")
        ax.set_ylabel("Predicted Effort")
    fig.suptitle(f"Predicted vs Actual Comparison - {model.dataset_name} {model.output_llm_name}")
    fig.tight_layout()
    pred_fig_path = f"reports/figures/{model.dataset_name}_{model.output_llm_name}_model_comparison_predicted_vs_actual.png"
    fig.savefig(pred_fig_path, bbox_inches="tight")
    plt.close(fig)
    figure_paths["predicted_vs_actual"] = pred_fig_path

    fig, axes = plt.subplots(1, len(model_predictions), figsize=(5 * len(model_predictions), 4))
    if len(model_predictions) == 1:
        axes = [axes]
    for ax, (model_name, preds) in zip(axes, model_predictions.items()):
        residuals = test_actual - preds["test"]
        ax.scatter(preds["test"], residuals, alpha=0.8)
        ax.axhline(0.0, color="r", linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("Predicted Effort")
        ax.set_ylabel("Residual")
    fig.suptitle(f"Residual Comparison - {model.dataset_name} {model.output_llm_name}")
    fig.tight_layout()
    residual_fig_path = f"reports/figures/{model.dataset_name}_{model.output_llm_name}_model_comparison_residuals.png"
    fig.savefig(residual_fig_path, bbox_inches="tight")
    plt.close(fig)
    figure_paths["residuals"] = residual_fig_path

    return comparison_path, prediction_paths, figure_paths, comparison_df


def load_current_comparison_records(summary_df):
    records = []
    seen_baselines = set()

    for _, summary_row in summary_df.iterrows():
        comparison_path = summary_row.get("Model Comparison")
        if not isinstance(comparison_path, str) or not os.path.exists(comparison_path):
            continue

        comparison_df = pd.read_csv(comparison_path)
        dataset_name = summary_row["Dataset"]
        llm_name = summary_row["LLM"]

        for _, row in comparison_df.iterrows():
            model_name = row["Model"]

            if model_name.startswith("Full Rule-Level Sugeno"):
                model_type = "Full Rule-Level Sugeno"
                output_llm = llm_name
            else:
                # Linear Regression ve Decision Tree her LLM icin ayni split ile tekrarlandigi icin
                # genel karsilastirmada her veri seti icin bir kez tutulur.
                baseline_key = (dataset_name, model_name)
                if baseline_key in seen_baselines:
                    continue
                seen_baselines.add(baseline_key)
                model_type = "ML Baseline"
                output_llm = ""

            records.append({
                "Dataset": dataset_name,
                "LLM": output_llm,
                "Model": model_name,
                "Model Type": model_type,
                "Evaluation Scope": "current_test_split",
                "Train RMSE": row.get("Train RMSE", np.nan),
                "Train MAE": row.get("Train MAE", np.nan),
                "Train MAPE (%)": row.get("Train MAPE (%)", np.nan),
                "Train R2": row.get("Train R2", np.nan),
                "Test RMSE": row.get("Test RMSE", np.nan),
                "Test MAE": row.get("Test MAE", np.nan),
                "Test MAPE (%)": row.get("Test MAPE (%)", np.nan),
                "Test R2": row.get("Test R2", np.nan),
                "Source": comparison_path,
            })

    return records


def load_old_sugeno_records():
    records = []

    for dataset_name in DATASETS:
        old_path = f"reports/results/{dataset_name}_evaluation.csv"
        if not os.path.exists(old_path):
            continue

        old_df = pd.read_csv(old_path)
        for _, row in old_df.iterrows():
            model_name = row["Model"]
            if not model_name.startswith("Sugeno FIS"):
                continue

            llm_name = _parse_llm_from_old_model_name(model_name)
            records.append({
                "Dataset": dataset_name,
                "LLM": llm_name,
                "Model": f"Old Label-Level Sugeno ({llm_name.capitalize()})",
                "Model Type": "Old Label-Level Sugeno",
                "Evaluation Scope": "old_full_dataset_in_sample",
                "Train RMSE": np.nan,
                "Train MAE": np.nan,
                "Train MAPE (%)": np.nan,
                "Train R2": np.nan,
                "Test RMSE": row.get("RMSE", np.nan),
                "Test MAE": row.get("MAE", np.nan),
                "Test MAPE (%)": row.get("MAPE (%)", np.nan),
                "Test R2": np.nan,
                "Source": old_path,
            })

    return records


def _display_label(row):
    if row["LLM"]:
        return f"{row['Model Type']} ({row['LLM']})"
    return row["Model"]


def save_aggregate_metric_figure(all_df, metric_name, output_path):
    column = f"Test {metric_name}"
    plot_df = all_df.dropna(subset=[column]).copy()
    if plot_df.empty:
        return None

    datasets = list(plot_df["Dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(9 * len(datasets), 7), squeeze=False)

    for ax, dataset_name in zip(axes.ravel(), datasets):
        dataset_df = plot_df[plot_df["Dataset"] == dataset_name].copy()
        dataset_df["Display Label"] = dataset_df.apply(_display_label, axis=1)
        ascending = metric_name != "R2"
        dataset_df = dataset_df.sort_values(column, ascending=ascending)

        colors = dataset_df["Model Type"].map({
            "Full Rule-Level Sugeno": "#4C78A8",
            "ML Baseline": "#F58518",
            "Old Label-Level Sugeno": "#54A24B",
        }).fillna("#777777")

        ax.barh(dataset_df["Display Label"], dataset_df[column], color=colors)
        if metric_name == "R2":
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.set_title(f"{dataset_name.capitalize()} - Test R2\nHigher is better; negative is worse than mean")
        else:
            ax.set_title(f"{dataset_name.capitalize()} - Test {metric_name}\nLower is better")
        ax.set_xlabel(column)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_old_vs_full_sugeno_figures(all_df):
    sugeno_df = all_df[
        all_df["Model Type"].isin(["Full Rule-Level Sugeno", "Old Label-Level Sugeno"])
        & all_df["LLM"].astype(bool)
    ].copy()
    if sugeno_df.empty:
        return []

    output_paths = []
    metrics = ["RMSE", "MAE", "MAPE (%)"]
    datasets = list(sugeno_df["Dataset"].drop_duplicates())

    for metric_name in metrics:
        column = f"Test {metric_name}"
        fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), squeeze=False)

        for ax, dataset_name in zip(axes.ravel(), datasets):
            dataset_df = sugeno_df[sugeno_df["Dataset"] == dataset_name].dropna(subset=[column]).copy()
            pivot_df = dataset_df.pivot_table(
                index="LLM",
                columns="Model Type",
                values=column,
                aggfunc="first",
            ).reindex(index=LLMS)

            pivot_df.plot(kind="bar", ax=ax)
            ax.set_title(f"{dataset_name.capitalize()} - {metric_name}")
            ax.set_xlabel("LLM")
            ax.set_ylabel(metric_name)
            ax.tick_params(axis="x", labelrotation=0)

        fig.suptitle(
            "Old label-level Sugeno vs full rule-level Sugeno\n"
            "Old values are from existing full-dataset evaluation CSVs; current values use the test split."
        )
        fig.tight_layout()
        output_path = f"reports/figures/old_vs_full_sugeno_{_slug_model_name(metric_name)}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_aggregate_model_comparisons(summary_df):
    os.makedirs("reports/results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    records = load_current_comparison_records(summary_df)
    records.extend(load_old_sugeno_records())

    all_df = pd.DataFrame(records)
    output_csv = "reports/results/all_models_comparison.csv"
    all_df.to_csv(output_csv, index=False)

    figure_paths = []
    for metric_name in ["RMSE", "MAE", "MAPE (%)", "R2"]:
        output_path = f"reports/figures/all_models_{_slug_model_name(metric_name)}_comparison.png"
        saved_path = save_aggregate_metric_figure(all_df, metric_name, output_path)
        if saved_path:
            figure_paths.append(saved_path)

    figure_paths.extend(save_old_vs_full_sugeno_figures(all_df))
    return output_csv, figure_paths


def run_single(dataset_name, llm_name):
    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset_name.capitalize()} | LLM: {llm_name.capitalize()}")
    print(f"{'=' * 72}")

    model = FullRuleSugenoModel(dataset_name, llm_name)
    df = model.load_training_frame()
    train_df, test_df = split_dataset(df, model.target_col)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    baselines_df, baseline_predictions = predict_with_baselines(train_df, test_df, model.input_vars, model.target_col)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_df)
        for warning in caught:
            print(f"[UYARI] {warning.message}")

    train_metrics = model.evaluate(train_df)
    test_metrics = model.evaluate(test_df)
    cv_metrics = cross_validate_full_sugeno(dataset_name, llm_name, df)

    equations_json, equations_txt = model.save_equations()
    train_pred_path, test_pred_path = save_combined_predictions(model, train_df, test_df)
    rule_contrib_path, dominant_rules_path = model.save_rule_analysis(df, split_name="all")
    plot_paths = model.save_plots(train_df, test_df)
    comparison_path, comparison_prediction_paths, comparison_figure_paths, comparison_df = save_model_comparison_outputs(
        model,
        train_df,
        test_df,
        baseline_predictions,
    )

    os.makedirs("reports/results", exist_ok=True)
    baseline_path = f"reports/results/{dataset_name}_{model.output_llm_name}_baselines.csv"
    baselines_df.to_csv(baseline_path, index=False)

    result_row = {
        "Dataset": dataset_name,
        "LLM": model.output_llm_name,
        "Rules": len(model.rules),
        "Inputs": len(model.input_vars),
        "Parameters": model.total_params,
        "Train RMSE": train_metrics["RMSE"],
        "Test RMSE": test_metrics["RMSE"],
        "Test MAE": test_metrics["MAE"],
        "Test MAPE (%)": test_metrics["MAPE (%)"],
        "Test R2": test_metrics["R2"],
        **cv_metrics,
        "Equations JSON": equations_json,
        "Equations TXT": equations_txt,
        "Train Predictions": train_pred_path,
        "Test Predictions": test_pred_path,
        "Rule Contributions": rule_contrib_path,
        "Dominant Rules": dominant_rules_path,
        "Baseline Results": baseline_path,
        "Model Comparison": comparison_path,
        "Comparison Train Predictions": comparison_prediction_paths["train"],
        "Comparison Test Predictions": comparison_prediction_paths["test"],
        "Comparison Metrics Figure": comparison_figure_paths["metrics"],
        "Comparison Predicted Figure": comparison_figure_paths["predicted_vs_actual"],
        "Comparison Residual Figure": comparison_figure_paths["residuals"],
    }

    print("\nÖZET")
    print(f"Dataset: {dataset_name.capitalize()}")
    print(f"LLM: {model.output_llm_name.capitalize()}")
    print(f"Rules: {len(model.rules)}")
    print(f"Inputs: {len(model.input_vars)}")
    print(f"Parameters: {model.total_params}")
    print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"MAE: {test_metrics['MAE']:.4f}")
    print(f"MAPE: {test_metrics['MAPE (%)']:.2f}")
    print(f"R2: {test_metrics['R2']:.4f}")
    if cv_metrics:
        print(f"CV RMSE: {cv_metrics['CV RMSE']:.4f}")
    print(f"Saved equations to: {equations_json}")
    print(f"Saved plots to: {', '.join(plot_paths.values())}")
    print(f"Saved model comparison to: {comparison_path}")
    print(f"Saved comparison figures to: {', '.join(comparison_figure_paths.values())}")
    print(f"Saved rule analysis to: {rule_contrib_path}, {dominant_rules_path}")
    print("\nMODEL KARSILASTIRMASI:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return result_row


def main():
    ensure_processed_data()

    all_results = []
    for dataset_name in DATASETS:
        for llm_name in LLMS:
            all_results.append(run_single(dataset_name, llm_name))

    summary_df = pd.DataFrame(all_results)
    summary_path = "reports/results/full_sugeno_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    aggregate_csv, aggregate_figures = save_aggregate_model_comparisons(summary_df)

    print(f"\n[OK] Full rule-level Sugeno run completed. Summary saved to {summary_path}")
    print(f"[OK] Aggregate model comparison saved to {aggregate_csv}")
    print(f"[OK] Aggregate comparison figures saved to: {', '.join(aggregate_figures)}")


if __name__ == "__main__":
    main()
