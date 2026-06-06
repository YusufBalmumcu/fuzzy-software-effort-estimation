import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.evaluation import evaluate_baselines, regression_metrics
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


def run_single(dataset_name, llm_name):
    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset_name.capitalize()} | LLM: {llm_name.capitalize()}")
    print(f"{'=' * 72}")

    model = FullRuleSugenoModel(dataset_name, llm_name)
    df = model.load_training_frame()
    train_df, test_df = split_dataset(df, model.target_col)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    baselines_df = evaluate_baselines(train_df, test_df, model.input_vars, model.target_col)

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
    print(f"Saved rule analysis to: {rule_contrib_path}, {dominant_rules_path}")

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

    print(f"\n[OK] Full rule-level Sugeno run completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
