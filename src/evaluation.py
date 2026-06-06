import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE (Mean Absolute Percentage Error) hesaplar."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def regression_metrics(y_true, y_pred):
    """RMSE, MAE, MAPE ve R2 metriklerini tek yerde hesaplar."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
    }


def evaluate_baselines(train_df, test_df, input_vars, target_col):
    """Linear Regression ve Decision Tree baseline modellerini egitir ve test metriklerini dondurur."""
    X_train = train_df[input_vars]
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_test = test_df[input_vars]
    y_test = test_df[target_col].to_numpy(dtype=float)

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
    ]

    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        row = {"Model": model_name}
        row.update(regression_metrics(y_test, preds))
        results.append(row)

    return pd.DataFrame(results)


def evaluate_models(dataset_name, data_path, input_vars):
    """
    Eski etiket seviyeli Sugeno modelini koruyan değerlendirme fonksiyonu.
    Yeni tam kural seviyeli Sugeno için scripts/run_full_sugeno.py kullanılmalıdır.
    """
    print(f"\n{'=' * 40}")
    print(f"EVALUATION FOR {dataset_name.upper()} DATASET")
    print(f"{'=' * 40}")

    df = pd.read_csv(data_path)
    target_col = "Effort"
    X = df[input_vars]
    y = df[target_col].values

    results = []

    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_preds = lr_model.predict(X)
    row = {"Model": "Linear Regression"}
    row.update(regression_metrics(y, lr_preds))
    results.append(row)

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X, y)
    dt_preds = dt_model.predict(X)
    row = {"Model": "Decision Tree"}
    row.update(regression_metrics(y, dt_preds))
    results.append(row)

    llms = ["gemini", "gpt", "claude"]
    for llm in llms:
        try:
            from src.sugeno_model import SugenoEffortModel

            fuzzy_model = SugenoEffortModel(dataset_name, llm)
            fuzzy_model.train(data_path)
            fuzzy_preds = fuzzy_model.predict(X)

            row = {"Model": f"Sugeno FIS ({llm.capitalize()})"}
            row.update(regression_metrics(y, fuzzy_preds))
            results.append(row)
        except Exception as e:
            print(f"[HATA] {llm} modeli {dataset_name} için değerlendirilemedi: {e}")

    results_df = pd.DataFrame(results)

    os.makedirs("reports/results", exist_ok=True)
    results_df.to_csv(f"reports/results/{dataset_name}_evaluation.csv", index=False)

    print("\nPERFORMANS SONUÇLARI:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return results_df


if __name__ == "__main__":
    albrecht_path = "data/processed_data/final_normalized/albrecht_normalized.csv"
    albrecht_vars = ["RawFPcounts", "Input", "File"]
    evaluate_models("albrecht", albrecht_path, albrecht_vars)

    desharnais_path = "data/processed_data/final_normalized/desharnais_normalized.csv"
    desharnais_vars = ["PointsAjust", "TeamExp", "Length"]
    evaluate_models("desharnais", desharnais_path, desharnais_vars)
