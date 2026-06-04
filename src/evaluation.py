import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.sugeno_model import SugenoEffortModel

def mean_absolute_percentage_error(y_true, y_pred):
    """ MAPE (Mean Absolute Percentage Error) Hesaplar """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Sıfıra bölmeyi engelle
    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def evaluate_models(dataset_name, data_path, input_vars):
    print(f"\n{'='*40}")
    print(f"EVALUATION FOR {dataset_name.upper()} DATASET")
    print(f"{'='*40}")
    
    # Veriyi Yükle
    df = pd.read_csv(data_path)
    X = df[input_vars]
    y = df["Effort"].values

    results = []

    # 1. Baseline Modeller
    # 1.1 Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_preds = lr_model.predict(X)
    
    results.append({
        "Model": "Linear Regression",
        "RMSE": np.sqrt(mean_squared_error(y, lr_preds)),
        "MAE": mean_absolute_error(y, lr_preds),
        "MAPE (%)": mean_absolute_percentage_error(y, lr_preds)
    })

    # 1.2 Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X, y)
    dt_preds = dt_model.predict(X)
    
    results.append({
        "Model": "Decision Tree",
        "RMSE": np.sqrt(mean_squared_error(y, dt_preds)),
        "MAE": mean_absolute_error(y, dt_preds),
        "MAPE (%)": mean_absolute_percentage_error(y, dt_preds)
    })

    # 2. Sugeno Fuzzy Modeller (3 Farklı LLM Kuralı İle)
    llms = ["gemini", "chatgpt", "claude"]
    for llm in llms:
        try:
            fuzzy_model = SugenoEffortModel(dataset_name, llm)
            # Ağırlıkları optimize et (eğit)
            fuzzy_model.train(data_path)
            # Tahmin yap
            fuzzy_preds = fuzzy_model.predict(X)
            
            results.append({
                "Model": f"Sugeno FIS ({llm.capitalize()})",
                "RMSE": np.sqrt(mean_squared_error(y, fuzzy_preds)),
                "MAE": mean_absolute_error(y, fuzzy_preds),
                "MAPE (%)": mean_absolute_percentage_error(y, fuzzy_preds)
            })
        except Exception as e:
            print(f"[HATA] {llm} modeli {dataset_name} için değerlendirilemedi: {e}")

    # Sonuçları DataFrame'e çevir ve ekrana bas
    results_df = pd.DataFrame(results)
    
    # Raporlama için kaydet
    import os
    os.makedirs("reports/results", exist_ok=True)
    results_df.to_csv(f"reports/results/{dataset_name}_evaluation.csv", index=False)
    
    print("\nPERFORMANS SONUÇLARI:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return results_df

if __name__ == "__main__":
    # Albrecht Değerlendirmesi
    albrecht_path = "data/processed_data/final_normalized/albrecht_normalized.csv"
    albrecht_vars = ["RawFPcounts", "Input", "File"]
    evaluate_models("albrecht", albrecht_path, albrecht_vars)

    # Desharnais Değerlendirmesi
    desharnais_path = "data/processed_data/final_normalized/desharnais_normalized.csv"
    desharnais_vars = ["PointsAjust", "TeamExp", "Length"]
    evaluate_models("desharnais", desharnais_path, desharnais_vars)
