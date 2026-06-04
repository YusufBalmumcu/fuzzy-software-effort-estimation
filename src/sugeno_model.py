import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from simpful import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import os

from src.fuzzy_design import setup_albrecht_fuzzy_system, setup_desharnais_fuzzy_system
from src.rules import get_all_rules

class SugenoEffortModel:
    def __init__(self, dataset_name, llm_name="gemini"):
        self.dataset_name = dataset_name.lower()
        self.llm_name = llm_name.lower()
        
        # Sistemi kur
        if self.dataset_name == "albrecht":
            self.FS = setup_albrecht_fuzzy_system()
            self.input_vars = ["RawFPcounts", "Input", "File"]
        elif self.dataset_name == "desharnais":
            self.FS = setup_desharnais_fuzzy_system()
            self.input_vars = ["PointsAjust", "TeamExp", "Length"]
        else:
            raise ValueError("Bilinmeyen veri seti")

        # Çıktı terimleri (Consequents)
        self.output_terms = ["Very_Low", "Low", "Medium", "High", "Very_High"]
        
        # Kuralları yükle
        rules_text = get_all_rules(self.dataset_name, self.llm_name)
        if not rules_text:
            raise ValueError(f"{self.llm_name}_{self.dataset_name} için kural bulunamadı!")
            
        self.FS.add_rules(rules_text)
        
        # 1. Derece Sugeno için parametre sayısı: Her terim için (girdi sayısı + 1)
        self.params_per_term = len(self.input_vars) + 1
        self.total_params = len(self.output_terms) * self.params_per_term

    def set_output_functions(self, weights):
        """ Optimizasyon sırasında ağırlıkları sisteme tanımlar. """
        idx = 0
        for term in self.output_terms:
            # Örn 3 girdi için: p*x + q*y + r*z + s
            # weights dizisinden o terime ait ağırlıkları çek
            term_weights = weights[idx : idx + self.params_per_term]
            
            # Fonksiyonu string olarak oluştur
            func_str = ""
            for i, var in enumerate(self.input_vars):
                func_str += f"{term_weights[i]}*{var} + "
            func_str += f"{term_weights[-1]}" # Bias (s)
            
            self.FS.set_output_function(term, func_str)
            idx += self.params_per_term

    def predict(self, X):
        """ X matrisi üzerinden tahmin yapar. """
        predictions = []
        for _, row in X.iterrows():
            for var in self.input_vars:
                self.FS.set_variable(var, row[var])
            
            try:
                # Sugeno çıkarımını çalıştır
                res = self.FS.Sugeno_inference(["Effort"])
                predictions.append(res["Effort"])
            except Exception as e:
                # Kural ateşlenmezse veya hata olursa 0 ata
                predictions.append(0.0)
                
        return np.array(predictions)

    def _objective_function(self, weights, X, y):
        """ Optimizasyon algoritması için hata (RMSE) hesaplar. """
        self.set_output_functions(weights)
        preds = self.predict(X)
        rmse = np.sqrt(np.mean((y - preds) ** 2))
        return rmse

    def train(self, data_path):
        """
        Veriyi okur, girdi ve çıktıları ayırır ve Scipy kullanarak
        1. Derece Sugeno lineer denklemlerindeki ağırlıkları öğrenir (optimize eder).
        """
        df = pd.read_csv(data_path)
        X = df[self.input_vars]
        y = df["Effort"].values # Veya veri setindeki effort sütununun adı
        
        print(f"[{self.dataset_name.upper()} - {self.llm_name.upper()}] Ağırlıklar öğreniliyor (Optimizasyon)...")
        
        # Başlangıç ağırlıkları: Hepsi 1.0 olsun (Veya rastgele)
        initial_weights = np.ones(self.total_params)
        
        # Optimizasyon (L-BFGS-B hızlı ve etkilidir)
        res = minimize(self._objective_function, initial_weights, args=(X, y), method='L-BFGS-B', options={'maxiter': 50})
        
        self.optimized_weights = res.x
        self.set_output_functions(self.optimized_weights)
        print(f"Optimizasyon tamamlandı! Eğitim RMSE: {res.fun:.4f}")

    def plot_surface(self):
        """
        Girdilerin (ilk iki girdinin) Output (Effort) üzerindeki etkisini 
        3D Surface (Yüzey) Grafiği ile gösterir.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        var1 = self.input_vars[0]
        var2 = self.input_vars[1]
        
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        X_grid, Y_grid = np.meshgrid(x, y)
        Z_grid = np.zeros_like(X_grid)
        
        # Üçüncü değişken varsa sabit tut (örneğin 0.5 ortalama değerinde)
        fixed_val = 0.5 
        
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                self.FS.set_variable(var1, X_grid[i, j])
                self.FS.set_variable(var2, Y_grid[i, j])
                if len(self.input_vars) > 2:
                    self.FS.set_variable(self.input_vars[2], fixed_val)
                    
                res = self.FS.Sugeno_inference(["Effort"])
                Z_grid[i, j] = res.get("Effort", 0)
                
        ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis')
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel('Effort')
        ax.set_title(f'Sugeno Output Surface ({self.dataset_name.capitalize()} - {self.llm_name.capitalize()})')
        
        os.makedirs("reports/figures", exist_ok=True)
        save_path = f"reports/figures/surface_{self.dataset_name}_{self.llm_name}.png"
        plt.savefig(save_path)
        print(f"[SAVED] Yüzey grafiği kaydedildi: {save_path}")

if __name__ == "__main__":
    # Test
    model = SugenoEffortModel("albrecht", "gemini")
    model.train("data/processed_data/final_normalized/albrecht_normalized.csv")
    model.plot_surface()
