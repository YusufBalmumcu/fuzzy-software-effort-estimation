import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.manual_sugeno_engine import ManualSugenoEngine
from src.rule_converter import convert_rules_to_rule_level
from src.rules import get_all_rules


DATASET_CONFIG = {
    "albrecht": {
        "input_vars": ["RawFPcounts", "Input", "File"],
        "target_col": "Effort",
        "normalized_path": "data/processed_data/final_normalized/albrecht_normalized.csv",
        "original_path": "data/processed_data/outlier_removed/albrecht_outlier_removed.csv",
    },
    "desharnais": {
        "input_vars": ["PointsAjust", "TeamExp", "Length"],
        "target_col": "Effort",
        "normalized_path": "data/processed_data/final_normalized/desharnais_normalized.csv",
        "original_path": "data/processed_data/outlier_removed/desharnais_outlier_removed.csv",
    },
}

LLM_NAME_ALIASES = {
    "gpt": "chatgpt",
    "chatgpt": "chatgpt",
    "gemini": "gemini",
    "claude": "claude",
}


class FullRuleSugenoModel:
    """
    Tam kural seviyeli birinci derece Sugeno modeli.

    Eski SugenoEffortModel her cikti etiketi icin tek denklem ogreniyordu.
    Bu sinif her kural icin ayri bir R_i_OUT terimi ve ayri denklem ogrenir.
    20 kural ve 3 girdi varsa toplam 20 * (3 + 1) = 80 parametre vardir.
    """

    def __init__(self, dataset_name, llm_name="gemini", regularization=1e-6):
        self.dataset_name = dataset_name.lower()
        self.llm_name = llm_name.lower()
        self.rule_file_llm_name = LLM_NAME_ALIASES.get(self.llm_name, self.llm_name)
        self.output_llm_name = "gpt" if self.rule_file_llm_name == "chatgpt" else self.rule_file_llm_name
        self.regularization = regularization

        if self.dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")

        self.config = DATASET_CONFIG[self.dataset_name]
        self.input_vars = self.config["input_vars"]
        self.target_col = self.config["target_col"]

        raw_rules = get_all_rules(self.dataset_name, self.rule_file_llm_name)
        if not raw_rules:
            raise ValueError(f"{self.dataset_name}/{self.rule_file_llm_name} icin kural bulunamadi.")

        self.rules = convert_rules_to_rule_level(raw_rules, self.input_vars)
        self.engine = ManualSugenoEngine(self.rules, self.input_vars)
        self.coefficients = None
        self.training_metrics = None

    @property
    def params_per_rule(self):
        return len(self.input_vars) + 1

    @property
    def total_params(self):
        return len(self.rules) * self.params_per_rule

    def load_training_frame(self):
        """
        Girdileri normalize dosyadan, hedef eforu mumkunse outlier_removed dosyasindan okur.

        Normalizasyon adiminda Effort artik normalize edilmiyor. Yine de rapor metriklerinin
        orijinal efor olceginde kalmasi icin hedef sutunu oncelikle outlier_removed dosyasindan
        aliyoruz.
        """
        normalized_path = self.config["normalized_path"]
        original_path = self.config["original_path"]

        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Normalize veri dosyasi bulunamadi: {normalized_path}")

        df = pd.read_csv(normalized_path)
        missing_inputs = [var for var in self.input_vars if var not in df.columns]
        if missing_inputs:
            raise ValueError(f"{normalized_path} icinde eksik girdi sutunlari var: {missing_inputs}")

        if os.path.exists(original_path):
            original_df = pd.read_csv(original_path)
            if self.target_col in original_df.columns and len(original_df) == len(df):
                df[self.target_col] = original_df[self.target_col].values
            elif self.target_col not in df.columns:
                raise ValueError(f"Hedef sutun bulunamadi: {self.target_col}")
        elif self.target_col not in df.columns:
            raise FileNotFoundError(f"Hedef icin ne normalize ne de orijinal dosya kullanilabiliyor: {original_path}")

        return df[self.input_vars + [self.target_col]].copy()

    def fit(self, df):
        X = df[self.input_vars]
        y = df[self.target_col].to_numpy(dtype=float)
        phi = self.engine.design_matrix(X)

        if len(y) < self.total_params:
            warnings.warn(
                f"{self.dataset_name} icin {len(y)} egitim satiri ve {self.total_params} parametre var. "
                "Tam kural seviyeli Sugeno bu durumda asiri ogrenmeye yatkindir.",
                RuntimeWarning,
            )

        reg = self.regularization * np.eye(phi.shape[1], dtype=float)
        try:
            params = np.linalg.solve(phi.T @ phi + reg, phi.T @ y)
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(phi, y, rcond=None)[0]

        self.coefficients = params.reshape(len(self.rules), self.params_per_rule)
        self.engine.set_coefficients(self.coefficients)
        preds = self.predict(df)
        self.training_metrics = regression_metrics(y, preds)
        return self

    def predict(self, df):
        return self.engine.predict(df[self.input_vars])

    def evaluate(self, df):
        actual = df[self.target_col].to_numpy(dtype=float)
        predicted = self.predict(df)
        return regression_metrics(actual, predicted)

    def _equation_string(self, rule_index):
        coeffs = self.coefficients[rule_index]
        parts = [f"{coeffs[i]:.6f}*{var}" for i, var in enumerate(self.input_vars)]
        parts.append(f"{coeffs[-1]:.6f}")
        return f"f{rule_index + 1}(x) = " + " + ".join(parts)

    def equation_records(self):
        if self.coefficients is None:
            raise ValueError("Denklemleri kaydetmeden once model egitilmelidir.")

        records = []
        for idx, rule in enumerate(self.rules):
            coeffs = self.coefficients[idx]
            records.append({
                "rule_id": rule["rule_id"],
                "original_label": rule["original_label"],
                "sugeno_output": rule["sugeno_output"],
                "antecedent": rule["antecedent"],
                "converted_rule": rule["converted_rule"],
                "equation": self._equation_string(idx),
                "coefficients": {
                    **{var: float(coeffs[i]) for i, var in enumerate(self.input_vars)},
                    "bias": float(coeffs[-1]),
                },
            })
        return records

    def save_equations(self, output_dir="models/sugeno_equations"):
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{self.dataset_name}_{self.output_llm_name}_equations"
        json_path = os.path.join(output_dir, f"{base_name}.json")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        payload = {
            "dataset": self.dataset_name,
            "llm": self.output_llm_name,
            "rule_source_llm": self.rule_file_llm_name,
            "input_vars": self.input_vars,
            "number_of_rules": len(self.rules),
            "number_of_inputs": len(self.input_vars),
            "total_learned_parameters": self.total_params,
            "regularization": self.regularization,
            "rules": self.equation_records(),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"LLM: {self.output_llm_name}\n")
            f.write(f"Kural sayisi: {len(self.rules)}\n")
            f.write(f"Girdi sayisi: {len(self.input_vars)}\n")
            f.write(f"Toplam ogrenilen parametre: {self.total_params}\n\n")
            f.write("Not: Her kural kendi R_i_OUT cikti terimine ve kendi birinci derece denklemine sahiptir.\n")
            f.write("Albrecht veri setinde satir sayisi az oldugu icin 80 parametre asiri ogrenmeye yatkindir.\n")
            f.write("Desharnais daha fazla ornek icerdigi icin bu tasarim acisindan daha guvenlidir.\n\n")

            for record in payload["rules"]:
                f.write(f"{record['rule_id']} | Orijinal etiket: {record['original_label']} | {record['sugeno_output']}\n")
                f.write(f"{record['antecedent']}\n")
                f.write(f"{record['equation']}\n\n")

        return json_path, txt_path

    def save_predictions(self, df, split_name, output_dir="reports/predictions"):
        os.makedirs(output_dir, exist_ok=True)
        actual = df[self.target_col].to_numpy(dtype=float)
        predicted = self.predict(df)
        out = df[self.input_vars].copy()
        out["split"] = split_name
        out["actual"] = actual
        out["predicted"] = predicted
        out["residual"] = actual - predicted

        path = os.path.join(output_dir, f"{self.dataset_name}_{self.output_llm_name}_{split_name}_predictions.csv")
        out.to_csv(path, index=False)
        return path

    def save_rule_analysis(self, df, split_name="all", output_dir="reports/rule_analysis"):
        os.makedirs(output_dir, exist_ok=True)
        actual = df[self.target_col].to_numpy(dtype=float)
        predicted = self.predict(df)
        contributions = self.engine.rule_contributions(
            df[self.input_vars],
            y_true=actual,
            y_pred=predicted,
            split_name=split_name,
        )
        summary = self.engine.dominant_rule_summary(contributions)

        contrib_path = os.path.join(output_dir, f"{self.dataset_name}_{self.output_llm_name}_rule_contributions.csv")
        summary_path = os.path.join(output_dir, f"{self.dataset_name}_{self.output_llm_name}_dominant_rules.csv")
        contributions.to_csv(contrib_path, index=False)
        summary.to_csv(summary_path, index=False)
        return contrib_path, summary_path

    def save_plots(self, train_df, test_df, output_dir="reports/figures"):
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        test_actual = test_df[self.target_col].to_numpy(dtype=float)
        test_pred = self.predict(test_df)
        residuals = test_actual - test_pred

        plt.figure(figsize=(7, 6))
        plt.scatter(test_actual, test_pred, alpha=0.8)
        min_val = min(test_actual.min(), test_pred.min())
        max_val = max(test_actual.max(), test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel("Actual Effort")
        plt.ylabel("Predicted Effort")
        plt.title(f"Predicted vs Actual - {self.dataset_name} {self.output_llm_name}")
        pred_path = os.path.join(output_dir, f"{self.dataset_name}_{self.output_llm_name}_predicted_vs_actual.png")
        plt.savefig(pred_path, bbox_inches="tight")
        plt.close()
        paths["predicted_vs_actual"] = pred_path

        plt.figure(figsize=(7, 5))
        plt.scatter(test_pred, residuals, alpha=0.8)
        plt.axhline(0.0, color="r", linestyle="--")
        plt.xlabel("Predicted Effort")
        plt.ylabel("Residual")
        plt.title(f"Residuals - {self.dataset_name} {self.output_llm_name}")
        residual_path = os.path.join(output_dir, f"{self.dataset_name}_{self.output_llm_name}_residuals.png")
        plt.savefig(residual_path, bbox_inches="tight")
        plt.close()
        paths["residuals"] = residual_path

        contributions = self.engine.rule_contributions(train_df[self.input_vars])
        summary = self.engine.dominant_rule_summary(contributions).head(10)
        plt.figure(figsize=(9, 5))
        plt.bar(summary["rule_id"], summary["average_normalized_firing_strength"])
        plt.xlabel("Rule")
        plt.ylabel("Average Normalized Firing Strength")
        plt.title(f"Top Dominant Rules - {self.dataset_name} {self.output_llm_name}")
        dominance_path = os.path.join(output_dir, f"{self.dataset_name}_{self.output_llm_name}_rule_dominance.png")
        plt.savefig(dominance_path, bbox_inches="tight")
        plt.close()
        paths["rule_dominance"] = dominance_path

        paths["surface"] = self.save_surface_plot(output_dir)
        return paths

    def save_surface_plot(self, output_dir="reports/figures"):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        var1, var2 = self.input_vars[:2]
        grid = np.linspace(0, 1, 25)
        x_grid, y_grid = np.meshgrid(grid, grid)
        rows = []
        for x_val, y_val in zip(x_grid.ravel(), y_grid.ravel()):
            row = {var: 0.5 for var in self.input_vars}
            row[var1] = x_val
            row[var2] = y_val
            rows.append(row)

        surface_df = pd.DataFrame(rows)
        z_grid = self.predict(surface_df).reshape(x_grid.shape)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", alpha=0.9)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel("Effort")
        ax.set_title(f"Sugeno Surface - {self.dataset_name} {self.output_llm_name}")

        path = os.path.join(output_dir, f"{self.dataset_name}_{self.output_llm_name}_sugeno_surface.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path
