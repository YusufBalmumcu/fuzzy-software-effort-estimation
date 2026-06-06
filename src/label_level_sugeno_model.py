import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.full_sugeno_model import DATASET_CONFIG, LLM_NAME_ALIASES
from src.manual_sugeno_engine import membership_degree
from src.rule_converter import parse_rule
from src.rules import get_all_rules


OUTPUT_LABELS = ["Very_Low", "Low", "Medium", "High", "Very_High"]


class LabelLevelSugenoModel:
    """
    Sugeno V1 - etiket seviyeli birinci derece Sugeno modeli.

    V1 etiket seviyelidir: her cikti etiketi icin bir denklem ogrenir.
    5 etiket ve 3 girdi varsa toplam 5 * (3 + 1) = 20 parametre vardir.

    V2 tam kural seviyelidir: her kural icin bir denklem ogrenir.
    20 kural ve 3 girdi varsa 20 * (3 + 1) = 80 parametre vardir.

    Bu nedenle V1 kucuk veri setlerinde daha kararlidir; V2 daha esnektir ama
    asiri ogrenme riski daha yuksektir.
    """

    def __init__(
        self,
        dataset_name,
        llm_name="gemini",
        regularization=1e-2,
        membership_function=None,
        fuzzification_name="uniform",
        mf_type="mixed",
    ):
        self.dataset_name = dataset_name.lower()
        self.llm_name = llm_name.lower()
        self.rule_file_llm_name = LLM_NAME_ALIASES.get(self.llm_name, self.llm_name)
        self.output_llm_name = "gpt" if self.rule_file_llm_name == "chatgpt" else self.rule_file_llm_name
        self.regularization = regularization
        self.membership_function = membership_function or membership_degree
        self.fuzzification_name = fuzzification_name
        self.mf_type = mf_type

        if self.dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")

        self.config = DATASET_CONFIG[self.dataset_name]
        self.input_vars = self.config["input_vars"]
        self.target_col = self.config["target_col"]
        self.output_labels = OUTPUT_LABELS
        self.label_to_index = {label: idx for idx, label in enumerate(self.output_labels)}

        raw_rules = get_all_rules(self.dataset_name, self.rule_file_llm_name)
        if not raw_rules:
            raise ValueError(f"{self.dataset_name}/{self.rule_file_llm_name} icin kural bulunamadi.")

        self.rules = self._parse_rules(raw_rules)
        self.coefficients = None
        self.training_metrics = None

    @property
    def params_per_label(self):
        return len(self.input_vars) + 1

    @property
    def total_params(self):
        return len(self.output_labels) * self.params_per_label

    def _parse_rules(self, raw_rules):
        parsed_rules = []
        expected_vars = set(self.input_vars)

        for idx, rule_text in enumerate(raw_rules, start=1):
            parsed = parse_rule(rule_text)
            label = parsed["original_label"]
            if label not in self.label_to_index:
                raise ValueError(f"Beklenmeyen cikti etiketi: {label}")

            variables = {condition["variable"] for condition in parsed["conditions"]}
            if not variables.issubset(expected_vars):
                missing = sorted(variables - expected_vars)
                raise ValueError(f"R{idx} beklenmeyen degiskenler iceriyor: {missing}")

            parsed_rules.append({
                "rule_id": f"R{idx}",
                "original_rule": rule_text,
                "antecedent": parsed["antecedent"],
                "conditions": parsed["conditions"],
                "consequent_label": label,
            })

        return parsed_rules

    def load_training_frame(self):
        """
        Girdileri normalize dosyadan, hedef eforu mumkunse outlier_removed dosyasindan okur.

        Metrikler orijinal Effort olceginde hesaplanir. Bu, Linear Regression,
        Decision Tree, Sugeno V1 ve Sugeno V2 karsilastirmasini ayni olcege getirir.
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

    def firing_strengths_for_row(self, row):
        strengths = []
        for rule in self.rules:
            degree = 1.0
            for condition in rule["conditions"]:
                degree *= self.membership_function(
                    condition["term"],
                    row[condition["variable"]],
                    condition["variable"],
                )
            strengths.append(degree)
        return np.asarray(strengths, dtype=float)

    def normalized_strengths_for_row(self, row):
        strengths = self.firing_strengths_for_row(row)
        total = strengths.sum()
        if total <= 1e-12:
            # No-rule-fire durumunda sessiz sifir yerine esit kural agirligi kullanilir.
            normalized = np.ones(len(self.rules), dtype=float) / len(self.rules)
        else:
            normalized = strengths / total
        return strengths, normalized

    def design_matrix(self, X):
        rows = []
        for _, row in X.iterrows():
            _, normalized = self.normalized_strengths_for_row(row)
            x_aug = [float(row[var]) for var in self.input_vars] + [1.0]
            design_row = np.zeros(self.total_params, dtype=float)

            for rule_idx, rule in enumerate(self.rules):
                label_idx = self.label_to_index[rule["consequent_label"]]
                start = label_idx * self.params_per_label
                design_row[start:start + self.params_per_label] += normalized[rule_idx] * np.asarray(x_aug)

            rows.append(design_row)

        return np.asarray(rows, dtype=float)

    def fit(self, df):
        X = df[self.input_vars]
        y = df[self.target_col].to_numpy(dtype=float)
        phi = self.design_matrix(X)

        if len(y) < self.total_params:
            warnings.warn(
                f"{self.dataset_name} icin {len(y)} egitim satiri ve {self.total_params} parametre var. "
                "Etiket seviyeli Sugeno daha kararlidir ama yine de dikkatli yorumlanmalidir.",
                RuntimeWarning,
            )

        reg = self.regularization * np.eye(phi.shape[1], dtype=float)
        try:
            params = np.linalg.solve(phi.T @ phi + reg, phi.T @ y)
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(phi, y, rcond=None)[0]

        self.coefficients = params.reshape(len(self.output_labels), self.params_per_label)
        preds = self.predict(df)
        self.training_metrics = regression_metrics(y, preds)
        return self

    def label_output_for_row(self, label, row):
        if self.coefficients is None:
            raise ValueError("Etiket denklemleri henuz egitilmedi.")

        label_idx = self.label_to_index[label]
        x_aug = np.asarray([float(row[var]) for var in self.input_vars] + [1.0], dtype=float)
        return float(self.coefficients[label_idx] @ x_aug)

    def predict(self, df):
        if self.coefficients is None:
            raise ValueError("Etiket denklemleri henuz egitilmedi.")

        phi = self.design_matrix(df[self.input_vars])
        return phi @ self.coefficients.reshape(-1)

    def evaluate(self, df):
        actual = df[self.target_col].to_numpy(dtype=float)
        predicted = self.predict(df)
        return regression_metrics(actual, predicted)

    def _equation_string(self, label):
        coeffs = self.coefficients[self.label_to_index[label]]
        parts = [f"{coeffs[i]:.6f}*{var}" for i, var in enumerate(self.input_vars)]
        parts.append(f"{coeffs[-1]:.6f}")
        return f"{label}(x) = " + " + ".join(parts)

    def equation_records(self):
        if self.coefficients is None:
            raise ValueError("Denklemleri kaydetmeden once model egitilmelidir.")

        records = {}
        for label in self.output_labels:
            coeffs = self.coefficients[self.label_to_index[label]]
            records[label] = {
                "equation": self._equation_string(label),
                "coefficients": {
                    **{var: float(coeffs[i]) for i, var in enumerate(self.input_vars)},
                    "bias": float(coeffs[-1]),
                },
            }
        return records

    def save_equations(self, output_dir="models/sugeno_label_equations"):
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{self.dataset_name}_{self.output_llm_name}_label_equations"
        json_path = os.path.join(output_dir, f"{base_name}.json")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        payload = {
            "dataset": self.dataset_name,
            "llm": self.output_llm_name,
            "rule_source_llm": self.rule_file_llm_name,
            "model_type": "label_level_first_order_sugeno",
            "input_vars": self.input_vars,
            "rules": len(self.rules),
            "output_label_count": len(self.output_labels),
            "parameter_count": self.total_params,
            "regularization": self.regularization,
            "fuzzification": self.fuzzification_name,
            "membership_function_type": self.mf_type,
            "output_labels": self.equation_records(),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"LLM: {self.output_llm_name}\n")
            f.write("Model: Sugeno V1 Label-Level First-Order\n")
            f.write(f"Kural sayisi: {len(self.rules)}\n")
            f.write(f"Cikti etiketi sayisi: {len(self.output_labels)}\n")
            f.write(f"Girdi sayisi: {len(self.input_vars)}\n")
            f.write(f"Toplam ogrenilen parametre: {self.total_params}\n\n")
            f.write(f"Fuzzification: {self.fuzzification_name} ({self.mf_type})\n\n")
            f.write("V1: Her cikti etiketi icin bir denklem ogrenir; daha az parametre ve daha kararlı davranis.\n")
            f.write("V2: Her kural icin bir denklem ogrenir; daha esnek fakat asiri ogrenme riski daha yuksek.\n\n")
            for label, record in payload["output_labels"].items():
                f.write(f"{label}\n")
                f.write(f"{record['equation']}\n\n")

        return json_path, txt_path

    def save_predictions(self, df, split_name, output_dir="reports/predictions"):
        os.makedirs(output_dir, exist_ok=True)
        actual = df[self.target_col].to_numpy(dtype=float)
        predicted = self.predict(df)
        residual = actual - predicted

        out = df[self.input_vars].copy()
        out["split"] = split_name
        out["actual"] = actual
        out["predicted"] = predicted
        out["residual"] = residual
        out["absolute_error"] = np.abs(residual)
        out["percentage_error"] = np.where(actual != 0, np.abs(residual / actual) * 100, np.nan)

        path = os.path.join(
            output_dir,
            f"{self.dataset_name}_{self.output_llm_name}_label_sugeno_{split_name}_predictions.csv",
        )
        out.to_csv(path, index=False)
        return path

    def rule_contributions(self, df, split_name="all"):
        records = []
        actual = df[self.target_col].to_numpy(dtype=float) if self.target_col in df.columns else None
        predicted = self.predict(df)

        for sample_pos, (_, row) in enumerate(df[self.input_vars].iterrows()):
            strengths, normalized = self.normalized_strengths_for_row(row)
            for rule_idx, rule in enumerate(self.rules):
                label_output = self.label_output_for_row(rule["consequent_label"], row)
                contribution = normalized[rule_idx] * label_output
                record = {
                    "sample_index": sample_pos,
                    "split": split_name,
                    "rule_id": rule["rule_id"],
                    "antecedent": rule["antecedent"],
                    "consequent_label": rule["consequent_label"],
                    "firing_strength": strengths[rule_idx],
                    "normalized_firing_strength": normalized[rule_idx],
                    "label_equation_output": label_output,
                    "rule_contribution": contribution,
                    "predicted": predicted[sample_pos],
                }
                for var in self.input_vars:
                    record[var] = row[var]
                if actual is not None:
                    record["actual"] = actual[sample_pos]
                records.append(record)

        return pd.DataFrame(records)

    def dominant_rule_summary(self, contributions_df):
        return (
            contributions_df
            .groupby(["rule_id", "consequent_label"], as_index=False)
            .agg(
                average_firing_strength=("firing_strength", "mean"),
                average_normalized_firing_strength=("normalized_firing_strength", "mean"),
                average_rule_contribution=("rule_contribution", "mean"),
                average_abs_rule_contribution=("rule_contribution", lambda values: np.mean(np.abs(values))),
            )
            .sort_values("average_normalized_firing_strength", ascending=False)
        )

    def save_rule_analysis(self, df, split_name="all", output_dir="reports/rule_analysis"):
        os.makedirs(output_dir, exist_ok=True)
        contributions = self.rule_contributions(df, split_name=split_name)
        summary = self.dominant_rule_summary(contributions)

        contrib_path = os.path.join(
            output_dir,
            f"{self.dataset_name}_{self.output_llm_name}_label_rule_contributions.csv",
        )
        summary_path = os.path.join(
            output_dir,
            f"{self.dataset_name}_{self.output_llm_name}_label_dominant_rules.csv",
        )
        contributions.to_csv(contrib_path, index=False)
        summary.to_csv(summary_path, index=False)
        return contrib_path, summary_path

    def save_plots(self, train_df, test_df, output_dir="reports/figures"):
        os.makedirs(output_dir, exist_ok=True)
        actual = test_df[self.target_col].to_numpy(dtype=float)
        predicted = self.predict(test_df)
        residuals = actual - predicted

        plt.figure(figsize=(7, 6))
        plt.scatter(actual, predicted, alpha=0.8)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel("Actual Effort")
        plt.ylabel("Predicted Effort")
        plt.title(f"Sugeno V1 Label-Level - {self.dataset_name} {self.output_llm_name}")
        predicted_path = os.path.join(
            output_dir,
            f"{self.dataset_name}_{self.output_llm_name}_label_sugeno_predicted_vs_actual.png",
        )
        plt.savefig(predicted_path, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.scatter(predicted, residuals, alpha=0.8)
        plt.axhline(0.0, color="r", linestyle="--")
        plt.xlabel("Predicted Effort")
        plt.ylabel("Residual")
        plt.title(f"Sugeno V1 Label-Level Residuals - {self.dataset_name} {self.output_llm_name}")
        residual_path = os.path.join(
            output_dir,
            f"{self.dataset_name}_{self.output_llm_name}_label_sugeno_residuals.png",
        )
        plt.savefig(residual_path, bbox_inches="tight")
        plt.close()

        return {
            "predicted_vs_actual": predicted_path,
            "residuals": residual_path,
        }
