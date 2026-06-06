import numpy as np
import pandas as pd


def _trapezoid_membership(x, a, b, c, d):
    if x <= a:
        return 1.0 if a == b else 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c)
    return 1.0 if c == d and x == d else 0.0


def _triangle_membership(x, a, b, c):
    if x == b:
        return 1.0
    if a == b and x <= b:
        return 1.0
    if b == c and x >= b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0.0


def membership_degree(term, value, variable_name=None):
    """
    fuzzy_design.py icindeki normalize [0, 1] uyelik fonksiyonlariyla ayni sekli kullanir.
    Low: trapezoid, Medium: Gaussian, High: triangular.

    Uniform fuzzification uses fixed symmetric boundaries. It is simple and
    interpretable, but it can ignore the actual distribution of each feature.
    Quantile runners inject a different membership function without changing
    this default behavior.
    """
    x = float(np.clip(value, 0.0, 1.0))
    normalized_term = term.lower()

    if normalized_term == "low":
        return _trapezoid_membership(x, 0.0, 0.0, 0.15, 0.35)
    if normalized_term == "medium":
        sigma = 0.15
        center = 0.5
        return float(np.exp(-((x - center) ** 2) / (2 * sigma ** 2)))
    if normalized_term == "high":
        return _triangle_membership(x, 0.6, 1.0, 1.0)

    raise ValueError(f"Bilinmeyen uyelik terimi: {term}")


class ManualSugenoEngine:
    """
    Kural seviyeli birinci derece Sugeno motoru.

    Kural atesleme gucu antecedent kosullarinin carpimiyle hesaplanir.
    Son tahmin, normalize kural agirliklari ile kural denklemlerinin agirlikli toplamidir.
    """

    def __init__(self, rules, input_vars, coefficients=None, epsilon=1e-12, membership_function=None):
        self.rules = rules
        self.input_vars = input_vars
        self.epsilon = epsilon
        self.membership_function = membership_function or membership_degree
        self.coefficients = None
        if coefficients is not None:
            self.set_coefficients(coefficients)

    @property
    def params_per_rule(self):
        return len(self.input_vars) + 1

    @property
    def total_params(self):
        return len(self.rules) * self.params_per_rule

    def set_coefficients(self, coefficients):
        arr = np.asarray(coefficients, dtype=float)
        self.coefficients = arr.reshape(len(self.rules), self.params_per_rule)

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

        if total <= self.epsilon:
            # No-rule-fire durumunda sessiz sifir dondurmek yerine butun kurallara
            # esit agirlik veriyoruz; bu tahmini sayisal olarak tanimli tutar.
            normalized = np.ones(len(self.rules), dtype=float) / len(self.rules)
        else:
            normalized = strengths / total

        return strengths, normalized

    def design_matrix(self, X):
        rows = []
        for _, row in X.iterrows():
            _, normalized = self.normalized_strengths_for_row(row)
            values = [float(row[var]) for var in self.input_vars]

            design_row = []
            for weight in normalized:
                design_row.extend([weight * value for value in values])
                design_row.append(weight)
            rows.append(design_row)

        return np.asarray(rows, dtype=float)

    def rule_outputs_for_row(self, row):
        if self.coefficients is None:
            raise ValueError("Kural denklemleri henuz egitilmedi.")

        x = np.asarray([float(row[var]) for var in self.input_vars] + [1.0], dtype=float)
        return self.coefficients @ x

    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Kural denklemleri henuz egitilmedi.")

        phi = self.design_matrix(X)
        return phi @ self.coefficients.reshape(-1)

    def rule_contributions(self, X, y_true=None, y_pred=None, split_name=None):
        records = []
        y_true_values = None if y_true is None else np.asarray(y_true, dtype=float)
        y_pred_values = None if y_pred is None else np.asarray(y_pred, dtype=float)

        for sample_pos, (_, row) in enumerate(X.iterrows()):
            strengths, normalized = self.normalized_strengths_for_row(row)
            outputs = self.rule_outputs_for_row(row)
            contributions = normalized * outputs

            for rule_idx, rule in enumerate(self.rules):
                record = {
                    "sample_index": sample_pos,
                    "split": split_name,
                    "rule_id": rule["rule_id"],
                    "original_label": rule["original_label"],
                    "sugeno_output": rule["sugeno_output"],
                    "firing_strength": strengths[rule_idx],
                    "normalized_firing_strength": normalized[rule_idx],
                    "rule_output": outputs[rule_idx],
                    "rule_contribution": contributions[rule_idx],
                }

                for var in self.input_vars:
                    record[var] = row[var]

                if y_true_values is not None:
                    record["actual"] = y_true_values[sample_pos]
                if y_pred_values is not None:
                    record["predicted"] = y_pred_values[sample_pos]

                records.append(record)

        return pd.DataFrame(records)

    def dominant_rule_summary(self, contributions_df):
        summary = (
            contributions_df
            .groupby(["rule_id", "original_label", "sugeno_output"], as_index=False)
            .agg(
                average_firing_strength=("firing_strength", "mean"),
                average_normalized_firing_strength=("normalized_firing_strength", "mean"),
                average_rule_contribution=("rule_contribution", "mean"),
                average_abs_rule_contribution=("rule_contribution", lambda values: np.mean(np.abs(values))),
            )
            .sort_values("average_normalized_firing_strength", ascending=False)
        )
        return summary
