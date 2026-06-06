import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.full_sugeno_model import DATASET_CONFIG


MF_TYPES = ("triangular", "trapezoidal", "gaussian")
TERMS = ("Low", "Medium", "High")
EPSILON = 1e-6
QUANTILE_SEPARATION = 1e-4


def _as_float(value):
    return float(np.asarray(value, dtype=float))


def triangular_membership_values(x, a, b, c):
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr, dtype=float)

    if a == b:
        y = np.where(x_arr <= b, 1.0, y)
    else:
        rising = (a < x_arr) & (x_arr < b)
        y = np.where(rising, (x_arr - a) / (b - a), y)

    y = np.where(x_arr == b, 1.0, y)

    if b == c:
        y = np.where(x_arr >= b, 1.0, y)
    else:
        falling = (b < x_arr) & (x_arr < c)
        y = np.where(falling, (c - x_arr) / (c - b), y)

    y = np.clip(y, 0.0, 1.0)
    return float(y) if np.isscalar(x) else y


def trapezoidal_membership_values(x, a, b, c, d):
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr, dtype=float)

    if a == b:
        y = np.where(x_arr <= b, 1.0, y)
    else:
        rising = (a < x_arr) & (x_arr < b)
        y = np.where(rising, (x_arr - a) / (b - a), y)

    plateau = (b <= x_arr) & (x_arr <= c)
    y = np.where(plateau, 1.0, y)

    if c == d:
        y = np.where(x_arr >= c, 1.0, y)
    else:
        falling = (c < x_arr) & (x_arr < d)
        y = np.where(falling, (d - x_arr) / (d - c), y)

    y = np.clip(y, 0.0, 1.0)
    return float(y) if np.isscalar(x) else y


def gaussian_membership_values(x, center, sigma):
    sigma = max(float(sigma), EPSILON)
    x_arr = np.asarray(x, dtype=float)
    y = np.exp(-((x_arr - float(center)) ** 2) / (2 * sigma ** 2))
    y = np.clip(y, 0.0, 1.0)
    return float(y) if np.isscalar(x) else y


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _safe_boundary_quantiles(stats, feature_name):
    q1 = _clip01(stats["q1"])
    q2 = _clip01(stats["q2"])
    q3 = _clip01(stats["q3"])
    warnings_for_feature = []

    if stats["max"] - stats["min"] <= EPSILON:
        warnings_for_feature.append(
            f"{feature_name} has constant or near-constant values; uniform fallback boundaries were used."
        )
        return 0.25, 0.50, 0.75, warnings_for_feature, "uniform_fallback_constant_feature"

    if q2 - q1 < QUANTILE_SEPARATION:
        adjusted = max(0.0, q2 - QUANTILE_SEPARATION)
        warnings_for_feature.append(
            f"{feature_name}: Q1 and Q2 are too close; Q1 boundary adjusted from {q1:.6f} to {adjusted:.6f}."
        )
        q1 = adjusted

    if q3 - q2 < QUANTILE_SEPARATION:
        adjusted = min(1.0, q2 + QUANTILE_SEPARATION)
        warnings_for_feature.append(
            f"{feature_name}: Q2 and Q3 are too close; Q3 boundary adjusted from {q3:.6f} to {adjusted:.6f}."
        )
        q3 = adjusted

    if not (0.0 <= q1 < q2 < q3 <= 1.0):
        warnings_for_feature.append(
            f"{feature_name}: adjusted quartiles are not strictly ordered; uniform fallback boundaries were used."
        )
        return 0.25, 0.50, 0.75, warnings_for_feature, "uniform_fallback_invalid_quantiles"

    return q1, q2, q3, warnings_for_feature, "quantile"


def _feature_stats(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("Feature has no numeric values.")

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "q1": float(values.quantile(0.25)),
        "q2": float(values.quantile(0.50)),
        "q3": float(values.quantile(0.75)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def _params_from_quantiles(q1, q2, q3):
    left_spread = max(q2 - q1, QUANTILE_SEPARATION)
    right_spread = max(q3 - q2, QUANTILE_SEPARATION)
    margin = max(min(left_spread, right_spread) * 0.25, QUANTILE_SEPARATION)
    mid_left = max(q1, q2 - margin)
    mid_right = min(q3, q2 + margin)

    if mid_right < mid_left:
        mid_left, mid_right = q2, q2

    return {
        "triangular": {
            "Low": [0.0, 0.0, q2],
            "Medium": [q1, q2, q3],
            "High": [q2, 1.0, 1.0],
        },
        "trapezoidal": {
            "Low": [0.0, 0.0, q1, q2],
            "Medium": [q1, mid_left, mid_right, q3],
            "High": [q2, q3, 1.0, 1.0],
        },
        "gaussian": {
            "Low": {"center": q1, "sigma": max((q2 - 0.0) / 2.0, EPSILON)},
            "Medium": {"center": q2, "sigma": max((q3 - q1) / 2.0, EPSILON)},
            "High": {"center": q3, "sigma": max((1.0 - q2) / 2.0, EPSILON)},
        },
    }


def load_quantile_source_frame(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASET_CONFIG[dataset_name]
    normalized_path = config["normalized_path"]
    original_path = config["original_path"]
    target_col = config["target_col"]

    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Normalized data file not found: {normalized_path}")

    df = pd.read_csv(normalized_path)
    if os.path.exists(original_path):
        original_df = pd.read_csv(original_path)
        if target_col in original_df.columns and len(original_df) == len(df):
            df[target_col] = original_df[target_col].values

    missing = [col for col in [*config["input_vars"], target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"{normalized_path} is missing required columns: {missing}")

    return df[config["input_vars"] + [target_col]].copy()


def build_quantile_payload(dataset_name, df=None):
    dataset_name = dataset_name.lower()
    config = DATASET_CONFIG[dataset_name]
    df = load_quantile_source_frame(dataset_name) if df is None else df

    stats_rows = []
    payload = {
        "dataset": dataset_name,
        "method": "quantile_based_fuzzification",
        "description": (
            "Quantile fuzzification uses Q1, Q2, and Q3 from the actual normalized "
            "feature distribution. This can balance rule activation better than "
            "fixed 0/0.25/0.5/0.75/1 boundaries, but small datasets or tied values "
            "can make quartiles unstable."
        ),
        "features": {},
    }
    all_warnings = []

    for feature in config["input_vars"]:
        stats = _feature_stats(df[feature])
        stats_rows.append({"feature": feature, **stats})
        q1, q2, q3, feature_warnings, boundary_method = _safe_boundary_quantiles(stats, feature)
        params = _params_from_quantiles(q1, q2, q3)
        all_warnings.extend(feature_warnings)

        payload["features"][feature] = {
            **stats,
            "boundary_q1": q1,
            "boundary_q2": q2,
            "boundary_q3": q3,
            "boundary_method": boundary_method,
            "warnings": feature_warnings,
            **params,
        }

    payload["warnings"] = all_warnings
    return payload, pd.DataFrame(stats_rows)


def save_quantile_statistics(dataset_name, stats_df, output_dir="reports/quantile_fuzzification/statistics"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset_name}_quantile_stats.csv")
    stats_df.to_csv(path, index=False)
    return path


def save_quantile_params(dataset_name, payload, output_dir="models/quantile_fuzzy_params"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset_name}_quantile_mf_params.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


class QuantileMembershipProvider:
    def __init__(self, payload, mf_type="triangular"):
        if mf_type not in MF_TYPES:
            raise ValueError(f"Unsupported quantile MF type: {mf_type}. Choose one of {MF_TYPES}.")
        self.payload = payload
        self.mf_type = mf_type
        self.features = payload["features"]

    @classmethod
    def from_file(cls, path, mf_type="triangular"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload, mf_type=mf_type)

    def __call__(self, term, value, variable_name=None):
        if variable_name is None:
            raise ValueError("Quantile membership requires the feature name.")
        if variable_name not in self.features:
            raise ValueError(f"No quantile parameters found for feature: {variable_name}")

        normalized_term = str(term).strip().lower()
        term_name = {"low": "Low", "medium": "Medium", "high": "High"}.get(normalized_term)
        if term_name is None:
            raise ValueError(f"Unknown fuzzy term: {term}")

        x = _clip01(value)
        params = self.features[variable_name][self.mf_type][term_name]
        if self.mf_type == "triangular":
            return triangular_membership_values(x, *params)
        if self.mf_type == "trapezoidal":
            return trapezoidal_membership_values(x, *params)
        return gaussian_membership_values(x, params["center"], params["sigma"])


def load_quantile_membership_function(dataset_name, mf_type="triangular", params_dir="models/quantile_fuzzy_params"):
    dataset_name = dataset_name.lower()
    path = os.path.join(params_dir, f"{dataset_name}_quantile_mf_params.json")
    if not os.path.exists(path):
        payload, stats_df = build_quantile_payload(dataset_name)
        save_quantile_statistics(dataset_name, stats_df)
        save_quantile_params(dataset_name, payload, params_dir)
    return QuantileMembershipProvider.from_file(path, mf_type=mf_type)


def calculate_membership_value(mf_type, term, value, feature_params):
    term_params = feature_params[mf_type][term]
    x = _clip01(value)
    if mf_type == "triangular":
        return triangular_membership_values(x, *term_params)
    if mf_type == "trapezoidal":
        return trapezoidal_membership_values(x, *term_params)
    return gaussian_membership_values(x, term_params["center"], term_params["sigma"])


def save_membership_table(dataset_name, df, payload, mf_type, output_dir="data/processed_data/quantile_memberships"):
    os.makedirs(output_dir, exist_ok=True)
    config = DATASET_CONFIG[dataset_name]
    out = df[config["input_vars"] + [config["target_col"]]].copy()

    for feature in config["input_vars"]:
        feature_params = payload["features"][feature]
        for term in TERMS:
            out[f"{feature}_{term}"] = out[feature].apply(
                lambda value, feature_params=feature_params, term=term: calculate_membership_value(
                    mf_type,
                    term,
                    value,
                    feature_params,
                )
            )

    path = os.path.join(output_dir, f"{dataset_name}_{mf_type}_quantile_memberships.csv")
    out.to_csv(path, index=False)
    return path


def save_feature_plot(dataset_name, feature, feature_params, mf_type, output_dir="reports/figures/quantile_fuzzification"):
    os.makedirs(output_dir, exist_ok=True)
    x_values = np.linspace(0.0, 1.0, 300)

    plt.figure(figsize=(8, 5))
    for term in TERMS:
        if mf_type == "triangular":
            y_values = triangular_membership_values(x_values, *feature_params[mf_type][term])
        elif mf_type == "trapezoidal":
            y_values = trapezoidal_membership_values(x_values, *feature_params[mf_type][term])
        else:
            params = feature_params[mf_type][term]
            y_values = gaussian_membership_values(x_values, params["center"], params["sigma"])
        plt.plot(x_values, y_values, label=term)

    for label, key, color in [
        ("Q1", "q1", "#666666"),
        ("Q2", "q2", "#222222"),
        ("Q3", "q3", "#666666"),
    ]:
        plt.axvline(feature_params[key], color=color, linestyle="--", linewidth=1.0, label=label)

    plt.title(f"{dataset_name.capitalize()} - {feature} - {mf_type.capitalize()} Quantile MFs")
    plt.xlabel("Normalized Feature Value")
    plt.ylabel("Membership Degree")
    plt.ylim(-0.02, 1.05)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, f"{dataset_name}_{feature}_{mf_type}_quantile.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def save_summary_text(dataset_name, payload, stats_path, params_path, membership_paths, plot_paths):
    output_dir = "reports/quantile_fuzzification"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset_name}_quantile_fuzzification_summary.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write("Fuzzification: Quantile-based / quartile-based\n\n")
        f.write("Why quantile-based fuzzification was used\n")
        f.write(
            "Uniform fuzzification uses fixed symmetric boundaries. It is simple and interpretable, "
            "but it may ignore the actual distribution of each input feature. Quantile fuzzification "
            "uses Q1, Q2, and Q3 from the normalized feature values, so Low, Medium, and High adapt "
            "to each dataset and can improve rule activation balance.\n\n"
        )
        f.write("How Q1, Q2, Q3 were calculated\n")
        f.write(
            "For each selected normalized input feature, Q1 is the 25th percentile, Q2 is the median, "
            "and Q3 is the 75th percentile, computed with pandas quantile on the processed dataset.\n\n"
        )
        f.write("Membership definitions\n")
        f.write("Triangular: Low=[0,0,Q2], Medium=[Q1,Q2,Q3], High=[Q2,1,1]\n")
        f.write("Trapezoidal: Low=[0,0,Q1,Q2], Medium around Q2, High=[Q2,Q3,1,1]\n")
        f.write("Gaussian: centers use Q1/Q2/Q3 and sigma is derived from neighboring spread.\n\n")

        if payload["warnings"]:
            f.write("Warnings\n")
            for item in payload["warnings"]:
                f.write(f"- {item}\n")
            f.write("\n")

        f.write(f"Statistics CSV: {stats_path}\n")
        f.write(f"MF parameters JSON: {params_path}\n")
        f.write("Membership tables:\n")
        for mf_type, table_path in membership_paths.items():
            f.write(f"- {mf_type}: {table_path}\n")
        f.write("Plots:\n")
        for plot_path in plot_paths:
            f.write(f"- {plot_path}\n")

    return path


def generate_quantile_fuzzification_for_dataset(dataset_name, mf_types=MF_TYPES, verbose=True):
    dataset_name = dataset_name.lower()
    df = load_quantile_source_frame(dataset_name)
    payload, stats_df = build_quantile_payload(dataset_name, df)

    stats_path = save_quantile_statistics(dataset_name, stats_df)
    params_path = save_quantile_params(dataset_name, payload)

    membership_paths = {}
    plot_paths = []
    for mf_type in mf_types:
        membership_paths[mf_type] = save_membership_table(dataset_name, df, payload, mf_type)
        for feature in DATASET_CONFIG[dataset_name]["input_vars"]:
            plot_paths.append(save_feature_plot(dataset_name, feature, payload["features"][feature], mf_type))

    summary_path = save_summary_text(dataset_name, payload, stats_path, params_path, membership_paths, plot_paths)

    if verbose:
        print(f"\nDataset: {dataset_name}")
        print("Fuzzification: Quantile-based")
        for feature, params in payload["features"].items():
            print(f"Feature: {feature}")
            print(f"Q1: {params['q1']:.6f}")
            print(f"Q2: {params['q2']:.6f}")
            print(f"Q3: {params['q3']:.6f}")
            print(f"Boundary method: {params['boundary_method']}")
            print(f"MF parameters: {json.dumps({mf: params[mf] for mf in mf_types}, ensure_ascii=False)}")
        for warning_text in payload["warnings"]:
            warnings.warn(warning_text, RuntimeWarning)
        print(f"Saved statistics: {stats_path}")
        print(f"Saved MF parameters: {params_path}")
        for table_path in membership_paths.values():
            print(f"Saved membership table: {table_path}")
        print(f"Saved plots: {', '.join(plot_paths)}")
        print(f"Saved summary: {summary_path}")

    return {
        "dataset": dataset_name,
        "statistics": stats_path,
        "params": params_path,
        "membership_tables": membership_paths,
        "plots": plot_paths,
        "summary": summary_path,
    }


def generate_quantile_fuzzification_outputs(datasets=None, mf_types=MF_TYPES, verbose=True):
    datasets = datasets or list(DATASET_CONFIG.keys())
    return [
        generate_quantile_fuzzification_for_dataset(dataset_name, mf_types=mf_types, verbose=verbose)
        for dataset_name in datasets
    ]
