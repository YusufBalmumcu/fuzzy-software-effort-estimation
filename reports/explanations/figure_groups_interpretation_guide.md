# Figure Groups Summary

| Figure Group                                 | Meaning                                                          |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `outlier_*.png`                              | Shows extreme effort values before modeling.                     |
| `*_quantile.png`                             | Shows Low / Medium / High membership functions using Q1, Q2, Q3. |
| `*_predicted_vs_actual.png`                  | Compares real Effort with predicted Effort.                      |
| `*_residuals.png`                            | Shows prediction error: actual minus predicted.                  |
| `*_label_sugeno_predicted_vs_actual.png`     | Shows prediction accuracy for Sugeno V1 Label-Level.             |
| `*_label_sugeno_residuals.png`               | Shows error behavior for Sugeno V1 Label-Level.                  |
| `*_model_comparison_metrics.png`             | Compares models using RMSE, MAE, MAPE, and R2.                   |
| `*_model_comparison_predicted_vs_actual.png` | Shows predicted vs actual plots side by side for models.         |
| `*_model_comparison_residuals.png`           | Compares residual errors across models.                          |
| `*_final_model_comparison_metrics.png`       | Final metric comparison of baselines, V1, and V2.                |
| `*_rule_dominance.png`                       | Shows which fuzzy rules are most active/influential.             |
| `*_sugeno_surface.png`                       | Shows how predicted Effort changes with two inputs.              |
| `all_models_*_comparison.png`                | Overall comparison across all models and datasets.               |
| `old_vs_full_sugeno_*.png`                   | Compares old Sugeno with full rule-level Sugeno.                 |
| `*_uniform_vs_quantile_rmse.png`             | Compares uniform vs quantile fuzzification using RMSE.           |
| `*_uniform_vs_quantile_mape.png`             | Compares uniform vs quantile fuzzification using MAPE.           |
| `quantile/triangular/`                       | Results using triangular quantile membership functions.          |
| `quantile/trapezoidal/`                      | Results using trapezoidal quantile membership functions.         |
| `quantile/gaussian/`                         | Results using gaussian quantile membership functions.            |

# Figure Groups Interpretation Guide

This file explains the figure groups generated in the project and what each group tells us. It is written for report usage: each group has **purpose**, **how to read**, **keywords**, and **what it tells us**.

Project figures are mainly stored under:

```text
reports/figures/
reports/figures/quantile/
reports/figures/quantile/<mf_type>/
reports/figures/quantile_fuzzification/
```

The project uses these main model families:

- `Linear Regression`
- `Decision Tree`
- `Sugeno V1 Label-Level`
- `Sugeno V2 Full Rule-Level`
- Uniform fuzzification Sugeno
- Quantile fuzzification Sugeno with `triangular`, `trapezoidal`, and `gaussian` membership functions

---

## 1. Outlier Analysis Figures

Pattern:

```text
reports/figures/outlier_<dataset>.png
```

Examples:

```text
outlier_albrecht.png
outlier_desharnais.png
outlier_china.png
outlier_kemerer.png
```

### Purpose

These figures show the outlier analysis stage before normalization and model training.

They are used to understand whether the target effort values contain extreme observations that may distort model learning.

### What It Tells Us

The outlier figures tell us whether the dataset contains unusually high or low effort values. If extreme points exist, they can strongly affect regression and Sugeno coefficient learning.

### How To Read

Look for values that are far away from the main data distribution. These points are potential outliers.

If many values are clustered but one or two points are very far away, the dataset has high outlier risk.

### Keywords

```text
outlier detection
extreme values
data cleaning
effort distribution
preprocessing
model stability
noise reduction
```

### Report Sentence

The outlier analysis figures show whether each dataset contains extreme effort values. Removing or controlling these values helps prevent the models from learning distorted effort patterns.

---

## 2. Quantile Fuzzification Membership Function Figures

Pattern:

```text
reports/figures/quantile_fuzzification/<dataset>_<feature>_<mf_type>_quantile.png
```

Examples:

```text
albrecht_RawFPcounts_triangular_quantile.png
albrecht_Input_trapezoidal_quantile.png
albrecht_File_gaussian_quantile.png
desharnais_PointsAjust_triangular_quantile.png
desharnais_TeamExp_gaussian_quantile.png
desharnais_Length_trapezoidal_quantile.png
```

### Purpose

These figures show how `Low`, `Medium`, and `High` fuzzy sets are defined using the actual data distribution.

Unlike uniform fuzzification, quantile fuzzification uses:

```text
Q1 = 25th percentile
Q2 = median
Q3 = 75th percentile
```

### Axes

| Axis                  | Meaning                                  |
| --------------------- | ---------------------------------------- |
| x-axis                | Normalized feature value between 0 and 1 |
| y-axis                | Membership degree between 0 and 1        |
| vertical dashed lines | Q1, Q2, Q3                               |

### What It Tells Us

These figures tell us how each feature is linguistically divided into `Low`, `Medium`, and `High`.

They also show whether the membership functions are balanced around the actual data distribution.

For example:

- If Q1, Q2, and Q3 are close together, the feature values are concentrated in a narrow range.
- If Q1 and Q3 are far apart, the feature has wider spread.
- Gaussian MFs give smoother transitions.
- Triangular MFs give sharper transitions.
- Trapezoidal MFs create plateau regions where membership is fully active.

### Keywords

```text
quantile fuzzification
Q1 Q2 Q3
membership degree
Low Medium High
data-driven boundaries
triangular MF
trapezoidal MF
gaussian MF
rule activation balance
feature distribution
```

### Report Sentence

The quantile fuzzification figures show how fuzzy sets are adapted to each feature's real distribution. This makes the Low, Medium, and High boundaries data-driven instead of fixed.

---

## 3. Predicted vs Actual Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_predicted_vs_actual.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_predicted_vs_actual.png
```

Examples:

```text
albrecht_gpt_predicted_vs_actual.png
desharnais_gemini_predicted_vs_actual.png
reports/figures/quantile/gaussian/albrecht_gpt_predicted_vs_actual.png
```

### Purpose

These figures compare the actual effort values with the effort values predicted by a Sugeno model.

### Axes

| Axis            | Meaning                                             |
| --------------- | --------------------------------------------------- |
| x-axis          | Actual Effort                                       |
| y-axis          | Predicted Effort                                    |
| red dashed line | Ideal prediction line where predicted equals actual |

### What It Tells Us

If points are close to the red diagonal line, the model predicts well.

If points are far above the line, the model overestimates effort.

If points are far below the line, the model underestimates effort.

A wide scatter around the diagonal indicates weak prediction accuracy.

### Keywords

```text
prediction accuracy
actual vs predicted
diagonal line
overestimation
underestimation
generalization
model fit
prediction error
```

### Report Sentence

The predicted vs actual figures show how close the model predictions are to real effort values. Points closer to the diagonal line indicate better prediction performance.

---

## 4. Label-Level Sugeno Predicted vs Actual Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_label_sugeno_predicted_vs_actual.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_label_sugeno_predicted_vs_actual.png
```

### Purpose

These figures specifically represent `Sugeno V1 Label-Level` prediction quality.

Sugeno V1 learns one equation per output label:

```text
Very_Low
Low
Medium
High
Very_High
```

### What It Tells Us

These plots tell us whether the lower-complexity V1 model can generalize better than the more flexible V2 model.

Since V1 has fewer parameters, its predicted vs actual plots are useful for checking stability.

### Keywords

```text
Sugeno V1
label-level model
shared output equations
stable prediction
lower complexity
generalization
actual vs predicted
```

### Report Sentence

The label-level predicted vs actual figures evaluate the Sugeno V1 model. They show whether shared label-level equations produce stable predictions on test data.

---

## 5. Residual Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_residuals.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_residuals.png
```

Examples:

```text
albrecht_claude_residuals.png
desharnais_gpt_residuals.png
reports/figures/quantile/trapezoidal/desharnais_gpt_residuals.png
```

### Purpose

Residual plots show prediction error directly.

Residual formula:

```text
residual = actual effort - predicted effort
```

### Axes

| Axis                | Meaning          |
| ------------------- | ---------------- |
| x-axis              | Predicted Effort |
| y-axis              | Residual         |
| red horizontal line | Zero error line  |

### What It Tells Us

If residuals are randomly scattered around zero, the model has no obvious systematic bias.

If residuals are mostly above zero, the model tends to underestimate effort.

If residuals are mostly below zero, the model tends to overestimate effort.

If residuals grow as prediction increases, the model has heteroscedasticity or scale-dependent error.

### Keywords

```text
residual
prediction error
zero error line
bias
underestimation
overestimation
heteroscedasticity
error pattern
model diagnostics
```

### Report Sentence

Residual figures show whether prediction errors are balanced around zero. A good model should have residuals distributed randomly without a clear pattern.

---

## 6. Label-Level Sugeno Residual Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_label_sugeno_residuals.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_label_sugeno_residuals.png
```

### Purpose

These figures show residuals specifically for the `Sugeno V1 Label-Level` model.

### What It Tells Us

They help diagnose whether V1 makes systematic errors despite having fewer parameters.

If V1 residuals are more balanced than V2 residuals, V1 may be more stable.

### Keywords

```text
Sugeno V1 residuals
label-level error
model stability
bias detection
generalization error
```

### Report Sentence

The label-level residual plots show the error behavior of Sugeno V1. Balanced residuals indicate that the label-level model is not strongly biased.

---

## 7. Model Comparison Metrics Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_model_comparison_metrics.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_<mf_type>_v2_model_comparison_metrics.png
```

Examples:

```text
albrecht_gpt_model_comparison_metrics.png
desharnais_claude_model_comparison_metrics.png
reports/figures/quantile/gaussian/desharnais_gpt_gaussian_v2_model_comparison_metrics.png
```

### Purpose

These bar charts compare multiple models using performance metrics.

Typical metrics:

| Metric   | Meaning                  | Better Direction |
| -------- | ------------------------ | ---------------- |
| RMSE     | Root Mean Squared Error  | Lower is better  |
| MAE      | Mean Absolute Error      | Lower is better  |
| MAPE (%) | Percentage error         | Lower is better  |
| R2       | Explained variance score | Higher is better |

### What It Tells Us

These figures tell us which model performs better on the test split.

They usually compare:

- Sugeno model
- Linear Regression
- Decision Tree

If a model has very low train error but high test error, it may be overfitting.

### Keywords

```text
model comparison
RMSE
MAE
MAPE
R2
test performance
baseline comparison
overfitting
generalization
```

### Report Sentence

The model comparison metrics figures compare Sugeno models with machine learning baselines. Lower RMSE, MAE, and MAPE indicate better accuracy, while higher R2 indicates better explanatory power.

---

## 8. Final Model Comparison Metrics Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_final_model_comparison_metrics.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_<mf_type>_final_quantile_model_comparison_metrics.png
```

### Purpose

These figures are broader final comparison plots. They usually include:

- Linear Regression
- Decision Tree
- Sugeno V1 Label-Level
- Sugeno V2 Full Rule-Level, when available

### What It Tells Us

They tell us the final relative performance of all selected models for a dataset and LLM rule source.

This group is useful for answering:

```text
Which model is best overall?
Does Sugeno V1 beat the baselines?
Does Sugeno V2 overfit?
Does quantile fuzzification improve test metrics?
```

### Keywords

```text
final comparison
best model
baseline vs Sugeno
V1 vs V2
model ranking
test RMSE
test MAPE
test R2
```

### Report Sentence

The final model comparison figures summarize the performance of all major models and help identify which approach generalizes best on the test set.

---

## 9. Model Comparison Predicted vs Actual Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_model_comparison_predicted_vs_actual.png
reports/figures/<dataset>_<llm>_final_model_comparison_predicted_vs_actual.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_<mf_type>_final_quantile_predicted_vs_actual.png
```

### Purpose

These figures compare predicted vs actual plots side by side for several models.

### What It Tells Us

They show visually which model places points closest to the ideal diagonal line.

This is more visual than a metric table because it shows where errors happen:

- low effort projects,
- medium effort projects,
- high effort projects,
- extreme cases.

### Keywords

```text
side-by-side comparison
actual effort
predicted effort
visual accuracy
diagonal reference
model behavior
error distribution
```

### Report Sentence

The comparison predicted vs actual figures show the prediction behavior of multiple models side by side, making it easier to see which model follows the ideal prediction line.

---

## 10. Model Comparison Residual Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_model_comparison_residuals.png
reports/figures/<dataset>_<llm>_final_model_comparison_residuals.png
```

### Purpose

These figures compare residual patterns for multiple models.

### What It Tells Us

They tell us which model has more balanced errors.

Good signs:

- residuals centered around zero,
- no strong curve pattern,
- no large systematic bias,
- fewer extreme residuals.

Bad signs:

- residuals mostly positive or negative,
- fan-shaped residual spread,
- very large residual points,
- clear trend with predicted effort.

### Keywords

```text
residual comparison
error balance
bias
systematic error
variance
diagnostic plot
generalization
```

### Report Sentence

The model comparison residual figures reveal whether different models make systematic errors or whether their residuals remain balanced around zero.

---

## 11. Rule Dominance Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_rule_dominance.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_rule_dominance.png
```

Examples:

```text
albrecht_gpt_rule_dominance.png
desharnais_gemini_rule_dominance.png
reports/figures/quantile/triangular/albrecht_gemini_rule_dominance.png
```

### Purpose

These figures show which fuzzy rules are most active on average.

The plotted value is usually average normalized firing strength.

### What It Tells Us

Rule dominance tells us which rules contribute most often and most strongly to predictions.

If only a few rules dominate heavily, the rule base may be imbalanced.

If many rules have meaningful activation, fuzzification may be distributing samples more evenly across the rule base.

### Keywords

```text
rule dominance
firing strength
normalized firing strength
rule contribution
rule activation
interpretability
dominant rules
fuzzy inference
```

### Report Sentence

The rule dominance figures show which fuzzy rules are most influential in the prediction process. This improves interpretability by identifying the rules that drive model outputs.

---

## 12. Sugeno Surface Figures

Pattern:

```text
reports/figures/<dataset>_<llm>_sugeno_surface.png
reports/figures/quantile/<mf_type>/<dataset>_<llm>_sugeno_surface.png
```

Examples:

```text
albrecht_gpt_sugeno_surface.png
desharnais_claude_sugeno_surface.png
reports/figures/quantile/gaussian/desharnais_gpt_sugeno_surface.png
```

### Purpose

These figures show the Sugeno prediction surface.

Because the models use three inputs, the figure varies two inputs and fixes the third input at `0.5`.

### Axes

For Albrecht:

| Axis           | Meaning            |
| -------------- | ------------------ |
| x-axis         | `RawFPcounts`      |
| y-axis         | `Input`            |
| z-axis         | Predicted `Effort` |
| fixed variable | `File = 0.5`       |

For Desharnais:

| Axis           | Meaning            |
| -------------- | ------------------ |
| x-axis         | `PointsAjust`      |
| y-axis         | `TeamExp`          |
| z-axis         | Predicted `Effort` |
| fixed variable | `Length = 0.5`     |

### What It Tells Us

The surface shows how predicted effort changes when two input variables change.

Steep areas mean the model prediction is sensitive to those input changes.

Flat areas mean the model prediction changes slowly.

Irregular or sharply oscillating surfaces may indicate overfitting, especially in V2.

### Keywords

```text
Sugeno surface
prediction surface
input-output relationship
sensitivity
nonlinear behavior
fixed third variable
predicted effort
model smoothness
overfitting signal
```

### Report Sentence

The Sugeno surface figures visualize how predicted effort changes with two selected input variables while the third input is fixed at a normalized middle value.

---

## 13. All Models Aggregate Comparison Figures

Pattern:

```text
reports/figures/all_models_rmse_comparison.png
reports/figures/all_models_mae_comparison.png
reports/figures/all_models_mape_pct_comparison.png
reports/figures/all_models_r2_comparison.png
```

### Purpose

These figures compare all recorded models across datasets.

They are generated from aggregate comparison CSV files such as:

```text
reports/results/all_models_comparison.csv
```

### What It Tells Us

They provide a high-level model ranking across datasets and LLM rule sources.

They are useful for final report conclusions because they summarize broad performance patterns.

### Keywords

```text
aggregate comparison
all models
RMSE ranking
MAE ranking
MAPE ranking
R2 ranking
overall performance
final evaluation
```

### Report Sentence

The aggregate comparison figures summarize all model results and provide an overall view of which approaches perform better across datasets.

---

## 14. Old vs Full Sugeno Comparison Figures

Pattern:

```text
reports/figures/old_vs_full_sugeno_rmse.png
reports/figures/old_vs_full_sugeno_mae.png
reports/figures/old_vs_full_sugeno_mape_pct.png
```

### Purpose

These figures compare the older label-level Sugeno FIS implementation with the newer full rule-level Sugeno model.

### What It Tells Us

They show whether increasing model flexibility from old label-level Sugeno to full rule-level Sugeno improves or worsens error metrics.

If full rule-level Sugeno has lower training error but worse test error, this supports the overfitting interpretation.

### Keywords

```text
old Sugeno
full rule-level Sugeno
V2 comparison
model complexity
overfitting
RMSE
MAE
MAPE
```

### Report Sentence

The old vs full Sugeno figures compare the earlier Sugeno approach with the full rule-level model and help evaluate whether the added rule-level flexibility improves generalization.

---

## 15. Uniform vs Quantile Comparison Figures

Pattern:

```text
reports/figures/quantile/albrecht_uniform_vs_quantile_rmse.png
reports/figures/quantile/albrecht_uniform_vs_quantile_mape.png
reports/figures/quantile/desharnais_uniform_vs_quantile_rmse.png
reports/figures/quantile/desharnais_uniform_vs_quantile_mape.png
```

### Purpose

These figures compare uniform fuzzification against quantile-based fuzzification.

They use results from:

```text
reports/results/final_uniform_vs_quantile_comparison.csv
```

### What It Tells Us

They answer this question:

```text
Did data-driven quantile fuzzification improve model performance compared with fixed uniform fuzzification?
```

If quantile bars are lower for RMSE or MAPE, quantile fuzzification performed better for that metric.

If uniform bars are lower, fixed boundaries performed better.

### Keywords

```text
uniform fuzzification
quantile fuzzification
data-driven boundaries
RMSE comparison
MAPE comparison
fuzzification impact
membership function comparison
```

### Report Sentence

The uniform vs quantile figures evaluate whether data-driven quartile boundaries improve prediction performance compared with the original fixed fuzzy boundaries.

---

## 16. Quantile MF-Type Experiment Figures

Folder pattern:

```text
reports/figures/quantile/triangular/
reports/figures/quantile/trapezoidal/
reports/figures/quantile/gaussian/
```

### Purpose

These folders store separate quantile experiments for each membership function type.

Each folder contains repeated figure groups:

```text
predicted_vs_actual
residuals
rule_dominance
sugeno_surface
label_sugeno_predicted_vs_actual
label_sugeno_residuals
v2_model_comparison_metrics
final_quantile_model_comparison_metrics
final_quantile_predicted_vs_actual
```

### What It Tells Us

These figures show how the selected membership function shape affects model behavior.

Interpretation:

| MF Type       | What to expect                                              |
| ------------- | ----------------------------------------------------------- |
| `triangular`  | Simple, sharp transitions, easy to interpret                |
| `trapezoidal` | Stable plateau regions, less sensitive around central zones |
| `gaussian`    | Smooth transitions, continuous rule activation              |

### Keywords

```text
MF type comparison
triangular
trapezoidal
gaussian
smoothness
rule activation
prediction stability
quantile experiment
separate experiment folders
```

### Report Sentence

The quantile MF-type experiment folders allow triangular, trapezoidal, and gaussian fuzzy sets to be evaluated separately, making it possible to compare how membership function shape affects prediction quality.

---

## 17. How To Decide If A Figure Shows Good Performance

Use these visual signals:

| Figure Type         | Good Sign                       | Bad Sign                                |
| ------------------- | ------------------------------- | --------------------------------------- |
| Predicted vs Actual | Points close to diagonal        | Points far from diagonal                |
| Residuals           | Random scatter around zero      | Pattern, bias, fan shape                |
| Metrics             | Low RMSE, MAE, MAPE; high R2    | High error metrics; negative R2         |
| Rule Dominance      | Several meaningful active rules | Only one or two rules dominate          |
| Sugeno Surface      | Smooth and explainable surface  | Very sharp, unstable oscillations       |
| Uniform vs Quantile | Lower quantile error bars       | Quantile error higher than uniform      |
| MF Membership Plot  | Clear Low/Medium/High coverage  | Overlapping too much or narrow coverage |

---

## 18. Common Keywords For Report Writing

Use these terms when explaining figures:

```text
prediction accuracy
generalization performance
overfitting
underfitting
residual error
model bias
rule activation
firing strength
membership degree
data-driven fuzzification
uniform fuzzification
quantile fuzzification
model comparison
baseline comparison
interpretability
surface sensitivity
error distribution
test performance
training performance
```

---

## 19. Short Oral Explanation

The figures are grouped to explain the whole modeling process. Outlier plots show data cleaning. Membership function plots show how numerical inputs are converted into fuzzy terms. Predicted vs actual and residual plots show how accurate each model is. Rule dominance plots show which fuzzy rules affect predictions most. Sugeno surface plots show how predicted effort changes with input variables. Metric comparison plots compare Sugeno models with Linear Regression and Decision Tree. Uniform vs quantile plots show whether data-driven fuzzification improves performance.

---

## 20. Turkish Report Paragraph

Bu projedeki grafikler modelleme sürecinin farklı aşamalarını açıklamak için kullanılmıştır. Aykırı değer grafikleri veri temizleme aşamasını gösterirken, üyelik fonksiyonu grafikleri sayısal girdilerin `Low`, `Medium` ve `High` bulanık kümelerine nasıl dönüştürüldüğünü göstermektedir. `Predicted vs Actual` grafikleri tahminlerin gerçek efor değerlerine ne kadar yakın olduğunu, residual grafikleri ise tahmin hatalarının dengeli olup olmadığını gösterir. Rule dominance grafikleri hangi fuzzy kuralların tahmin sürecinde daha etkili olduğunu açıklar. Sugeno surface grafikleri iki girdi değiştiğinde tahmin edilen eforun nasıl değiştiğini görselleştirir. Model karşılaştırma grafikleri Sugeno modellerini Linear Regression ve Decision Tree gibi baseline modellerle karşılaştırır. Uniform vs quantile grafikleri ise veri dağılımına dayalı quantile fuzzification yaklaşımının sabit sınır kullanan uniform fuzzification yaklaşımına göre performans farkını göstermektedir.
