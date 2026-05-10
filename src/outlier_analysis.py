import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class OutlierAnalyzer:
    def __init__(self, output_path="reports/figures/"):
        """
        Initializes the analyzer. output_path is relative to where main.py runs.
        """
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def detect_and_remove_outliers(self, df, column_name, dataset_name="Dataset"):
        """
        Uses IQR to clean outliers and saves a boxplot for the report.
        """
        # IQR Calculation
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
        
        print(f"\n[ANALYSIS] {dataset_name} - Column: {column_name}")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"Detected Outliers: {len(outliers)}")
        
        # Visualization for the report
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df[column_name], color='salmon')
        plt.title(f'Before Cleaning: {column_name}')
        
        df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].copy()
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df_cleaned[column_name], color='skyblue')
        plt.title(f'After Cleaning: {column_name}')
        
        plt.tight_layout()
        
        # Save figure
        fig_name = f"outlier_{dataset_name.lower()}.png"
        save_path = os.path.join(self.output_path, fig_name)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.close() # Close plot to free memory
        
        return df_cleaned