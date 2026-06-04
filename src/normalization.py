import glob
import os

import pandas as pd


class DataNormalizer:
    # Veri setlerinde kullanılan farklı efor sütunu isimleri
    EFFORT_KEYWORDS = [
        'Effort', 'N_effort', 'S_effort', 'Actual_Effort',
        'dev.eff.hrs.', 'EffortMM', 'Actual.effort', 'MM', 'Actual_effort'
    ]

    def __init__(self, output_path="data/processed_data/final_normalized/"):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def find_effort_column(self, df):
        # Önce bilinen efor sütunu isimleriyle tam eşleşme ara
        for col in self.EFFORT_KEYWORDS:
            if col in df.columns:
                return col

        # Tam eşleşme yoksa sütun adının içinde efor ifadesi geçiyor mu kontrol et
        for col in df.columns:
            lower_col = col.lower()
            if 'effort' in lower_col or 'eff.' in lower_col:
                return col

        return None

    def min_max_normalize(self, df):
        """
        Efor sütunu hariç tüm sayısal sütunlara Min-Max normalizasyonu uygular.
        Efor sütununu orijinal değeriyle korur ve en sona taşır.
        """
        effort_col = self.find_effort_column(df)

        # Sadece sayısal (int ve float) sütunları seç
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Efor sütunu hedef değişken olduğu için normalize edilmez
        cols_to_normalize = [col for col in numeric_cols if col != effort_col]

        df_norm = df.copy()
        for col in cols_to_normalize:
            min_val = df[col].min()
            max_val = df[col].max()

            # Paydanın 0 olmaması için kontrol et (tüm değerler aynıysa 0 yap)
            if max_val - min_val != 0:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0

        # Efor sütununu çıktı dosyasında son sütun olarak konumlandır
        if effort_col is not None:
            ordered_cols = [col for col in df_norm.columns if col != effort_col] + [effort_col]
            df_norm = df_norm[ordered_cols]

        return df_norm

    def process_all_files(self, input_dir):
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

        if not csv_files:
            print(f"Uyarı: {input_dir} içinde CSV dosyası bulunamadı.")
            return

        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            df = pd.read_csv(file_path)

            print(f"[NORMALIZING] {file_name}...")
            df_normalized = self.min_max_normalize(df)

            save_path = os.path.join(self.output_path, file_name.replace("outlier_removed", "normalized"))
            df_normalized.to_csv(save_path, index=False)
            print(f"[SAVED] {save_path}")


# Bağımsız test için
if __name__ == "__main__":
    normalizer = DataNormalizer()
    normalizer.process_all_files("data/processed_data/outlier_removed/")
