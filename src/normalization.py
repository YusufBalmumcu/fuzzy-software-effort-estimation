import pandas as pd
import os
import glob

class DataNormalizer:
    def __init__(self, output_path="data/processed_data/final_normalized/"):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def min_max_normalize(self, df):
        """
        Sadece sayısal sütunları bulur ve Min-Max normalizasyonu uygular.
        """
        # Sadece sayısal (int ve float) sütunları seç
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # ID gibi normalize edilmemesi gereken sütunları hariç tut (isteğe bağlı)
        cols_to_normalize = [col for col in numeric_cols if col.lower() not in ['id', 'project_id']]

        df_norm = df.copy()
        for col in cols_to_normalize:
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Payda 0 olmasın diye kontrol (tüm değerler aynıysa 0 yap)
            if max_val - min_val != 0:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0
                
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