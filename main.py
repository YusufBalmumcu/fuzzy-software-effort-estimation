import pandas as pd
import os
import glob
from src.outlier_analysis import OutlierAnalyzer
from src.normalization import DataNormalizer

def main():
    # 1. Configuration
    RAW_DATA_DIR = "data/raw_data/"
    OUTLIER_REMOVED_DIR = "data/processed_data/outlier_removed/"
    FIGURES_DIR = "reports/figures/"
    
    # Çok daha geniş ve esnek efor sütun isimleri listesi
    # Gönderdiğin yeni datasetlerdeki 'dev.eff.hrs.', 'EffortMM', 'MM' vb. eklendi.
    EFFORT_KEYWORDS = [
        'Effort', 'N_effort', 'S_effort', 'Actual_Effort', 
        'dev.eff.hrs.', 'EffortMM', 'Actual.effort', 'MM', 'Actual_effort'
    ] 
    
    print("--- Phase 1: Advanced Batch Outlier Analysis ---")

    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))

    if not csv_files:
        print(f"ERROR: No CSV files found in {RAW_DATA_DIR}")
        return

    analyzer = OutlierAnalyzer(output_path=FIGURES_DIR)

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        dataset_name = os.path.splitext(file_name)[0]
        
        print(f"\n[SCANNING] {file_name}...")

        try:
            # Bazı dosyalar virgül yerine noktalı virgül kullanıyor olabilir 
            # veya isimlerinde tırnak işareti olabilir.
            df = pd.read_csv(file_path)
            
            # Sütun isimlerini temizle (başındaki/sonundaki boşluk ve tırnakları sil)
            df.columns = df.columns.str.replace("'", "").str.strip()
            
            found_col = None
            for col in EFFORT_KEYWORDS:
                if col in df.columns:
                    found_col = col
                    break
            
            # Eğer tam eşleşme bulamazsa, içinde 'effort' geçen sütunları ara (Case-insensitive)
            if found_col is None:
                for col in df.columns:
                    if 'effort' in col.lower() or 'eff.' in col.lower():
                        found_col = col
                        break

            if found_col is None:
                print(f"!!! WARNING: Target column not found in {file_name}. Columns: {list(df.columns[:5])}...")
                continue

            print(f"--> Target column identified: '{found_col}'")

            # Outlier temizliği
            df_outlier_free = analyzer.detect_and_remove_outliers(
                df, 
                column_name=found_col, 
                dataset_name=dataset_name.capitalize()
            )

            # Checkpoint kaydı
            if not os.path.exists(OUTLIER_REMOVED_DIR):
                os.makedirs(OUTLIER_REMOVED_DIR, exist_ok=True)
                
            output_save_path = os.path.join(OUTLIER_REMOVED_DIR, f"{dataset_name}_outlier_removed.csv")
            df_outlier_free.to_csv(output_save_path, index=False)
            
            print(f"[DONE] Saved: {output_save_path}")

        except Exception as e:
            print(f"FAILED to process {file_name}: {str(e)}")

    print("\n--- Batch processing finished ---")

    # --- Phase 2: Normalization ---
    print("\n--- Phase 2: Normalization Started ---")
    OUTLIER_REMOVED_DIR = "data/processed_data/outlier_removed/"
    
    normalizer = DataNormalizer()
    normalizer.process_all_files(OUTLIER_REMOVED_DIR)
    
    print("\n[SUCCESS] Tüm veri setleri temizlendi ve normalize edildi.")



if __name__ == "__main__":
    main()