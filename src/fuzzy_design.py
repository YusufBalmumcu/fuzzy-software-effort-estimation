import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from simpful import *

def get_normalized_mfs(var_name):
    """
    Tüm değişkenler 0-1 aralığında normalize edildiği için standart MF'ler kullanılabilir.
    Proje isterlerine göre Gaussian, Triangular ve Trapezoidal türleri bir arada kullanılmıştır.
    """
    # 1. LOW: Trapezoidal MF (0'da başlar, 0.15'e kadar tam üyedir, 0.35'te biter)
    mf_low = TrapezoidFuzzySet(0.0, 0.0, 0.15, 0.35, term="Low")
    
    # 2. MEDIUM: Gaussian MF (0.15 standart sapmalı, 0.5 merkezli)
    mf_medium = GaussianFuzzySet(0.15, 0.5, term="Medium")
    
    # 3. HIGH: Triangular MF (0.6'da başlar, 1.0'da zirve yapar)
    mf_high = TriangleFuzzySet(0.6, 1.0, 1.0, term="High")
    
    return LinguisticVariable([mf_low, mf_medium, mf_high], concept=var_name, universe_of_discourse=[0.0, 1.0])

def setup_albrecht_fuzzy_system():
    """ Albrecht veri seti için Sugeno FIS kurar. """
    FS = FuzzySystem(show_banner=False)
    
    # Girdi değişkenlerini ekle (Albrecht)
    FS.add_linguistic_variable("RawFPcounts", get_normalized_mfs("RawFPcounts"))
    FS.add_linguistic_variable("Input", get_normalized_mfs("Input"))
    FS.add_linguistic_variable("File", get_normalized_mfs("File"))
    
    return FS

def setup_desharnais_fuzzy_system():
    """ Desharnais veri seti için Sugeno FIS kurar. """
    FS = FuzzySystem(show_banner=False)
    
    # Girdi değişkenlerini ekle (Desharnais)
    FS.add_linguistic_variable("PointsAjust", get_normalized_mfs("PointsAjust"))
    FS.add_linguistic_variable("TeamExp", get_normalized_mfs("TeamExp"))
    FS.add_linguistic_variable("Length", get_normalized_mfs("Length"))
    
    return FS

def plot_mfs():
    """ Raporlama için Membership Function grafikleri çizer ve kaydeder. """
    FS = setup_albrecht_fuzzy_system() # Sadece birini çizmek yeterli, MFs aynı.
    
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Simpful plot çizimi (None döndürebilir, bu yüzden unpack etmiyoruz)
    FS.plot_variable("RawFPcounts")
    plt.title("Membership Functions (Normalized [0, 1])")
    plt.xlabel("Normalized Value")
    plt.ylabel("Degree of Membership")
    
    save_path = os.path.join(figures_dir, "membership_functions.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[SAVED] Membership function plot saved to {save_path}")

if __name__ == '__main__':
    plot_mfs()
