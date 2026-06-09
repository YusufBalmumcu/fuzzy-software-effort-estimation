# Sugeno Katsayılarının Projede Nasıl Öğrenildiği

Bu doküman, **"Sugeno Fuzzy Inference System ile Yazılım Efor Tahmini Yapılması"** projesinde Sugeno çıktı denklemlerindeki katsayıların nasıl bulunduğunu açıklar.

Ana soru şudur:

> Sugeno modelindeki `a1`, `a2`, `a3` ve `bias` katsayıları elle mi yazıldı, yoksa model tarafından mı öğrenildi?

Bu projede cevap:

> Katsayılar elle belirlenmedi. Eğitim verisi kullanılarak otomatik olarak öğrenildi.

Ancak önemli bir teknik ayrım vardır:

- Eski `src/sugeno_model.py` içindeki `SugenoEffortModel`, `scipy.optimize.minimize` ve `L-BFGS-B` ile RMSE'yi iteratif olarak minimize eder.
- Güncel ana modeller olan `Sugeno V1 Label-Level` ve `Sugeno V2 Full Rule-Level`, katsayıları iteratif optimizer ile değil, **tasarım matrisi** kurup **regularized least squares / ridge regression** denklemini çözerek öğrenir.

Güncel raporlarda kullanılan V1 ve V2 için asıl katsayı öğrenme denklemi şudur:

```text
theta = (Phi.T @ Phi + lambda I)^(-1) @ Phi.T @ y
```

Kodda bu işlem şu satır mantığıyla yapılır:

```python
reg = self.regularization * np.eye(phi.shape[1], dtype=float)
params = np.linalg.solve(phi.T @ phi + reg, phi.T @ y)
```

Bu akış `src/label_level_sugeno_model.py` ve `src/full_sugeno_model.py` dosyalarındaki `fit()` fonksiyonlarında bulunur.

---

## 1. Birinci Derece Sugeno Kuralı Nedir?

Birinci derece Sugeno kuralı iki bölümden oluşur:

```text
IF RawFPcounts is Low AND Input is Low AND File is Low
THEN f_i(x) = a_i1*RawFPcounts + a_i2*Input + a_i3*File + c_i
```

Burada:

| Bölüm | Anlamı |
|---|---|
| `IF` kısmı | Bulanık koşul bölümüdür. Girdilerin `Low`, `Medium`, `High` gibi üyelik derecelerini kullanır. |
| `THEN` kısmı | Sugeno çıktı denklemidir. Birinci derece olduğu için doğrusal bir fonksiyondur. |
| `a_i1`, `a_i2`, `a_i3` | Girdi değişkenlerinin katsayılarıdır. |
| `c_i` veya `bias` | Sabit terimdir. |

Örnek olarak 3 girdili bir kural için:

```text
f_i(x) = a_i1*x1 + a_i2*x2 + a_i3*x3 + c_i
```

Bu projede `x1`, `x2`, `x3` veri setine göre değişir.

---

## 2. Kullanılan Girdi Değişkenleri

Girdi değişkenleri `src/full_sugeno_model.py` içindeki `DATASET_CONFIG` sözlüğünde tanımlıdır. Aynı yapı V1 modelinde de kullanılır.

### Albrecht

| Sembol | Değişken |
|---|---|
| `x1` | `RawFPcounts` |
| `x2` | `Input` |
| `x3` | `File` |

Yorum:

- `RawFPcounts`, yazılımın fonksiyonel büyüklüğünü temsil eder.
- `Input`, sisteme giren veri/işlev miktarıyla ilişkilidir.
- `File`, yazılım yapısındaki dosya sayısı veya karmaşıklık göstergesi olarak kullanılır.

### Desharnais

| Sembol | Değişken |
|---|---|
| `x1` | `PointsAjust` |
| `x2` | `TeamExp` |
| `x3` | `Length` |

Yorum:

- `PointsAjust`, fonksiyonel büyüklük/ayarlanmış fonksiyon puanı göstergesidir.
- `TeamExp`, ekibin deneyimini temsil eder.
- `Length`, proje süresi veya büyüklüğüyle ilişkilidir.

Hedef değişken her iki veri setinde de:

```text
Effort
```

---

## 3. Fuzzification: Katsayı Öğrenmeden Önce Ne Olur?

Sugeno katsayıları öğrenilmeden önce sayısal girdiler bulanık değerlere dönüştürülür.

Ön işleme sırasında girdiler normalize edilir. Model, normalize edilmiş `0-1` aralığındaki değerleri kullanır. `Effort` ise metriklerin gerçek efor ölçeğinde kalması için orijinal ölçekte korunur.

### 3.1 Uniform Fuzzification

Eski/varsayılan üyelik fonksiyonu yapısı `src/manual_sugeno_engine.py` içindeki `membership_degree()` fonksiyonunda tanımlıdır:

| Terim | Kullanılan şekil |
|---|---|
| `Low` | Trapezoidal |
| `Medium` | Gaussian |
| `High` | Triangular |

Bu yapı `src/fuzzy_design.py` ile aynı mantığı takip eder:

```text
Low    -> düşük değerlerde yüksek üyelik
Medium -> 0.5 civarında yüksek üyelik
High   -> yüksek değerlerde yüksek üyelik
```

Uniform fuzzification basit ve yorumlanabilirdir; ancak her özelliğin gerçek veri dağılımını dikkate almaz.

### 3.2 Quantile Fuzzification

Yeni eklenen quantile tabanlı yapı `src/fuzzy_design_quantile.py` içindedir.

Her özellik için şu değerler hesaplanır:

```text
Q1 = 25. yüzdelik
Q2 = medyan / 50. yüzdelik
Q3 = 75. yüzdelik
```

Sonra `Low`, `Medium`, `High` üyelik fonksiyonları bu değerlere göre kurulur. Örneğin triangular quantile için:

```text
Low    = [0, 0, Q2]
Medium = [Q1, Q2, Q3]
High   = [Q2, 1, 1]
```

Quantile yaklaşımı her veri setinin kendi dağılımına uyum sağlar. Bu nedenle kural aktivasyonlarını daha dengeli hale getirebilir.

---

## 4. Kural Ateşleme Gücü Nasıl Hesaplanır?

Her kuralın `IF` kısmındaki koşullar için üyelik dereceleri hesaplanır. Sonra bu üyelik dereceleri çarpılır.

Formül:

```text
w_i = product of membership degrees
```

3 girdili bir örnek:

```text
RawFPcounts_Low = 0.8
Input_Low       = 0.6
File_Low        = 0.5
```

Bu durumda kuralın ateşleme gücü:

```text
w_1 = 0.8 * 0.6 * 0.5 = 0.24
```

Kod karşılığı:

- V1 için: `LabelLevelSugenoModel.firing_strengths_for_row()`
- V2 için: `ManualSugenoEngine.firing_strengths_for_row()`

Her iki fonksiyonda da her koşulun üyelik derecesi çarpılarak kuralın ham ateşleme gücü bulunur.

---

## 5. Normalize Kural Ağırlıkları

Ham ateşleme güçleri doğrudan kullanılmaz. Önce normalize edilir:

```text
w_bar_i = w_i / sum(w_i)
```

Burada:

- `w_i`: i. kuralın ham ateşleme gücü
- `w_bar_i`: i. kuralın normalize edilmiş katkı ağırlığı

Yorum:

- `w_bar_i` yüksekse, o kural tahmini daha fazla etkiler.
- `w_bar_i` sıfıra yakınsa, o kuralın katkısı çok azdır.
- Bütün normalize ağırlıkların toplamı 1 olur.

Kod karşılığı:

- V1: `LabelLevelSugenoModel.normalized_strengths_for_row()`
- V2: `ManualSugenoEngine.normalized_strengths_for_row()`

Eğer hiçbir kural ateşlenmezse, mevcut kod sessizce sıfır döndürmez. Tüm kurallara eşit ağırlık verir:

```python
normalized = np.ones(len(self.rules), dtype=float) / len(self.rules)
```

Bu, tahminin sayısal olarak tanımlı kalmasını sağlar.

---

## 6. Sugeno Tahmini Nasıl Hesaplanır?

Genel Sugeno tahmini:

```text
y_hat = sum(w_bar_i * f_i(x))
```

Burada:

- `f_i(x)`: i. kuralın veya ilgili etiketin doğrusal Sugeno çıktı denklemi
- `w_bar_i`: normalize kural ağırlığı
- `y_hat`: tahmin edilen `Effort`

Yani model tek bir global denklem kullanmaz. Birden fazla yerel doğrusal çıktı denkleminden gelen sonuçları, bulanık kural ağırlıklarıyla birleştirir.

---

## 7. Sugeno V1 ve V2 Arasındaki Fark

Projede iki güncel Sugeno modeli vardır:

1. `Sugeno V1 Label-Level First-Order Sugeno`
2. `Sugeno V2 Full Rule-Level First-Order Sugeno`

### 7.1 Sugeno V1 Label-Level

Kod dosyası:

```text
src/label_level_sugeno_model.py
```

V1'de 20 kural vardır; ancak her kural ayrı denklem öğrenmez. Kurallar 5 çıktı etiketinden birine bağlanır:

```text
Very_Low
Low
Medium
High
Very_High
```

Her çıktı etiketi için bir doğrusal denklem öğrenilir:

```text
Very_Low(x)  = a1*x1 + a2*x2 + a3*x3 + b
Low(x)       = a1*x1 + a2*x2 + a3*x3 + b
Medium(x)    = a1*x1 + a2*x2 + a3*x3 + b
High(x)      = a1*x1 + a2*x2 + a3*x3 + b
Very_High(x) = a1*x1 + a2*x2 + a3*x3 + b
```

Her denklemde 3 girdi katsayısı ve 1 bias vardır:

```text
3 + 1 = 4 parametre
```

5 çıktı etiketi olduğu için:

```text
5 * (3 + 1) = 20 parametre
```

Kodda:

```python
OUTPUT_LABELS = ["Very_Low", "Low", "Medium", "High", "Very_High"]
params_per_label = len(input_vars) + 1
total_params = len(output_labels) * params_per_label
```

### 7.2 Sugeno V2 Full Rule-Level

Kod dosyası:

```text
src/full_sugeno_model.py
```

V2'de her kural kendi çıktı denklemine sahiptir. Kurallar `R1_OUT`, `R2_OUT`, ..., `R20_OUT` şeklinde dönüştürülür.

Örnek:

```text
R1_OUT  = a11*x1 + a12*x2 + a13*x3 + c1
R2_OUT  = a21*x1 + a22*x2 + a23*x3 + c2
...
R20_OUT = a201*x1 + a202*x2 + a203*x3 + c20
```

20 kural ve her kural için 4 parametre olduğundan:

```text
20 * (3 + 1) = 80 parametre
```

Kodda bu dönüşüm:

```text
src/rule_converter.py -> convert_rules_to_rule_level()
```

fonksiyonu ile yapılır.

---

## 8. Katsayı Vektörü Nedir?

Modelin öğrendiği bütün sayılar tek bir vektör olarak düşünülebilir. Bu vektöre `theta` diyelim.

### V1 için theta

V1'de theta şu bloklardan oluşur:

```text
theta =
[
  Very_Low katsayıları,
  Low katsayıları,
  Medium katsayıları,
  High katsayıları,
  Very_High katsayıları
]
```

3 girdi varsa her blok:

```text
[a1, a2, a3, bias]
```

Toplam:

```text
5 * 4 = 20 değer
```

Kodda öğrenilen parametreler şu şekle getirilir:

```python
self.coefficients = params.reshape(len(self.output_labels), self.params_per_label)
```

### V2 için theta

V2'de theta şu bloklardan oluşur:

```text
theta =
[
  R1 katsayıları,
  R2 katsayıları,
  ...
  R20 katsayıları
]
```

Her blok:

```text
[a_i1, a_i2, a_i3, bias_i]
```

Toplam:

```text
20 * 4 = 80 değer
```

Kodda:

```python
self.coefficients = params.reshape(len(self.rules), self.params_per_rule)
```

---

## 9. Güncel V1/V2 Kodunda Katsayılar Nasıl Öğreniliyor?

Bu projedeki güncel V1 ve V2 modelleri katsayıları **lineer cebirle** öğrenir.

Ana fikir:

1. Her eğitim satırı için bulanık kural ağırlıkları hesaplanır.
2. Bu ağırlıklar ve girdi değerleri kullanılarak bir tasarım matrisi oluşturulur.
3. Hedef vektör `y = Effort` alınır.
4. Katsayı vektörü `theta`, regularized least squares denklemiyle çözülür.

### 9.1 Tasarım Matrisi

Tasarım matrisi için sembol:

```text
Phi
```

Her satır bir eğitim örneğini temsil eder. Her sütun ise öğrenilecek katsayılardan birine karşılık gelir.

Tahmin şu hale getirilir:

```text
y_hat = Phi @ theta
```

Bu form çok önemlidir. Çünkü Sugeno tahmini bulanık ağırlıklardan dolayı karmaşık görünse de, katsayılar açısından doğrusal hale getirilebilir.

### 9.2 V2 Tasarım Matrisi

V2'de her kuralın ayrı denklemi vardır.

Bir satır için:

```text
x_aug = [x1, x2, x3, 1]
```

Her kural için bu değerler normalize kural ağırlığıyla çarpılır:

```text
phi_row =
[
  w_bar_1*x1,  w_bar_1*x2,  w_bar_1*x3,  w_bar_1,
  w_bar_2*x1,  w_bar_2*x2,  w_bar_2*x3,  w_bar_2,
  ...
  w_bar_20*x1, w_bar_20*x2, w_bar_20*x3, w_bar_20
]
```

Kod karşılığı:

```text
src/manual_sugeno_engine.py -> ManualSugenoEngine.design_matrix()
```

Bu fonksiyonda:

```python
for weight in normalized:
    design_row.extend([weight * value for value in values])
    design_row.append(weight)
```

### 9.3 V1 Tasarım Matrisi

V1'de her kuralın ayrı denklemi yoktur; her çıktı etiketinin ayrı denklemi vardır.

Bu nedenle aynı etikete giden kuralların normalize ağırlıkları aynı etiket bloğuna eklenir.

Kod karşılığı:

```text
src/label_level_sugeno_model.py -> LabelLevelSugenoModel.design_matrix()
```

Temel kod mantığı:

```python
for rule_idx, rule in enumerate(self.rules):
    label_idx = self.label_to_index[rule["consequent_label"]]
    start = label_idx * self.params_per_label
    design_row[start:start + self.params_per_label] += normalized[rule_idx] * x_aug
```

Bu şu anlama gelir:

```text
Aynı çıktı etiketine bağlı kurallar, o etiketin tek ortak denklemine katkı verir.
```

---

## 10. Katsayı Öğrenme Denklemi

Hem V1 hem V2 için `fit()` fonksiyonu aynı matematiksel çözümü kullanır.

Kod:

```python
X = df[self.input_vars]
y = df[self.target_col].to_numpy(dtype=float)
phi = self.design_matrix(X)

reg = self.regularization * np.eye(phi.shape[1], dtype=float)
params = np.linalg.solve(phi.T @ phi + reg, phi.T @ y)
```

Matematiksel olarak:

```text
theta = (Phi^T Phi + lambda I)^(-1) Phi^T y
```

Burada:

| Sembol | Anlamı |
|---|---|
| `Phi` | Tasarım matrisi |
| `theta` | Öğrenilecek katsayı vektörü |
| `y` | Gerçek `Effort` değerleri |
| `lambda` | Regularization katsayısı |
| `I` | Birim matris |

Bu çözüm, şu objective fonksiyonunu minimize eder:

```text
minimize ||y - Phi*theta||^2 + lambda * ||theta||^2
```

Yani model:

- gerçek efor değerleri ile tahminler arasındaki kare hatayı küçültür,
- aynı zamanda katsayıların aşırı büyümesini regularization ile sınırlar.

Regularization değerleri:

| Model | Varsayılan regularization |
|---|---:|
| Sugeno V1 Label-Level | `1e-2` |
| Sugeno V2 Full Rule-Level | `1e-6` |

Eğer `np.linalg.solve()` başarısız olursa kod şu yedeği kullanır:

```python
params = np.linalg.lstsq(phi, y, rcond=None)[0]
```

Bu da least squares çözümüdür.

---

## 11. RMSE Nerede Kullanılıyor?

Güncel V1/V2 kodunda RMSE optimizer'ın doğrudan minimize ettiği fonksiyon değildir. Katsayılar regularized least squares ile bulunur. Eğitimden sonra model tahmin üretir ve metrikler hesaplanır.

RMSE formülü:

```text
RMSE = sqrt((1/N) * sum((y_i - y_hat_i)^2))
```

Kod karşılığı:

```text
src/evaluation.py -> regression_metrics()
```

Bu fonksiyon şunları hesaplar:

- `RMSE`
- `MAE`
- `MAPE (%)`
- `R2`

Özet:

```text
Güncel V1/V2:
katsayı öğrenme -> ridge / least squares
performans ölçme -> RMSE, MAE, MAPE, R2
```

---

## 12. Eski SugenoEffortModel Akışı: L-BFGS-B ile Optimizasyon

Projede eski bir model sınıfı da vardır:

```text
src/sugeno_model.py -> SugenoEffortModel
```

Bu sınıf, katsayıları gerçekten iteratif optimizer ile bulur.

Kod akışı:

1. `train(data_path)` veri setini okur.
2. `initial_weights = np.ones(self.total_params)` ile başlangıç katsayılarını oluşturur.
3. `minimize()` çağrılır.
4. `_objective_function()` içinde:
   - `set_output_functions(weights)` ile Sugeno çıktı fonksiyonları kurulur,
   - `predict(X)` ile tahmin yapılır,
   - RMSE hesaplanır.
5. `L-BFGS-B` algoritması katsayıları adım adım değiştirir.
6. En düşük RMSE'yi veren katsayılar `self.optimized_weights` içine yazılır.

Kod:

```python
res = minimize(
    self._objective_function,
    initial_weights,
    args=(X, y),
    method='L-BFGS-B',
    options={'maxiter': 50}
)
```

Bu eski akış şu objective fonksiyonunu kullanır:

```text
minimize RMSE
```

Fakat güncel raporlanan V1/V2 modelleri bu eski sınıfı değil, `LabelLevelSugenoModel` ve `FullRuleSugenoModel` sınıflarını kullanır.

---

## 13. Denklemler Nereye Kaydediliyor?

Öğrenilen katsayılar eğitimden sonra JSON ve TXT dosyalarına kaydedilir.

### Uniform V1

```text
models/sugeno_label_equations/
```

Örnek:

```text
models/sugeno_label_equations/albrecht_gpt_label_equations.json
models/sugeno_label_equations/albrecht_gpt_label_equations.txt
```

### Uniform V2

```text
models/sugeno_equations/
```

Örnek:

```text
models/sugeno_equations/albrecht_gpt_equations.json
models/sugeno_equations/albrecht_gpt_equations.txt
```

### Quantile V1

```text
models/sugeno_label_equations_quantile/<mf_type>/
```

Örnek:

```text
models/sugeno_label_equations_quantile/triangular/albrecht_gpt_label_equations.json
models/sugeno_label_equations_quantile/gaussian/desharnais_claude_label_equations.txt
```

### Quantile V2

```text
models/sugeno_equations_quantile/<mf_type>/
```

Örnek:

```text
models/sugeno_equations_quantile/trapezoidal/desharnais_gemini_equations.json
```

JSON dosyaları makine tarafından okunabilir yapıdadır. TXT dosyaları ise rapora veya sunuma koymak için daha okunabilir formattadır.

Her kayıtta:

- denklem metni,
- girdi katsayıları,
- bias,
- model tipi,
- veri seti,
- LLM kural kaynağı,
- fuzzification tipi

bulunur.

---

## 14. Linear Regression ile Sugeno Katsayı Öğrenme Farkı

`Linear Regression` baseline modeli `src/evaluation.py` içinde `predict_with_baselines()` fonksiyonunda eğitilir.

Linear Regression tek bir global denklem öğrenir:

```text
y_hat = a1*x1 + a2*x2 + a3*x3 + b
```

Bu denklem tüm veri uzayı için aynıdır.

Sugeno ise birden fazla yerel denklem kullanır:

```text
y_hat = sum(w_bar_i * f_i(x))
```

Temel fark:

| Model | Öğrendiği yapı |
|---|---|
| Linear Regression | Tüm veri için tek global doğrusal denklem |
| Decision Tree | Kurallara benzer ağaç bölmeleri; doğrusal Sugeno katsayısı yok |
| Sugeno V1 | Her çıktı etiketi için bir yerel doğrusal denklem |
| Sugeno V2 | Her kural için bir yerel doğrusal denklem |

Sugeno'da hangi denklemin ne kadar etkili olacağını fuzzy kurallar belirler.

---

## 15. V2 Neden Overfitting Yaptı?

V2 modeli daha esnektir; çünkü her kural için ayrı denklem öğrenir.

Parametre sayısı:

```text
20 kural * 4 parametre = 80 parametre
```

Bu özellikle Albrecht gibi küçük veri setlerinde risklidir. Örneğin uniform sonuçlarda:

| Dataset | LLM | Model | Parametre | Train RMSE | Test RMSE |
|---|---|---|---:|---:|---:|
| Albrecht | Gemini | Sugeno V2 | 80 | yaklaşık 0.001 | yaklaşık 9.31 |
| Albrecht | GPT | Sugeno V2 | 80 | yaklaşık 0.001 | yaklaşık 9.32 |
| Albrecht | Claude | Sugeno V2 | 80 | yaklaşık 0.001 | yaklaşık 18.13 |

Train RMSE'nin neredeyse sıfır olması ilk bakışta iyi görünebilir. Fakat test hatasının yüksek olması şunu gösterir:

```text
Model eğitim verisini ezberlemiştir; yeni veriye iyi genellememiştir.
```

Desharnais için de uniform V2 sonuçlarında eğitim hatası test hatasına göre çok düşüktür. Örneğin `reports/results/full_sugeno_summary.csv` içinde Desharnais V2 test RMSE değerleri oldukça yüksektir.

Bu nedenle:

```text
Düşük Train RMSE tek başına iyi model anlamına gelmez.
```

---

## 16. V1 Neden Daha Stabil?

V1 daha az parametre öğrenir:

```text
5 etiket * 4 parametre = 20 parametre
```

Bu modelde birden fazla kural aynı çıktı etiketi denklemine katkı verir. Yani kurallar tamamen bağımsız 20 ayrı denklem öğrenmek yerine 5 ortak denklem üzerinden çalışır.

Bu nedenle:

- model karmaşıklığı azalır,
- küçük veri setinde aşırı öğrenme riski düşer,
- test performansı daha stabil olur.

Projede `reports/results/final_all_models_summary.csv` içinde V1 sonuçları V2'ye göre daha dengeli görünmektedir. Örneğin Albrecht için bazı LLM'lerde V1 test RMSE değeri V2'den daha düşüktür.

---

## 17. Küçük Sayısal Örnek

Bir kural için Sugeno çıktı denklemi:

```text
f_1(x) = a1*x1 + a2*x2 + a3*x3 + b
```

Örnek girdi:

```text
x1 = 0.6
x2 = 0.8
x3 = 0.5
w_bar = 0.7
```

Varsayılan öğrenilmiş katsayılar:

```text
a1 = 20
a2 = 10
a3 = 5
b  = 3
```

Denklem çıktısı:

```text
f_1(x) = 20*0.6 + 10*0.8 + 5*0.5 + 3
f_1(x) = 12 + 8 + 2.5 + 3
f_1(x) = 25.5
```

Bu kuralın tahmine katkısı:

```text
w_bar * f_1(x) = 0.7 * 25.5 = 17.85
```

Final tahmin, aktif olan bütün kuralların bu şekilde hesaplanan katkılarının toplamıdır:

```text
y_hat = contribution_1 + contribution_2 + ... + contribution_n
```

---

## 18. Kod Akışı

### 18.1 Sugeno V1 Label-Level Akışı

İlgili dosyalar:

```text
scripts/run_label_level_sugeno.py
src/label_level_sugeno_model.py
src/manual_sugeno_engine.py
src/rules.py
src/evaluation.py
```

Adımlar:

1. `scripts/run_label_level_sugeno.py` çalıştırılır.
2. `LabelLevelSugenoModel(dataset_name, llm_name)` oluşturulur.
3. `DATASET_CONFIG` üzerinden girdi değişkenleri ve hedef sütun seçilir.
4. `get_all_rules()` ile `models/rules_*.json` dosyasından kurallar okunur.
5. `_parse_rules()` ile kurallar koşullara ve çıktı etiketlerine ayrılır.
6. `load_training_frame()` normalize girdileri ve orijinal `Effort` değerini yükler.
7. `train_test_split()` ile eğitim/test ayrımı yapılır.
8. `fit(train_df)` çağrılır.
9. `design_matrix()` her eğitim satırı için fuzzy ağırlıklarla `Phi` matrisini oluşturur.
10. `np.linalg.solve(phi.T @ phi + reg, phi.T @ y)` ile katsayılar öğrenilir.
11. Katsayılar `self.coefficients` içine yazılır.
12. `predict()` ile tahminler üretilir.
13. `regression_metrics()` ile RMSE, MAE, MAPE, R2 hesaplanır.
14. `save_equations()` JSON/TXT denklem dosyalarını kaydeder.
15. `save_predictions()`, `save_rule_analysis()`, `save_plots()` rapor çıktılarını kaydeder.

### 18.2 Sugeno V2 Full Rule-Level Akışı

İlgili dosyalar:

```text
scripts/run_full_sugeno.py
src/full_sugeno_model.py
src/manual_sugeno_engine.py
src/rule_converter.py
src/rules.py
src/evaluation.py
```

Adımlar:

1. `scripts/run_full_sugeno.py` çalıştırılır.
2. `FullRuleSugenoModel(dataset_name, llm_name)` oluşturulur.
3. `get_all_rules()` ile kurallar yüklenir.
4. `convert_rules_to_rule_level()` her kuralı `R_i_OUT` çıktısına dönüştürür.
5. `ManualSugenoEngine` kuralları ve girdileri alır.
6. `fit(train_df)` çağrılır.
7. `ManualSugenoEngine.design_matrix()` V2 tasarım matrisini oluşturur.
8. `np.linalg.solve()` ile 80 parametre öğrenilir.
9. `engine.set_coefficients()` ile katsayılar inference motoruna aktarılır.
10. Tahminler, metrikler, denklemler, rule contribution dosyaları ve grafikler kaydedilir.

### 18.3 Quantile Sugeno Akışı

İlgili dosyalar:

```text
scripts/run_quantile_fuzzification.py
scripts/run_label_level_sugeno_quantile.py
scripts/run_full_sugeno_quantile.py
src/fuzzy_design_quantile.py
```

Adımlar:

1. `generate_quantile_fuzzification_for_dataset()` Q1/Q2/Q3 istatistiklerini hesaplar.
2. Membership parametreleri `models/quantile_fuzzy_params/` altına kaydedilir.
3. `load_quantile_membership_function()` seçilen MF tipine göre üyelik fonksiyonu sağlar.
4. Bu membership function, V1 veya V2 model constructor'ına verilir.
5. Katsayı öğrenme kısmı yine aynı kalır:

```text
theta = (Phi.T Phi + lambda I)^(-1) Phi.T y
```

Yani quantile yaklaşımı katsayı öğrenme matematiğini değil, kural ağırlıklarını değiştiren fuzzification aşamasını değiştirir.

---

## 19. Rapor İçin Kısa Açıklama

Bu projede Sugeno modelinin çıktı denklemlerindeki katsayılar elle belirlenmemiş, eğitim verisi üzerinden otomatik olarak öğrenilmiştir. Sugeno V1 Label-Level modelinde her çıktı etiketi için bir doğrusal denklem öğrenilmiş ve 5 çıktı etiketi için toplam 20 parametre elde edilmiştir. Sugeno V2 Full Rule-Level modelinde ise her fuzzy kural için ayrı bir doğrusal denklem öğrenilmiş ve 20 kural için toplam 80 parametre elde edilmiştir. Güncel V1 ve V2 uygulamalarında katsayılar, fuzzy kural ağırlıklarıyla oluşturulan tasarım matrisi üzerinden regularized least squares yöntemiyle hesaplanmıştır. Eğitim sonrasında RMSE, MAE, MAPE ve R2 metrikleriyle model başarısı ölçülmüştür. V2 modeli daha fazla parametreye sahip olduğu için özellikle küçük veri setlerinde eğitim verisini çok iyi öğrenmiş, fakat test verisinde daha zayıf performans göstererek overfitting eğilimi sergilemiştir. V1 modeli daha az parametre kullandığı ve kuralları ortak çıktı etiketi denklemleri altında topladığı için daha stabil sonuçlar üretmiştir.

---

## 20. Öğretmene Nasıl Anlatılır?

Hocam, Sugeno modelinde kuralların `THEN` kısmındaki katsayıları elle belirlemedik. Önce her kuralın veya çıktı etiketinin doğrusal denklem formunu tanımladık. Daha sonra eğitim verisindeki her satır için fuzzy üyelik derecelerini hesapladık, bu derecelerden kural ağırlıklarını bulduk ve bu ağırlıklarla bir tasarım matrisi oluşturduk. Güncel V1 ve V2 modellerinde katsayılar bu tasarım matrisi üzerinden regularized least squares yöntemiyle otomatik olarak hesaplandı. V1 modelinde her çıktı etiketi için bir denklem öğrenildiği için 20 parametre vardır. V2 modelinde her kural için ayrı denklem öğrenildiği için 80 parametre vardır. V2 daha esnek olduğu için eğitim hatası çok düşük çıkabilir; fakat bu her zaman iyi değildir, çünkü küçük veri setlerinde modeli ezberlemeye yani overfitting'e götürebilir.

---

## 21. İncelenen Kod Dosyaları

Bu açıklama hazırlanırken şu dosyalar incelendi:

```text
src/sugeno_model.py
src/label_level_sugeno_model.py
src/full_sugeno_model.py
src/manual_sugeno_engine.py
src/fuzzy_design.py
src/fuzzy_design_quantile.py
src/rules.py
src/evaluation.py
scripts/run_label_level_sugeno.py
scripts/run_full_sugeno.py
scripts/run_label_level_sugeno_quantile.py
scripts/run_full_sugeno_quantile.py
```

---

## 22. Sonuç

Projede Sugeno katsayı öğrenme süreci şu şekilde özetlenebilir:

1. Girdi değerleri fuzzy üyelik derecelerine dönüştürülür.
2. Her kural için ateşleme gücü ve normalize ağırlık hesaplanır.
3. Bu ağırlıklar ile bir tasarım matrisi oluşturulur.
4. V1 veya V2 model yapısına göre katsayı vektörü öğrenilir.
5. Güncel V1/V2 uygulamalarında katsayılar `np.linalg.solve()` ile regularized least squares çözümü olarak hesaplanır.
6. Öğrenilen katsayılar JSON ve TXT denklem dosyalarına kaydedilir.
7. Model performansı RMSE, MAE, MAPE ve R2 metrikleriyle değerlendirilir.

En kritik fark:

```text
Linear Regression tek global denklem öğrenir.
Sugeno ise fuzzy kuralların ağırlıklandırdığı birden fazla yerel denklem öğrenir.
```

