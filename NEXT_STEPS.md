# Gelecek Adımlar ve Yapılacaklar (Roadmap)

Bu dosya, LHC Olympics 2020 projesinde şu ana kadar tamamlanan işleri ve bundan sonra atılacak adımları içerir.

## 1. Mevcut Durum
- **Veri Seti:** Pythia arka plan verisi (`events_LHCO2020_backgroundMC_Pythia.h5`, 1M olay) ile çalışıldı.
- **Model:** Particle Transformer (ParT) Autoencoder kuruldu ve 20 epoch eğitildi.
- **Değerlendirme:** GPU (`--device cuda`) kullanılarak anomali skorları ve Bump-Hunt analizi hesaplandı.

---

## 2. Sıradaki Adımlar

### A. Model Eğitimini İyileştirme (Stabilite ve Hız)
- **Learning Rate:** `lr=0.001` eğitim sırasında bazı epoch'larda NaN kaybına sebep oldu. Bir sonraki eğitimde `lr: 0.0003` kullanılmalı.
- **Gradient Clipping:** Gradyan patlamalarını önlemek için `trainer.py` içerisine `clip_grad_norm_` eklenebilir.
- **Batch & Worker:** `batch_size: 256` ve `num_workers: 4` ile GPU daha verimli kullanılmalı.

### B. R&D Veri Seti İle Sinyal Tespiti (ROC / AUC)
- **Veriyi İndirme:**
  ```bash
  python scripts/download_data.py --dataset rnd
  ```
- **AUC Hesabı:** Etiketli R&D verisi (`events_LHCO2020_RnD.h5`) üzerinden modelin sinyal yakalama başarısı (ROC-AUC skoru) ölçülebilir:
  ```bash
  python scripts/evaluate.py --checkpoint outputs/models/<model_checkpoint>.pt --config configs/part_autoencoder.yaml --data data/raw/events_LHCO2020_RnD.h5 --device cuda
  ```

### C. Gözetimli Model (ParT Classifier) Eğitimi
- Sinyal ve arka plan ayrımını doğrudan öğrenen supervised sınıflandırıcı model eğitilebilir:
  ```bash
  python scripts/train.py --config configs/part_classifier.yaml --data data/raw/events_LHCO2020_RnD.h5 --device cuda
  ```

---

## 3. Pratik Komutlar

```bash
# 1. R&D Verisini İndir
python scripts/download_data.py --dataset rnd

# 2. Düzeltilmiş LR ile Autoencoder Eğit
python scripts/train.py --config configs/part_autoencoder.yaml --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5 --lr 0.0003 --batch-size 256

# 3. Model Değerlendir (GPU ile)
python scripts/evaluate.py --checkpoint outputs/models/<model_adi>.pt --config configs/part_autoencoder.yaml --data data/raw/events_LHCO2020_RnD.h5 --device cuda
```
