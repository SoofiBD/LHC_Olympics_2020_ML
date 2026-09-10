# ParT Autoencoder Training Plan (for RTX 4070 Ti)

Bu döküman, LHC Olympics 2020 Anomaly Detection projesinin Particle Transformer (ParT) Autoencoder modelini, 12 GB VRAM'e sahip RTX 4070 Ti bilgisayarınızda eğitmek için hazırlanmıştır.

Mac üzerinde kodlarda şu düzeltmeler yapıldı ve GitHub'a eklendi:
1. `scripts/download_data.py` içerisindeki eskimiş Zenodo indirme linkleri güncellendi.
2. `src/data/dataset.py` içerisindeki zero-padding'in (LHC verisinde boşlukların 0 ile doldurulmasının) yanlışlıkla etiket (label) sütunu sanılıp kesilmesine neden olan hata giderildi.

Aşağıdaki adımları masaüstü bilgisayarınızda uygulayarak eğitime başlayabilirsiniz.

## Adım 1: Gereksinimlerin Kurulması

Proje klasörüne terminalden girin ve sanal ortam oluşturup kütüphaneleri yükleyin. Python paket yöneticisi olarak `uv` veya standart `pip` kullanabilirsiniz:

```bash
# Eğer uv kullanıyorsanız (Hızlı Kurulum):
uv venv
uv pip install -r requirements.txt

# Eğer standart pip kullanıyorsanız:
python -m venv .venv
# Windows için sanal ortamı aktif etme:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Adım 2: Veri Setinin İndirilmesi

Mac üzerinde düzeltmiş olduğumuz indirme script'i sayesinde veriyi artık tek komutla hatasız indirebilirsiniz:

```bash
python scripts/download_data.py --dataset background
```

*Not: İndirme işlemi yaklaşık 2.5 GB'tır. Tamamlandığında dosya `data/raw/events_LHCO2020_backgroundMC_Pythia.h5` yolunda olacaktır.*

## Adım 3: Eğitim Komutları (Training Runs)

RTX 4070 Ti, 12 GB VRAM ve 4. Nesil Tensor Çekirdeklerine sahiptir. Projemizde **AMP (Automatic Mixed Precision - FP16)** aktif olduğu için hem Batch Size 128 hem de Batch Size 64 desteklenir.

### Seçenek A: En Hızlı Seçenek — Batch Size 128 (~3.5 – 4 Saat) *(Önerilen)*
Batch size 128, GPU çekirdeklerini en verimli şekilde kullanarak adım sayısını yarıya indirir ve toplam süreyi %25-30 kısaltır:

```bash
python scripts/train.py \
  --config configs/part_autoencoder.yaml \
  --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5 \
  --batch-size 128 \
  --epochs 20 \
  --run-name partAE_ep20_bs128_cuda \
  --device cuda
```

---

### Seçenek B: Güvenli Seçenek — Batch Size 64 (~4.5 – 5.5 Saat)
Eğer sisteminizde arka planda çok fazla VRAM kullanan uygulama varsa veya 128'de `CUDA out of memory` hatası alırsanız:

```bash
python scripts/train.py \
  --config configs/part_autoencoder.yaml \
  --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5 \
  --batch-size 64 \
  --epochs 20 \
  --run-name partAE_ep20_bs64_cuda \
  --device cuda
```

---

### Seçenek C: Hızlı Ön Test — R&D Veri Seti (~30 – 40 Dakika)
Tüm boru hattının (HDF5 okuma, GPU eğitimi, grafik ve ağırlık kaydetme) sorunsuz çalıştığını kısa sürede teyit etmek için 110 bin olaylık R&D veri setiyle hızlı eğitim:

```bash
# R&D verisini indirme (yaklaşık 250 MB):
python scripts/download_data.py --dataset rnd

# R&D üzerinde eğitim:
python scripts/train.py \
  --config configs/part_autoencoder.yaml \
  --data data/raw/events_LHCO2020_RnD.h5 \
  --batch-size 64 \
  --epochs 20 \
  --run-name partAE_rnd_quick \
  --device cuda
```

---

### Parametre Notları:
- **`--batch-size 128 / 64`**: GPU VRAM'ine göre ayarlanır. Önce 128 ile başlayıp sorunsuzsa devam etmeniz önerilir.
- **`--device cuda`**: Eğitimin RTX 4070 Ti ekran kartınızda çalışmasını sağlar.
- **`--epochs 20`**: Verinin modelden kaç tur geçeceğidir. Daha hızlı sonuç görmek için 5 veya 10 yapılabilir.

## Adım 4: Değerlendirme (Evaluation) Komutu

Eğitilen `best_model` ağırlıklarını test verisi veya yarışma verisi (BlackBox1) üzerinde anomali tespiti için değerlendirmek:

```bash
# BlackBox1 verisini indirme:
python scripts/download_data.py --dataset blackbox1

# Modeli BlackBox1 üzerinde değerlendirme ve anomali skorlarını çıkarma:
python scripts/evaluate.py \
  --checkpoint outputs/models/best_model_partAE_ep20_bs128_cuda.pt \
  --config configs/part_autoencoder.yaml \
  --data data/raw/events_LHCO2020_BlackBox1.h5 \
  --model-type part_autoencoder \
  --tag partAE_eval_bb1 \
  --device cuda
```

## Adım 5: Sonuçların Kontrolü

Eğitim bittiğinde şu dosyalar otomatik oluşturulacaktır:
- `outputs/models/best_model_*.pt`: En düşük validation loss'a sahip en iyi ağırlıklar.
- `outputs/figures/loss_curves_*.png`: Eğitimin Train/Val Loss eğrileri grafiği.
- `outputs/logs/`: Eğitimle alakalı JSON ve CSV metadataları.

Şimdiden kolay gelsin!

