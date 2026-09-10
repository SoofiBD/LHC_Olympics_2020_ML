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

## Adım 3: Eğitim Komutu (Full Training Run)

RTX 4070 Ti, 12 GB VRAM'e sahip güçlü bir ekran kartıdır. Mac'te 24 GB Unified Memory olduğu için 128 batch size kullanabiliyorduk, ancak burada "CUDA Out of Memory" hatası almamak adına `batch-size` değerini 32 veya 64 olarak tutmamız güvenli olacaktır.

Modelin tamamı GPU üzerinde (CUDA ile) çalışacak şekilde şu komutu çalıştırın:

```bash
python scripts/train.py \
  --config configs/part_autoencoder.yaml \
  --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5 \
  --batch-size 64 \
  --epochs 20 \
  --run-name partAE_ep20_bs64_cuda \
  --device cuda
```

### Parametre Notları:
- **`--batch-size 64`**: Eğer OOM (Out of Memory) hatası alırsanız bu değeri `32` veya `16`'ya düşürebilirsiniz. Sorunsuz başlarsa böyle bırakabilirsiniz.
- **`--device cuda`**: Eğitimin ekran kartınızda (RTX 4070 Ti) çalışmasını sağlar.
- **`--epochs 20`**: Verinin tamamen kaç kez üzerinden geçileceğini belirler. Daha hızlı sonuç görmek isterseniz `5` veya `10`'a çekebilirsiniz.

## Adım 4: Sonuçların Kontrolü

Eğitim bittiğinde şu dosyalar otomatik oluşturulacaktır:
- `outputs/models/best_model_partAE_ep20_bs64_cuda.pt`: En iyi doğruluk oranına sahip ağırlıklar.
- `outputs/figures/loss_curves_partAE_ep20_bs64_cuda.png`: Eğitimin Loss grafiği.
- `outputs/logs/`: Eğitimle alakalı JSON ve YAML metadataları.

Şimdiden kolay gelsin!
