# Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

*Advanced CNNs and Transfer Learning*.
Disusun oleh **Kenneth Howen (2802504346)**.

Repositori ini membandingkan tiga paradigma pelatihan ResNet-18 pada *small-data regime*, yaitu pelatihan dari nol, *feature extraction*, dan *progressive fine-tuning*. Repositori ini juga memuat uji kontrol *data augmentation*. Semua eksperimen dijalankan dengan satu protokol yang sama dan seed tetap.

## Struktur Repositori

```
.
├── TransferLearning.ipynb            # notebook lengkap beserta output sel
├── requirements.txt                   # versi pustaka yang dipakai saat eksekusi
├── laporan/
│   └── TransferLearning_Laporan.pdf   # laporan analisis (3–5 halaman)
├── outputs/
│   ├── summary.csv                    # ringkasan metrik per kondisi (B1–B4)
│   ├── history_per_epoch.csv          # log train/val loss & accuracy per epoch
│   ├── metrics.json                   # summary + history + seed + device
│   ├── split_indices.json             # daftar file train/val/test (bukti split)
│   ├── environment.json               # versi Python, pustaka, GPU
│   ├── failure_cases.csv              # 8 misklasifikasi model terbaik
│   ├── loss_curves_B1_B2_B3.png       # kurva training vs validation loss
│   ├── B4_augmentation_control.png    # akurasi & generalization gap, dengan vs tanpa augmentasi
│   ├── confusion_matrix_best.png      # confusion matrix model terbaik (test set)
│   └── failure_cases.png              # grid citra misklasifikasi
└── README.md
```

## Dataset

- **Sumber:** Oxford-IIIT Pet ([Parkhi et al., 2012](https://www.robots.ox.ac.uk/~vgg/data/pets/)). Dataset diunduh otomatis melalui `torchvision.datasets.OxfordIIITPet`.
- **Subset 10 kelas:**
  - Kucing: Abyssinian, Bengal, Birman, Bombay, Persian
  - Anjing: Beagle, Boxer, Pug, Samoyed, Shiba Inu
- **Split:** Split *trainval* dan *test* resmi digabung, lalu dibagi ulang secara *stratified* 70/15/15 (`random_state=42`). Hasilnya **1.386 train / 297 validation / 298 test**. Notebook memeriksa bahwa ketiga split tidak saling beririsan. Daftar file per split tersimpan di `outputs/split_indices.json`.

## Protokol Eksperimen

| Komponen | Nilai |
|---|---|
| Arsitektur | ResNet-18 (`torchvision.models.resnet18`) |
| Bobot pretrained | `ResNet18_Weights.IMAGENET1K_V1` |
| Input | 224 × 224, normalisasi ImageNet (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) |
| Batch size / optimizer / loss | 32 / AdamW (weight decay 1e-2) / CrossEntropyLoss |
| Early stopping | Memantau validation loss, patience 4. Bobot epoch terbaik dimuat ulang sebelum evaluasi test. |
| Seed | 42 untuk Python `random`, NumPy, dan PyTorch, dengan `cudnn.deterministic=True` |
| Mixed precision | `torch.autocast` float16 di GPU |
| Augmentasi (hanya train) | `RandomResizedCrop(224, scale=(0.8,1.0))`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2)` |
| Transformasi val/test | `ResNet18_Weights.IMAGENET1K_V1.transforms()` (deterministik) |

### Matriks Kondisi

| Kode | Kondisi | Konfigurasi |
|---|---|---|
| B1 | From scratch | `weights=None`, seluruh parameter dilatih, LR 1e-3 |
| B2 | Feature extraction | Backbone dibekukan, `fc` diganti `nn.Linear(512, 10)`, LR 1e-3 |
| B3 | Progressive fine-tuning | `layer4` + `fc` dibuka, *differential LR*: 1e-4 (`layer4`) dan 1e-3 (`fc`) |
| B4 | Uji kontrol augmentasi | Konfigurasi B3 dijalankan dengan augmentasi (= run B3) dan tanpa augmentasi (Resize/CenterCrop + normalisasi) |
| B5 | Arsitektur pembanding (bonus) | Tersedia di kode (MobileNetV3-Small, `RUN_B5 = True`), **tidak dijalankan** pada eksekusi ini |

Catatan implementasi: pada layer yang dibekukan, modul BatchNorm dipaksa ke mode `eval()` selama training. Langkah ini perlu karena `requires_grad=False` saja tidak menghentikan pembaruan *running mean/var* ketika `model.train()` dipanggil.

## Hasil

Angka di bawah disalin dari `outputs/summary.csv`. Eksekusi dilakukan pada Google Colab dengan GPU Tesla T4.

| Kondisi | Trainable params | Best / total epoch | Waktu latih (s) | Val acc | Test acc | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| B1 — from scratch | 11.181.642 | 3 / 7 | 52,8 | 0,3502 | 0,3255 | 0,3204 |
| B2 — feature extraction | 5.130 | 13 / 17 | 119,5 | 0,9764 | 0,9698 | 0,9703 |
| B3 — fine-tune `layer4` (+aug) | 8.398.858 | 2 / 6 | 43,5 | 0,9697 | 0,9664 | 0,9667 |
| B4 — B3 tanpa augmentasi | 8.398.858 | 4 / 8 | 23,1 | 0,9731 | 0,9765 | 0,9767 |

Model terbaik untuk *failure analysis* dipilih berdasarkan **validation accuracy** (B2), bukan test accuracy, agar test set tidak ikut dipakai dalam pemilihan model. Analisis dan pembahasan lengkap ada di laporan.

## Cara Menjalankan

### Google Colab (disarankan)
1. Buka `TransferLearning.ipynb` di Google Colab.
2. Pilih **Runtime → Change runtime type → T4 GPU**.
3. Jalankan **Runtime → Run all**. Seluruh proses memakan waktu sekitar 15–20 menit, termasuk mengunduh dataset (sekitar 800 MB).
4. Setelah selesai, notebook otomatis mengunduh `transferLearning_outputs.zip` yang berisi seluruh isi folder `outputs/`.

### Lokal
```bash
git clone https://github.com/Kyurein/Advanced-CNNs-and-TransferLearning.git
cd Advanced-CNNs-and-TransferLearning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter nbconvert --to notebook --execute TransferLearning.ipynb --output executed.ipynb
```
GPU CUDA sangat disarankan. Di CPU notebook tetap bisa berjalan, tetapi jauh lebih lambat dan tanpa mixed precision.

## Reproduksibilitas

- Lingkungan eksekusi tercatat di `outputs/environment.json`: Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, Tesla T4.
- Dua kali *Run all* menghasilkan best epoch, validation accuracy, test accuracy, macro-F1, dan daftar *failure case* yang identik untuk B1–B4. Yang berbeda hanya durasi pelatihan.
- Hasil di GPU atau versi pustaka lain bisa sedikit berbeda karena perbedaan kernel CUDA.

## Keterbatasan

- Eksperimen memakai satu seed, sehingga selisih kecil antar kondisi belum diuji signifikansinya.
- Subset hanya 10 kelas, dengan sekitar 30 citra per kelas di test set.
- Kondisi B5 dan eksplorasi C5 (Grad-CAM / feature map) tidak dijalankan.

## Referensi

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*.
- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and Dogs. *CVPR*.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
- Bahan tayang Sesi 05 — *Advanced CNNs and Transfer Learning*.

## Pernyataan Penggunaan AI

Repositori dan laporan ini disusun dengan bantuan alat AI. Claude (Anthropic) dipakai untuk menyusun kode pipeline eksperimen, merangkum isi slide, memeriksa konsistensi log, menyusun README ini, serta menyusun kerangka dan memformat laporan. ChatGPT (OpenAI) dipakai dalam penyusunan draf bagian analisis laporan. Rincian ruang lingkup dan cara verifikasinya ada di Lampiran — Pernyataan Penggunaan AI pada laporan.
