# Generalizability Evaluation on Deepfake Video Datasets using Spatiotemporal Architectures

Official implementation accompanying the research paper: **"Generalizability Evaluation on Deepfake Video Datasets using Spatiotemporal Architectures"**  
*Faculty of Artificial Intelligence, FPT University, Danang Campus, Vietnam*

---

## Abstract

Deepfake detection plays an important role in reducing the negative social impacts of manipulated videos. However, existing models often struggle to generalize when deployed in real-world scenarios due to dataset bias. In this work, an experimental framework is designed to evaluate the generalizability of a dataset using spatiotemporal architectures.

Our spatiotemporal model employs an **EfficientNet-B0** backbone to extract spatial features, followed by either a **Global Average Pooling (GAP)** layer or a **3-layer Long Short-Term Memory (LSTM)** module to capture temporal dynamics. Intra-dataset and inter-dataset evaluations are conducted across four datasets:
1. **FaceForensics++ (FF++)** (c23 compression)
2. **DeepFake Detection Challenge (DFDC-10)**
3. **Celeb-DF (v2)**
4. **WildDeepfake** (In-the-wild cross-test benchmark)

Additionally, the **Dataset Generalizability Score (DGS)** is introduced as a unified metric combining intra-dataset and inter-dataset classification performances to quantify dataset transferability.

---

## Model Configurations (Table I)

All four configurations share the same ImageNet pre-trained **EfficientNet-B0** backbone, evaluated under two backbone adaptation strategies (Fully Frozen vs. Fine-tuning top two blocks: Blocks 7 & 8) and two temporal modeling heads (Global Avg Pooling vs. 3-Layer LSTM):

| Configuration | LaTeX Symbol | Backbone Adaptation | Temporal Modeling | Classification Layer |
| :--- | :--- | :--- | :--- | :--- |
| **`M_CNN_F`** | $M_{CNN-F}$ | Frozen | Global Avg Pooling | Linear |
| **`M_CNN_T`** | $M_{CNN-T}$ | Fine-tuned (Blocks 7, 8) | Global Avg Pooling | Linear |
| **`M_CL_F`** | $M_{CL-F}$ | Frozen | 3-Layer LSTM | Linear |
| **`M_CL_T`** | $M_{CL-T}$ | Fine-tuned (Blocks 7, 8) | 3-Layer LSTM | Linear |

---

## Dataset Splits (Table II)

Number of videos (with $N = 30$ uniform frames extracted per video at $224 \times 224$ resolution) for training, validation, and testing splits (partitioned with seed `42`):

| Dataset | Training Set | Validation Set | Testing Set |
| :--- | :---: | :---: | :---: |
| **FaceForensics++ (c23)** | 4,800 | 600 | 600 |
| **DFDC-10** | 6,027 | 753 | 756 |
| **Celeb-DF (v2)** | 4,911 | 1,061 | 557 |
| **WildDeepfake** | — | — | 1,230 |

---

## Dataset Generalizability Score (DGS — Equation 1)

To evaluate how effectively knowledge learned from a training dataset transfers to unseen external benchmarks, **DGS** is formulated in Section 3.3 (Equation 1) as:

$$\text{DGS} = \bar{A}_{\text{cross}} \times \left( \frac{\bar{A}_{\text{cross}}}{A_{\text{internal}}} \right) \times (1 - \sigma_{\text{cross}})$$

Where:
- $\bar{A}_{\text{cross}}$: Mean inter-dataset AUC over the remaining three unseen cross-test datasets.
- $\sigma_{\text{cross}}$: Sample standard deviation (`ddof=1`) of inter-dataset AUC over the three unseen cross-test datasets.
- $A_{\text{internal}}$: Intra-dataset AUC on the testing split of the same dataset used for training.

---

## Experimental Results (Tables III–VI, 3 Seeds: 42, 36, 75)

Each model configuration was trained and evaluated across three independent runs using random seeds **42, 36, and 75** (controlling weight initialization, balanced 1:1 batch sampling, and data augmentation). Results are reported in percentage (`%`) as $\text{mean} \pm \text{standard deviation}$:

### Summary of $\text{DGS}_\text{AUC}$ (%) Across All Models

| Training Dataset | $M_{CNN-F}$ (Table III) | $M_{CNN-T}$ (Table IV) | $M_{CL-F}$ (Table V) | $M_{CL-T}$ (Table VI) |
| :--- | :---: | :---: | :---: | :---: |
| **DFDC** | 40.74 | 43.08 | 33.43 | **46.24** |
| **CELEB_DF** | 40.50 | 37.39 | 41.57 | 41.12 |
| **FF++** | **47.75** | **48.78** | **47.01** | 44.35 |

### Detailed Results for $M_{CL-T}$ (Table VI, Unit: %)

| Training Set | Testing Set | AUC (%) | ACC (%) | F1-Score (%) | Recall Real (%) | Recall Fake (%) | EER (%) | $\text{DGS}_\text{AUC}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DFDC** | DFDC (Intra) | $99.48 \pm 0.11$ | $95.94 \pm 0.61$ | $96.92 \pm 0.46$ | $96.43 \pm 1.43$ | $95.70 \pm 0.30$ | $3.97 \pm 0.60$ | **46.24** |
| | CELEB_DF | $78.77 \pm 1.31$ | $81.93 \pm 2.97$ | $89.12 \pm 1.97$ | $49.24 \pm 1.74$ | $88.06 \pm 3.41$ | $29.53 \pm 2.99$ | |
| | FF++ | $61.97 \pm 0.26$ | $69.83 \pm 1.50$ | $80.79 \pm 1.32$ | $38.00 \pm 4.58$ | $76.20 \pm 2.71$ | $40.00 \pm 1.00$ | |
| | Wild | $71.91 \pm 1.26$ | $59.40 \pm 0.54$ | $66.22 \pm 0.81$ | $35.86 \pm 2.36$ | $87.76 \pm 2.79$ | $32.82 \pm 1.30$ | |
| **CELEB_DF** | CELEB_DF (Intra) | $96.69 \pm 0.42$ | $91.68 \pm 0.10$ | $94.89 \pm 0.06$ | $91.67 \pm 1.31$ | $91.68 \pm 0.22$ | $8.41 \pm 0.41$ | **41.12** |
| | DFDC | $62.53 \pm 0.60$ | $39.73 \pm 1.63$ | $19.36 \pm 4.49$ | $97.36 \pm 0.83$ | $10.91 \pm 2.86$ | $41.80 \pm 1.50$ | |
| | FF++ | $65.13 \pm 0.49$ | $22.56 \pm 1.51$ | $13.27 \pm 3.14$ | $99.67 \pm 0.58$ | $7.13 \pm 1.80$ | $38.75 \pm 0.31$ | |
| | Wild | $62.86 \pm 1.08$ | $59.21 \pm 1.02$ | $35.87 \pm 7.41$ | $87.10 \pm 4.57$ | $25.62 \pm 7.25$ | $42.01 \pm 1.03$ | |
| **FF++** | FF++ (Intra) | $91.38 \pm 1.18$ | $83.55 \pm 2.18$ | $89.44 \pm 1.56$ | $83.00 \pm 1.73$ | $83.67 \pm 2.86$ | $16.47 \pm 0.92$ | **44.35** |
| | DFDC | $59.63 \pm 2.09$ | $51.41 \pm 5.00$ | $54.25 \pm 7.74$ | $66.66 \pm 2.48$ | $43.79 \pm 8.74$ | $43.78 \pm 2.98$ | |
| | CELEB_DF | $74.88 \pm 3.44$ | $59.30 \pm 2.85$ | $69.30 \pm 2.78$ | $84.09 \pm 3.01$ | $54.65 \pm 3.26$ | $32.58 \pm 4.30$ | |
| | Wild | $64.39 \pm 0.76$ | $56.56 \pm 0.37$ | $60.46 \pm 0.72$ | $42.71 \pm 2.54$ | $73.24 \pm 2.51$ | $38.94 \pm 1.18$ | |

---

## Preprocessed Datasets & Downloads

See [`dataset_links.md`](dataset_links.md) for direct download links:

* **Preprocessed Datasets (Uniform 30 frames, SCRFD face crops, 224x224):**
  * FaceForensics++ (c23): [Kaggle Dataset](https://www.kaggle.com/datasets/nghgb0101/ff-30frames)
  * DFDC (Subset 10): [Kaggle Dataset](https://www.kaggle.com/datasets/thohintrnhngc/dfdc-preprocessing)
  * Celeb-DF (v2): [Kaggle Dataset](https://www.kaggle.com/datasets/ndhoang2310/new-evaluate-celeb-df)
  * WildDeepfake: [Kaggle Dataset](https://www.kaggle.com/datasets/thohintrnhngc/wild-deepfake-30frame)

---

## Quick Start

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/thaohientrinhngoc-byte/Deepfake-Generalizability.git
cd Deepfake-Generalizability
pip install -r requirements.txt
```

### 2. Dataset Preprocessing (4-Phase Pipeline)

To extract 30 uniform frames per video with SCRFD face detection, IoU tracking, and refinement:

```bash
python preprocess.py \
  --split_file /path/to/dataset_split.csv \
  --output_dir ./processed_dataset \
  --num_frames 30 \
  --target_size 224 \
  --crop_margin 0.4
```

### 3. Model Training (Single Seed or 3-Seed Protocol)

Train any of the 4 paper configurations (`M_CNN_F`, `M_CNN_T`, `M_CL_F`, `M_CL_T`) across the three paper seeds (`42, 36, 75`):

```bash
# Train Tuned CNN-LSTM model (M_CL_T) across all 3 random seeds (42, 36, 75)
python train.py \
  --model M_CL_T \
  --data_dir ./processed_dataset \
  --output_dir ./models_checkpoints \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0001 \
  --seeds 42 36 75
```

### 4. Cross-Dataset Matrix Evaluation & DGS Computation

To evaluate trained checkpoints across all 4 benchmark datasets and automatically compute `Mean ± Std (%)` and `DGS_AUC`:

```bash
python evaluate.py \
  --checkpoints_dir ./models_checkpoints \
  --datasets_config datasets.json \
  --train_dataset DFDC \
  --output_dir ./results
```

---

## Citation

If you find this repository useful for your research, please cite:

```bibtex
@inproceedings{trinh2026generalizability,
  title={Generalizability Evaluation on Deepfake Video Datasets using Spatiotemporal Architectures},
  author={Ngoc Thao Hien Trinh and Nguyen Dinh Hoang and Nguyen Huu Gia Bao and Quoc-Trinh Vo},
  booktitle={2026 International Conference on Cognitive and Intelligent Systems (ICogSys 2026)},
  year={2026}
}
```
