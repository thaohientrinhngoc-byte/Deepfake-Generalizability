# Generalizability Evaluation on Datasets using Spatiotemporal Architectures in Deepfake Detection

Official implementation accompanying the research paper: **"Generalizability Evaluation on Datasets using Spatiotemporal Architectures in Deepfake Detection"**  
*Faculty of Artificial Intelligence, FPT University, Danang Campus, Vietnam*

---

## Abstract

Deepfake detection plays an essential role in mitigating the negative social impacts of AI-manipulated videos. However, existing detectors often struggle to generalize when deployed in real-world scenarios due to dataset bias. In this work, an experimental framework is designed to systematically evaluate the generalizability of datasets using spatiotemporal architectures. 

Our framework employs an **EfficientNet-B0** backbone to extract spatial features, followed by either a **Global Average Pooling (GAP)** layer or a **3-layer Long Short-Term Memory (LSTM)** module to capture temporal dynamics. Intra-dataset and inter-dataset evaluations are conducted across four major benchmarks:
1. **FaceForensics++ (FF++)** (c23 compression)
2. **DeepFake Detection Challenge (DFDC-10)**
3. **Celeb-DF (v2)**
4. **WildDeepfake** (In-the-wild out-of-distribution benchmark)

Additionally, we introduce the **Dataset Generalizability Score (DGS)** to quantify dataset transferability across model architectures.

---

## Model Configurations (Table I Alignment)

All models share the same EfficientNet-B0 visual backbone pre-trained on ImageNet, evaluated under two backbone adaptation strategies (Fully Frozen vs. Fine-tuning Blocks 7 & 8) and two temporal aggregation heads (GAP vs. 3-Layer LSTM):

| Configuration Name | LaTeX Symbol | Backbone Adaptation Strategy | Temporal Modeling Head | Classification Layer |
| :--- | :--- | :--- | :--- | :--- |
| **`M_CNN_F`** | $M_{CNN-F}$ | Frozen (All parameters fixed) | Global Average Pooling | Linear Layer |
| **`M_CNN_T`** | $M_{CNN-T}$ | Fine-tuned (Blocks 7 & 8 learnable) | Global Average Pooling | Linear Layer |
| **`M_CL_F`** | $M_{CL-F}$ | Frozen (All parameters fixed) | 3-Layer LSTM | Linear Layer |
| **`M_CL_T`** | $M_{CL-T}$ | Fine-tuned (Blocks 7 & 8 learnable) | 3-Layer LSTM | Linear Layer |

---

## Dataset Generalizability Score (DGS)

The **Dataset Generalizability Score (DGS)** evaluates how effectively knowledge learned from a given training dataset transfers to unseen external benchmark datasets:

$$\text{DGS} = \bar{A}_{\text{cross}} \times \left( \frac{\bar{A}_{\text{cross}}}{A_{\text{internal}}} \right) \times (1 - \sigma_{\text{cross}})$$

Where:
- $\bar{A}_{\text{cross}}$: Mean classification performance (e.g. AUC) across all out-of-distribution cross-test datasets.
- $A_{\text{internal}}$: In-distribution test performance on the training dataset's testing split.
- $\sigma_{\text{cross}}$: Standard deviation of cross-dataset performances (penalizes high variance across evaluation domains).

---

## Experimental Benchmark Results ($M_{CL-T}$)

Cross-dataset generalizability evaluation of the best performing architecture ($M_{CL-T}$, Tuned EfficientNet-B0 + 3-Layer LSTM) across 3 independent random seeds:

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

### 2. Dataset Preprocessing

To extract 30 uniform frames per video with SCRFD face detection and IoU tracking:

```bash
python preprocess.py \
  --split_file /path/to/dataset_split.csv \
  --output_dir ./processed_dataset \
  --num_frames 30 \
  --target_size 224 \
  --crop_margin 0.4
```

### 3. Model Training

Train any of the 4 paper configurations (`M_CNN_F`, `M_CNN_T`, `M_CL_F`, `M_CL_T`):

```bash
# Train Tuned CNN-LSTM model (M_CL_T)
python train.py \
  --model M_CL_T \
  --data_dir ./processed_dataset \
  --output_dir ./models_checkpoints \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0001
```

### 4. Cross-Dataset Matrix Evaluation

To evaluate trained checkpoints across all 4 benchmark datasets and automatically compute DGS:

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
  title={Generalizability Evaluation on Datasets using Spatiotemporal Architectures in Deepfake Detection},
  author={Ngoc Thao Hien Trinh and Nguyen Dinh Hoang and Nguyen Huu Gia Bao and Quoc-Trinh Vo},
  booktitle={2026 International Conference on Cognitive and Intelligent Systems (ICogSys 2026)},
  year={2026}
}
```
