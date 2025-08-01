# Finger Tapping Analysis for Parkinson’s Disease
This repository provides the **official implementation** of the methods described in our paper:



---

## 📂 Repository Structure

```
finger-tapping-analysis/
│
├── data/
│   ├── raw/                 # Downloaded pickle data go here
│   └── processed/           # CSVs and processed feature files
│
├── src/
│   ├── feature_extraction/
│   │   └── feature_extaction.py
│   ├── preprocessing/
│   │   └── keypoint_extraction.py
│   ├── training/
│   │   └── optimization_training.py
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/finger-tapping-analysis.git
cd finger-tapping-analysis
```

### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate mediapip_torch
```

### Option 2: pip
```bash
pip install -r requirements.txt
```

---

## 📥 Data Access

The raw pickle data (~800 MB) are **too large for GitHub**.  
They are hosted externally:

➡️ [Download Raw Data](https://your-link-to-download.com)

After downloading, place them into:
```
data/raw/
    ├── video_keypoints.pkl
    └── control_keypoints.pkl
```

If you only want to test the pipeline, you may use a subset of the dataset.

---

## ▶️ Usage

### Preprocess Videos & Extract Features
```bash
python src/preprocessing/FT_myHC_savefeature_annotated.py
```
- Processes finger tapping videos with Mediapipe  
- Extracts keypoints and distance signals  
- Saves pickle & feature CSV files

### Feature Extraction
After downloading and placing `video_keypoints.pkl` in `data/raw/`, run:

```bash
python src/feature_extraction/feature_extaction.py

This will generate:
```bash
data/processed/combined_features.csv

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{finger_tapping_analysis,
  author       = {Your Name},
  title        = {Finger Tapping Analysis for Parkinson’s Disease},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/your-username/finger-tapping-analysis}}
}
```

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
