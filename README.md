# Finger Tapping Analysis for Parkinson’s Disease

This repository provides a pipeline for the **video-based quantification of motor characteristics in Parkinson’s disease** using the **finger tapping test**.  
It includes preprocessing with Mediapipe, feature extraction, statistical testing, and clustering to capture bradykinesia, sequence effects, and hesitation-halts.

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
│   │   └── FT_myHC_savefeature_annotated.py
│   └── utils/
│       └── plotting.py      # optional shared plotting functions
│
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_clustering_visualization.ipynb
│
├── results/
│   ├── figures/             # saved plots
│   └── stats/               # CSV results
│
├── tests/                   # Unit tests for reproducibility
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

### 1. Preprocess Videos & Extract Features
```bash
python src/preprocessing/FT_myHC_savefeature_annotated.py
```
- Processes finger tapping videos with Mediapipe  
- Extracts keypoints and distance signals  
- Saves pickle & feature CSV files

### 2. Run Feature Extraction & Analysis
```bash
python src/feature_extraction/feature_extaction.py
```
- Computes quantitative features (amplitude, speed, tapping interval, etc.)  
- Performs ANOVA, t-tests, or Mann–Whitney U tests based on normality  
- Generates boxplots, correlation heatmaps, clustering visualizations

### 3. Explore in Jupyter
```bash
jupyter notebook notebooks/01_feature_extraction.ipynb
```

---

## 📊 Results

- **Figures**  
  - Boxplots with ANOVA brackets  
  - Correlation heatmaps (Pearson & Spearman)  
  - PCA & clustering plots (KMeans, DBSCAN, HDBSCAN)

- **Statistical Outputs**  
  - ANOVA results  
  - Pairwise t-tests & Mann–Whitney U with Bonferroni correction  
  - Correlation coefficients with p-values  

All results are saved under:
```
results/
    ├── figures/
    └── stats/
```

---

## 🧪 Testing

Run unit tests with `pytest`:

```bash
pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome!  
Please open an issue or submit a pull request if you’d like to improve this repository.

Steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to your fork (`git push origin feature-branch`)
5. Open a Pull Request

---

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
