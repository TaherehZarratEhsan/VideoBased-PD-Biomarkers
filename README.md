# Finger Tapping Analysis for Parkinson Disease
This repository provides the **official implementation** of the methods described in our paper:



---

## 📂 Repository Structure

```
Parkinson-Digital-Biomarkers/
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
git clone https://github.com/TaherehZarratEhsan/Parkinson-Digital-Biomarkers.git
cd Parkinson-Digital-Biomarkers
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
They are hosted externally and must be downloaded before running the feature extraction.

➡️ [Download Raw Data](https://your-link-to-download.com)

After downloading, place the file(s) into:
```
data/raw/
    └── video_keypoints.pkl
```

### 📦 Contents of `video_keypoints.pkl`

pickle file is a Python dictionary containing the following keys:

- **`video_path`**: List of video file paths corresponding to each sample.  
- **`distances`**: List of distance signals (thumb–index distance or angle) for each video.  
- **`keypoints`**: List of Mediapipe hand keypoints per frame (shape: frames × 21 landmarks × 3 coordinates).  
- **`id`**: Patient ID for each video.  
- **`label`**: Clinical MDS-UPDRS score (0–4).  
- **`fps`**: Frames per second of the corresponding video.

Example code to inspect the pickle file:

```python
import pickle

with open("data/raw/video_keypoints.pkl", "rb") as f:
    annotated_data = pickle.load(f)

print("Keys:", annotated_data.keys())
print("Number of samples:", len(annotated_data['video_path']))

# Example: show first entry
print("Video path:", annotated_data['video_path'][0])
print("Patient ID:", annotated_data['id'][0])
print("Label:", annotated_data['label'][0])
print("FPS:", annotated_data['fps'][0])
print("Distance signal length:", len(annotated_data['distances'][0]))
```

This file serves as the **input** for `feature_extaction.py`, which extracts motor features and saves them in `data/processed/combined_features.csv`.

---

## ▶️ Usage

### Feature Extraction
After downloading and placing `video_keypoints.pkl` in `data/raw/`, run:

```bash
python src/feature_extraction/feature_extaction.py

This will generate:
data/processed/combined_features.csv

### Preprocess Videos & Extract keypoints
🔧 Generating Your Own Pickle File

If you want to generate the pickle from raw videos, use:

```bash
python src/preprocessing/keypoint_extraction.py
```

This script:
- Processes each video using Mediapipe’s `HandLandmarker`.  
- Extracts and normalizes hand keypoints (distance- or angle-based).  
- Optionally trims irrelevant parts of the signal.  
- Saves the results to:

```
data/raw/video_keypoints.pkl
```

The configuration inside `keypoint_extraction.py` specifies:
- CSV files mapping videos to patient IDs and clinical scores (`vid2score`, `id2vid`, `ids`).  
- Whether to use distance- or angle-based signals (`distance: True/False`).  
- Whether to trim irrelevant actions (`trimmed: True/False`).  
- The save path for the generated pickle.

Example config snippet:

```python
CONFIG = {
    'vid2score': 'path/to/segmented_ft_vid2score.csv',
    'id2vid': 'path/to/id2vid.csv',
    'ids': 'path/to/patient_id_all.csv',
    'save_path': 'data/raw/',
    'distance': False,   # False = angle-based, True = distance-based
    'trimmed': False,    # Whether to trim irrelevant actions
}
```
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
