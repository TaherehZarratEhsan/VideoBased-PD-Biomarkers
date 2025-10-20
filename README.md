### Quantifying Motor Characteristics in Parkinson Disease Using Computer Vision Techniques

This repository contains the official implementation of our paper:  
**[Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger Tapping Test in Parkinson Disease](https://arxiv.org/abs/2506.18925).**

Tahereh Zarrat Ehsan, Michael Tangermann, Yağmur Güçlütürk, Bastiaan R. Bloem, Luc J. W. Evers  
Radboud University, Donders Institute for Brain, Cognition and Behaviour

---


<p align="center">
  <img src="assets/ft.gif" width="60%" />
</p>

## 📂 Repository Structure

```
📁 VideoBased-PD-Biomarkers/
│
├── 📁 data/
│ ├── 📁 raw/ # Raw data and extracted keypoints (.pkl)
│ └── 📁 processed/ # Processed CSVs and derived feature files
│
├── 📁 src/
│ ├── 📁 preprocessing/
│ │ └── 📄 keypoint_extraction.py # Extracts hand keypoints using Mediapipe
│ ├── 📁 feature_extraction/
│ │ └── 📄 feature_extraction.py # Computes motor features (amplitude, speed, etc.)
│ ├── 📁 training/
│ │ └── 📄 optimization_training.py # Model training and evaluation scripts
│ └── 📁 demo/
│ └── 📄 ft_video_analysis.py # demo: video → features → plots
│
├── 📄 requirements.txt # Python dependencies
├── 📄 environment.yml # Conda environment setup
└── 📘 README.md # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/TaherehZarratEhsan/VideoBased-PD-Biomarkers.git
cd VideoBased-PD-Biomarkers
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
## ▶️ Usage
### 🔹 Part 1: Official Implementation
#### 🔹 Keypoint Extraction

If you want to build your own pickle file (`video_keypoints.pkl`) from raw videos, first prepare a CSV file with the following columns:

- **video_path**: Full path to each video  
- **score**: Clinical MDS‑UPDRS score 
- **id**: Patient ID  

Save it in `data/raw/segmented_ft_vid2score.csv`.

The Mediapipe hand landmark model is required to extract keypoints.  
It will be automatically downloaded on first run from:

[Google Cloud Storage – Mediapipe Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

Alternatively, you can download it manually. After download, the file should be located at:

```
src/preprocessing/hand_landmarker.task
```

Then run:

```bash
python src/preprocessing/keypoint_extraction.py
```

This will:
- Process all listed videos using Mediapipe’s HandLandmarker  
- Extract distance‑based signals 
- Save a dictionary with `video_path`, `distance signal`, `keypoints`, `id`, `label`, and `fps`  

The output will be stored in:

```
data/raw/video_keypoints.pkl
```
#### 🔹 Feature Extraction

After downloading keypoints (or generating) and placing `video_keypoints.pkl` in `data/raw/`, run:

```bash
python src/feature_extraction/feature_extaction.py
```

This will extract motor features (amplitude, speed, cycle duration, etc.) and generate:

```
data/processed/combined_features.csv
```
#### 🔹 Model Training and Evaluation

Once you have extracted the features (`data/processed/combined_features.csv`), you can train and evaluate classification models.

Run:

```bash
python src/training/optimization_training.py
```

- Saves:
  - ** results** to:
    ```
    data/processed/dynamic_save.csv
    ```
  - ** plots of metrics with confidence intervals** to:
    ```
    data/processed/<model>_performance_metrics.png
    ```

---

### 🔹 Part 2: Easy Demo (Quick Video-to-Results)

This demo offers a simple end-to-end example that runs the full analysis pipeline — from a raw video to automatic feature extraction and visualization.
It can be executed locally without any dataset setup or preprocessing steps from the main implementation.

📁 Script location
    ```
src/demo/ft_video_analysis.py
    ```

▶️ Run the demo
```bash
python ft_video_analysis.py --video_path "C:/Users/Tahereh/video.MP4" --hand2track Right
```
Arguments:
    ```
--video_path: Path to the input video file
--hand2track: Which hand to analyze (Left or Right)
    ```
    
## 📥 Data Access

Data from the [Personalized Parkinson Project](https://www.personalizedparkinsonproject.com/home) used in the present study were retrieved from the [PEP database](https://pep.cs.ru.nl/index.html).  
The PPP data are available upon request via [ppp-data@radboudumc.nl](mailto:ppp-data@radboudumc.nl).  
More details on the procedure can be found on the [project website](https://www.personalizedparkinsonproject.com/home).

---

## 📚 Citation

If you use this repository in your research, please cite:

> Zarrat Ehsan T, Tangermann M, Güçlütürk Y, Bloem B R, Evers L J.  
> *Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger Tapping Test in Parkinson Disease.*  
> arXiv e-prints, 2025 Jun: arXiv-2506.  
> [📘 View on arXiv](https://arxiv.org/abs/2506.18925)

---

## 📘 BibTeX

@article{zarratehsan2025finger,
  title={Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger Tapping Test in Parkinson Disease},
  author={Zarrat Ehsan, Tahereh and Tangermann, Michael and Güçlütürk, Yağmur and Bloem, Bastiaan R. and Evers, Luc J.W.},
  journal={arXiv e-prints},
  year={2025},
  month={Jun},
  eprint={2506.18925},
  archivePrefix={arXiv}
}

---

## 📜 License

This project is licensed under the Apache 2.0 License.
