# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:42:20 2025

@author: Z984222
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, balanced_accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lightgbm as lgb
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
import ast
from sklearn.utils import resample
import csv
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
from math import pi
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pickle
import json
from factor_analyzer import Rotator
from factor_analyzer import FactorAnalyzer
import matplotlib.colors as mcolors
from scipy.stats import f_oneway
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, shapiro, ttest_ind, mannwhitneyu
sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (255, 0, 0)  # vibrant red
from scipy.signal import welch, correlate
from scipy.stats import entropy
import pywt  
from scipy.signal import butter, filtfilt


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.results = {}
 
    def angle_signal(self, landmarks, width, height):
            """
            Calculate the angle between the vector connecting wrist to thumb tip
            and the vector connecting wrist to index finger tip.
            """
            # Get wrist, thumb tip, and index finger tip landmarks
            wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
            thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        
            # Convert normalized coordinates to pixel coordinates
            wrist_x, wrist_y = wrist.x * width, wrist.y * height
            thumb_x, thumb_y = thumb_tip.x * width, thumb_tip.y * height
            index_x, index_y = index_tip.x * width, index_tip.y * height
        
            # Create vectors
            vector_wrist_thumb = np.array([thumb_x - wrist_x, thumb_y - wrist_y])
            vector_wrist_index = np.array([index_x - wrist_x, index_y - wrist_y])
        
            # Calculate dot product and magnitudes of the vectors
            dot_product = np.dot(vector_wrist_thumb, vector_wrist_index)
            magnitude_wrist_thumb = np.linalg.norm(vector_wrist_thumb)
            magnitude_wrist_index = np.linalg.norm(vector_wrist_index)
        
            # Avoid division by zero
            if magnitude_wrist_thumb == 0 or magnitude_wrist_index == 0:
                return 0.0
        
            # Calculate cosine of the angle
            cos_angle = dot_product / (magnitude_wrist_thumb * magnitude_wrist_index)
            # Clamp value to avoid domain error in acos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
            # Calculate angle in degrees
            angle = np.degrees(np.arccos(cos_angle))
        
            return angle
    
    
    def _init_hand_landmarker(self):
        model_path = r'//chansey.umcn.nl/diag/Tahereh/new/src/Keypoint/hand_landmarker.task'
        with open(model_path, 'rb') as file:
            model_data = file.read()
        return model_data
      
    
    def _extract_features(self, distances):

        
        distances_ = np.array(distances)
       
        # === Define sampling rate and filter order ===
        fs = 30.0                   # Sampling frequency in Hz
        order = 4                   # Filter order
        nyq = 0.5 * fs              # Nyquist frequency
        
        # === Define cutoff frequencies for Butterworth filters ===
        # We'll include the original signal (no filtering) and 3 filtered versions
        cutoff_frequencies = [None, 5.0, 7.0, 9.0]
        colors = ['black', 'purple', 'green', 'red']
        titles = [
            'Original Signal',
            'Butterworth Filter (Cutoff = 5 Hz)',
            'Butterworth Filter (Cutoff = 7 Hz)',
            'Butterworth Filter (Cutoff = 9 Hz)'
        ]
        
        # === Create a list to store all signals: original + filtered versions ===
        signals = []
        
        # --- Step 1: Original signal (no filtering) ---
        signals.append(distances_)  # first entry is the raw/original signal
        
        # --- Step 2: Apply Butterworth filters for each cutoff frequency ---
        for cutoff in cutoff_frequencies[1:]:  # skip None (already added)
            normal_cutoff = cutoff / nyq  # normalize the cutoff
            if not 0 < normal_cutoff < 1:
                raise ValueError(f"Cutoff frequency {cutoff} is invalid. It must be less than Nyquist ({nyq}).")
            
            # Design the Butterworth filter and apply it using filtfilt
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, distances_)
            signals.append(filtered_signal)
        
        # === Step 3: Create a figure with 4 subplots (1 for each signal) ===
        fig, axs = plt.subplots(4, 1, figsize=(12, 14), dpi=150, sharex=True)
        
        # === Step 4: For each signal, detect and plot peaks and troughs ===
        for i, (signal, title, color) in enumerate(zip(signals, titles, colors)):
            # --- Detect peaks and troughs for this signal ---

            peaks, _ = find_peaks(signal, distance=5 , height=np.mean(signal)/2, prominence=np.mean(signal)/2)
            troughs, _ = find_peaks(-signal, distance=5, height=-np.mean(signal), prominence=np.mean(signal)/2)
            # --- Plot the signal and its detected peaks/troughs ---
            axs[i].plot(signal, color=color, label=title, linewidth=1.5)
            axs[i].scatter(peaks, signal[peaks], color='red', s=20, label='Peaks')
            axs[i].scatter(troughs, signal[troughs], color='blue', s=20, label='Troughs')
            axs[i].set_ylim(-0.2, 2.2)
            axs[i].set_title(title, fontsize=15)
            axs[i].legend(fontsize=9)
            axs[i].grid(True)
        
        # === Final labels and layout adjustments ===
        axs[-1].set_xlabel('Frame Number', fontsize=12)
        fig.text(0.04, 0.5, 'Normalized Distance', va='center', rotation='vertical', fontsize=13)
        
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        self.signal_img_path = os.path.join(self.config['save_path'], f"{self.new_filename}_signal.png")
        #fig.savefig(self.signal_img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        ##################################################################
        
        peaks, _ = find_peaks(distances_, distance=5 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
        troughs, _ = find_peaks(-distances_, distance=5, height=-np.mean(distances_), prominence=np.mean(distances_)/2)
        ############################### Compute speed signal
        time_interval = 1 / self.fps
        speed_signal = np.diff(distances_) / time_interval # Speed = Δdistance / Δtime

        ###################################################  amp

        
        while len(peaks) > 0 and len(troughs) > 0 and troughs[0] > peaks[0]:
            peaks = peaks[1:]
        
        amplitudes = []
        
        for i in range(min(len(peaks), len(troughs))):
            peak = peaks[i]
            valid_troughs = troughs[troughs < peak]
            if len(valid_troughs) == 0:
                continue
            last_trough = valid_troughs[-1]
            amp = abs(distances_[peak] - distances_[last_trough])
            amplitudes.append(amp)
        
        
        # Compute median and max amplitude
        median_amplitude = np.median(amplitudes)
        max_amplitude = np.max(amplitudes)
        avg_amplitude = np.mean(amplitudes)


        time_points_amp = np.arange(len(amplitudes)).reshape(-1, 1)
        model_amp = LinearRegression()
        model_amp.fit(time_points_amp, amplitudes)
        amp_slope = model_amp.coef_[0]

        ################################################################# Generate per-cycle speed 
        per_cycle_speed_maxima = []
        per_cycle_speed_avg = []

        for i in range(len(amplitudes) - 1):
            start_idx = peaks[i]  # Start of the window
            end_idx = peaks[i + 1]  # End of the window
            window_speed = speed_signal[start_idx:end_idx]  # Slice the speed signal
            
            if len(window_speed) > 0:  # Ensure the window is not empty
                per_cycle_speed_maxima.append(np.percentile(np.abs(window_speed), 95))
                per_cycle_speed_avg.append(np.mean(np.abs(window_speed)))
        
        # Compute the median and max of per-cycle speed maxima
        mean_percycle_max_speed = np.mean(per_cycle_speed_maxima)
        mean_percycle_avg_speed = np.mean(per_cycle_speed_avg)

        avg_speed = np.mean(np.abs(speed_signal))
        time_points_speed = np.arange(len(per_cycle_speed_avg)).reshape(-1, 1)
        model_speed = LinearRegression()
        model_speed.fit(time_points_speed, np.abs(per_cycle_speed_avg))
        speed_slope = model_speed.coef_[0]
        # Compute tapping intervals (time between consecutive maxima)
        tapping_intervals = np.diff(peaks) / self.fps

        median_tapping_interval = np.median(tapping_intervals)
        mean_tapping_interval = np.mean(tapping_intervals)

        time_points_ti = np.arange(len(tapping_intervals)).reshape(-1, 1)
        model_ti = LinearRegression()
        model_ti.fit(time_points_ti, tapping_intervals)
        ti_slope = model_ti.coef_[0]


        
        
        ################################## ratio decrement
        mid = len(amplitudes) // 2
        first_half = amplitudes[:mid]
        second_half = amplitudes[mid:]
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        amp_decrement_ratio = mean_second / mean_first 
        mid = len(np.abs(speed_signal)) // 2
        first_half = np.abs(speed_signal)[:mid]
        second_half = np.abs(speed_signal)[mid:]
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        speed_decrement_ratio = mean_second / mean_first 
        #############################  new amp ti decrement 
        

        
        

        
        ####################################################################   hesitation-halts

        std_tapping_intervals = np.std(tapping_intervals)
        cov_tapping_interval = std_tapping_intervals/mean_tapping_interval

        std_amp = np.std(amplitudes)
        cov_amp = std_amp/avg_amplitude

        std_per_cycle_speed_maxima = np.std(per_cycle_speed_maxima)
        cov_per_cycle_speed_maxima = std_amp/mean_percycle_max_speed

        std_per_cycle_speed_avg = np.std(per_cycle_speed_avg)
        cov_per_cycle_speed_avg = std_per_cycle_speed_avg/mean_percycle_avg_speed
        

        std_speed = np.std(speed_signal)
        cov_speed = std_speed/avg_speed



        # Compute total number of interruptions
        threshold = 2 * median_tapping_interval
        num_interruptions1 = sum(interval > threshold for interval in tapping_intervals)
        
        threshold = 1.5 * median_tapping_interval
        num_interruptions2 = sum(interval > threshold for interval in tapping_intervals)
        ################################################################################################
            
        # Spectral Entropy
        freqs, psd = welch(distances_, fs=self.fps)
        psd_norm = psd / np.sum(psd)
        spectral_entropy_val = entropy(psd_norm)
    
        #  Autocorrelation decay (average over first 50 lags after lag=0)
        autocorr = correlate(distances_ - np.mean(distances_), distances_ - np.mean(distances_), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr /= autocorr[0]  # normalize
        autocorr_decay = np.mean(np.abs(autocorr[1:50]))
    
    
        #  High-Frequency Power Ratio
        f_thresh = 6
        high_power = np.sum(psd[freqs > f_thresh])
        total_power = np.sum(psd)
        hf_power_ratio = high_power / total_power if total_power > 0 else 0
    
        
        #  Hjorth Parameters
        first_deriv = np.diff(distances_)
        second_deriv = np.diff(first_deriv)
        var_zero = np.var(distances_)
        var_d1 = np.var(first_deriv)
        hjorth_mob = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    
    
        #  ZCR in derivative
        zcr_deriv = np.sum(np.diff(np.sign(first_deriv)) != 0) / len(first_deriv)    
            
            
        
        #########################################################################################
        
        self.feat_name = ['ids', 'video_path', 'label',
                          'avg_amplitude', 'mean_percycle_max_speed', 'mean_percycle_avg_speed',
                          'mean_tapping_interval', 'amp_slope', 'ti_slope', 'speed_slope',
                          'cov_tapping_interval', 'cov_amp', 'cov_per_cycle_speed_maxima',
                          'cov_per_cycle_speed_avg', 'num_interruptions2',
                          'spectral_entropy', 'autocorr_decay',  'hf_power_ratio', 'hjorth_mobility', 'zcr_deriv'
                           'yesno']
    
        features = [avg_amplitude, mean_percycle_max_speed, mean_percycle_avg_speed,
                    mean_tapping_interval, amp_slope, ti_slope, speed_slope,
                    cov_tapping_interval, cov_amp, cov_per_cycle_speed_maxima,
                    cov_per_cycle_speed_avg, num_interruptions2,
                    spectral_entropy_val, autocorr_decay, 
                    hf_power_ratio, hjorth_mob, zcr_deriv]


 
        
        return features

   

    def _normalize_keypoints_distance(self, hand_landmarks):
        wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
        index_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

        wrist_to_index_mcp_length = np.sqrt((index_mcp.x - wrist.x) ** 2 +
                                            (index_mcp.y - wrist.y) ** 2 +
                                            (index_mcp.z - wrist.z) ** 2)

        if wrist_to_index_mcp_length < 1e-5:
            wrist_to_index_mcp_length = 1e-5

        normalized_keypoints = []
        for landmark in hand_landmarks:
            centered_x = landmark.x - wrist.x
            centered_y = landmark.y - wrist.y
            centered_z = landmark.z - wrist.z

            normalized_x = centered_x / wrist_to_index_mcp_length
            normalized_y = centered_y / wrist_to_index_mcp_length
            normalized_z = centered_z / wrist_to_index_mcp_length

            normalized_keypoints.append((normalized_x, normalized_y, normalized_z))


        return normalized_keypoints
    
    def draw_landmarks_on_image(self, rgb_image, detection_result, hand_to_track, label):
      hand_landmarks_list = detection_result.hand_landmarks
      handedness_list = detection_result.handedness
      annotated_image = np.copy(rgb_image)

      # Define drawing styles
      default_drawing_spec = solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
      red_drawing_spec = solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))
      blue_drawing_spec = solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(255, 0, 0))  # Blue for keypoints in normalization
      
      # Loop through the detected hands to visualize.
      for idx in range(len(hand_landmarks_list)):
          hand_landmarks = hand_landmarks_list[idx]
          handedness = handedness_list[idx]

          # Draw the key landmarks with distinct colors depending on normalization type
          for i, landmark in enumerate(hand_landmarks):
              landmark_proto = landmark_pb2.NormalizedLandmark(
                  x=landmark.x, y=landmark.y, z=landmark.z
              )
              if self.config['distance']:
                  # Keypoints used in distance-based normalization
                  if i in [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]:
                      solutions.drawing_utils.draw_landmarks(
                          annotated_image,
                          landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                          None,
                          landmark_drawing_spec=blue_drawing_spec
                      )
                  elif i in [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.THUMB_TIP]:
                      solutions.drawing_utils.draw_landmarks(
                          annotated_image,
                          landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                          None,
                          landmark_drawing_spec=red_drawing_spec
                      )
                  else:
                      solutions.drawing_utils.draw_landmarks(
                          annotated_image,
                          landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                          None,
                          landmark_drawing_spec=default_drawing_spec
                      )
              else:
                  # Keypoints used in angle-based normalization
                  if i in [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.THUMB_CMC]:
                      solutions.drawing_utils.draw_landmarks(
                          annotated_image,
                          landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                          None,
                          landmark_drawing_spec=blue_drawing_spec
                      )
                  elif i in [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.THUMB_TIP]:
                      solutions.drawing_utils.draw_landmarks(
                          annotated_image,
                          landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                          None,
                          landmark_drawing_spec=red_drawing_spec
                      )
                  else:
                      solutions.drawing_utils.draw_landmarks(
                          annotated_image,
                          landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                          None,
                          landmark_drawing_spec=default_drawing_spec
                      )

          # Draw the connections
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])
          solutions.drawing_utils.draw_landmarks(
              annotated_image,
              hand_landmarks_proto,
              solutions.hands.HAND_CONNECTIONS,
              default_drawing_spec,
              default_drawing_spec)

          # Get the top left corner of the detected hand's bounding box.
          height, width, _ = annotated_image.shape
          x_coordinates = [landmark.x for landmark in hand_landmarks]
          y_coordinates = [landmark.y for landmark in hand_landmarks]
          text_x = int(min(x_coordinates) * width)
          text_y = int(min(y_coordinates) * height) - MARGIN

          # Draw handedness (left or right hand) and the label on the image.
          cv2.putText(annotated_image, f"Label: {label}",
                      (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                      FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

      return annotated_image

    def trim_irrelevant_actions(self, sequence):
        peaks, _ = find_peaks(sequence, height=np.mean(sequence), prominence=np.mean(sequence))
        
        if len(peaks) > 10:
            start = peaks[0]
            end = peaks[-1]
            trimmed_sequence = sequence[start:end+1]
            removed_start = sequence[:start]
            removed_end = sequence[end+1:]
        else:
            trimmed_sequence = sequence
            removed_start = []
            removed_end = []
  
        return trimmed_sequence, removed_start, removed_end, peaks

    def plot_sequences(self, original_sequence, trimmed_sequence, removed_start, removed_end, peaks, label):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(original_sequence, 'r-', label='Original')
        plt.plot(peaks, np.array(original_sequence)[peaks], 'ko', label='Peaks')
        if len(removed_start) > 0:
            plt.plot(range(len(removed_start)), removed_start, 'b-', label='Removed Start')
        if len(removed_end) > 0:
            plt.plot(range(len(original_sequence) - len(removed_end), len(original_sequence)), removed_end, 'b-', label='Removed End')
        plt.title(f"Original Sequence (Score: {label})")
        plt.xlabel("Frame Index")
        plt.ylabel("Sequence")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(trimmed_sequence, 'g-', label='Trimmed')
        plt.title("Trimmed Sequence")
        plt.xlabel("Frame Index")
        plt.ylabel("Sequence")
        plt.legend()
        plt.grid(True)
        
        plt.show()

    def preprocess_and_display_video(self, video_path, label):
        """Process video and display frames with calculated distances or angles."""
        hand_to_track = 'Right' if '2R' in video_path else 'Left' if '2L' in video_path else None

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)


        normalized_path = os.path.normpath(video_path)
        base_name = os.path.basename(normalized_path)
        subfolder_name = os.path.basename(os.path.dirname(normalized_path))
        self.new_filename = f"{subfolder_name}_{base_name}"
        save_vid_path = os.path.join(self.config['save_path'], self.new_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #output = cv2.VideoWriter(save_vid_path, fourcc, self.fps, (width, height))




        sequences = []
        total_frames = 0
        detected_frames = 0
        self.palm_length = []
        model_data = self._init_hand_landmarker()
        all_keypoints = []
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            num_hands=2,
            running_mode=VisionRunningMode.VIDEO
        )
        hand_landmarker = HandLandmarker.create_from_options(options)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Filter hands based on the handedness
            if detection_result.handedness:
                filtered_landmarks = []
                for idx, handedness in enumerate(detection_result.handedness):
                    if handedness[0].category_name == hand_to_track:

                            #if self._validate_hand_with_pose(frame, detection_result.hand_landmarks[idx],  handedness[0].category_name):
    
                                filtered_landmarks.append(detection_result.hand_landmarks[idx])
                            #else:
                               #print(f"Pose validation failed for {hand_to_track} hand. Skipping frame.")
                               #self.wrong_hands.append(video_path)



                detection_result.hand_landmarks = filtered_landmarks

            if detection_result.hand_landmarks:
                detected_frames += 1
                for hand_landmarks in detection_result.hand_landmarks:
                    if self.config['distance']:
                        normalized_keypoints = self._normalize_keypoints_distance(hand_landmarks)
                        index_finger_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.THUMB_TIP]
                        sequence_value = np.sqrt((thumb_tip[0] - index_finger_tip[0]) ** 2 +
                                                 (thumb_tip[1] - index_finger_tip[1]) ** 2 +
                                                 (thumb_tip[2] - index_finger_tip[2]) ** 2)
                    else:
                        sequence_value = self.angle_signal(hand_landmarks, width, height)

                    sequences.append(sequence_value)

                    # Annotate the frame with the label only (no sequence value)
                    annotated_frame = self.draw_landmarks_on_image(frame, detection_result, hand_to_track, label)
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    #cv2.imshow('Annotated Frame', annotated_frame_bgr)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #break
                    #output.write(annotated_frame)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    frame_keypoints = []
                    for landmark in hand_landmarks:
                        frame_keypoints.append([landmark.x, landmark.y, landmark.z])  # Save original x, y, z coordinates
                    all_keypoints.append(frame_keypoints)  # Shape (21, 3) per frame



        if detected_frames / total_frames >= 0.5 :
            
            if self.config['trimmed']==False:
                sequences = np.array(sequences)
                
                #plt.plot(sequences)
                return np.array(sequences), all_keypoints
            elif self.config['trimmed']==True:
                trimmed_sequences, removed_start, removed_end, peaks = self.trim_irrelevant_actions(sequences)
                #self.plot_sequences(sequences, trimmed_sequences, removed_start, removed_end, peaks, label)
                return np.array(trimmed_sequences), all_keypoints

        else:
            print(f"Skipping video {video_path} as keypoints were not detected in more than 50% of the frames.")
            return None

        cap.release()
        hand_landmarker.close()
        #cv2.destroyAllWindows()
        #output.release()

                                    
              
    def run_savefeatures_new(self):

         pickle_file_path = os.path.join(self.config['save_path'], 'video_keypoints.pkl')
         csv_file_path = os.path.join(self.config['save_path'], 'video_features.csv')        
         failed_csv_path = os.path.join(self.config['save_path'], 'failed_videos.csv')

         video_labels = pd.read_csv(self.config['vid2score'])
         videos = video_labels['video_path']
         labels = video_labels['score']
         ids =  video_labels['id']

         # Initialize data containers
         all_video_paths = []
         all_distances = []
         all_keypoints = []
         all_labels = []
         all_ids = []
         all_fps = []
         failed_videos = []

         # Open CSV file for writing
         with open(csv_file_path, mode='w', newline='') as file:
             writer = csv.writer(file)

             self.feat_name = ['ids', 'video_path', 'label',
                                       'avg_amplitude', 'mean_percycle_max_speed', 'mean_percycle_avg_speed',
                                       'mean_tapping_interval', 'amp_slope', 'ti_slope', 'speed_slope',
                                       'cov_tapping_interval', 'cov_amp', 'cov_per_cycle_speed_maxima',
                                       'cov_per_cycle_speed_avg', 'num_interruptions2',
                                       'spectral_entropy', 'autocorr_decay', 
                                       'hf_power_ratio', 'hjorth_mobility', 'zcr_deriv', 'yesno']
             writer.writerow(self.feat_name)
             c = 0
             for vid, idi, label in tqdm(zip(videos, ids, labels), total=len(videos), desc="Processing videos"):
                 c = c + 1
                 print(c, vid)
                 #vid = vid.replace('//chansey.umcn.nl', '/data').replace('\\', '/') 

                 self.vid = vid
                 #try: 
                 distance, p_keypoint = self.preprocess_and_display_video(vid, label)
                 feature = self._extract_features(distance)
        
                    # Accumulate data in memory
                 all_video_paths.append(vid)
                 all_distances.append(distance)
                 all_keypoints.append(p_keypoint)
                 all_labels.append(label)
                 all_ids.append(idi)
                 all_fps.append(self.fps)
                    # Save features row to CSV
                 writer.writerow([idi, vid, label] + feature + ['nan'])
                 #except Exception as e:
                  #print(f"Error processing {vid}: {e}")
                  #failed_videos.append([vid, str(e)])  
         # Save all accumulated data to pickle once at the end
         final_data = {
             'video_path': all_video_paths,
             'distances': all_distances,
             'keypoints': all_keypoints,
             'label':all_labels, 
             'id':all_ids,
             'fps':all_fps
         }
         with open(pickle_file_path, 'wb') as f:
             pickle.dump(final_data, f)

         #if failed_videos:
          #  with open(failed_csv_path, mode='w', newline='') as f:
           #     writer = csv.writer(f)
            #    writer.writerow(['video_path', 'error_message'])
             #   writer.writerows(failed_videos)
            #print(f"{len(failed_videos)} videos failed. Logged to: {failed_csv_path}")
if __name__ == "__main__":
    ################### annotated
    CONFIG = {
        'id2vid': r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/id2vid.csv',
        'ids': r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/patient_id_all.csv',
        'vid2score': r'//chansey.umcn.nl/diag/Tahereh/new/src/datasets/dataset_preprocessing/segmented_ft_vid2score.csv',
        'save_path':  r'//chansey.umcn.nl/diag/Tahereh/new/src1/my HC/id_based split/ft/classification/angle',
        'distance': False,
        'trimmed':False,
    }


    # Instantiate the class and run the cross-validation
    trainer = ModelTrainer(CONFIG)
    trainer.run_savefeatures_new()

    
    
    
    
    
    
    
    
    