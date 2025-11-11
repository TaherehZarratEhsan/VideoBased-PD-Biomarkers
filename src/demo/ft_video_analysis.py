<<<<<<< HEAD
=======

>>>>>>> 2c9a4eaa7c45901e735327b566c39a485dab54ec

import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
from math import pi
import pickle
from factor_analyzer import Rotator
from factor_analyzer import FactorAnalyzer
from scipy.stats import f_oneway, shapiro, ttest_ind, mannwhitneyu
from scipy.signal import butter, filtfilt
from scipy.signal import welch, correlate
from scipy.stats import entropy
import matplotlib.cm as cm
from scipy.stats import spearmanr
sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)
import h5py
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from sklearn.linear_model import LinearRegression
import urllib.request

class ft_video_analysis:
    def __init__(self, config):
        self.config = config
        
    def _init_hand_landmarker(self):
            
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(CURRENT_DIR, 'hand_landmarker.task')
    
        # If model doesn't exist, download it
        if not os.path.exists(model_path):
            print("hand_landmarker.task not found. Downloading...")
    
            # Mediapipe official model URL (Google-hosted)
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ Downloaded hand_landmarker.task to: {model_path}")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to download model: {e}")
    
<<<<<<< HEAD
=======
        #
>>>>>>> 2c9a4eaa7c45901e735327b566c39a485dab54ec
        with open(model_path, 'rb') as file:
            model_data = file.read()
    
        return model_data
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
        
    def _extract_features(self, distances):
         distances_ = np.array(distances)
         test_type = 'ft'
         if test_type=='la':
            distances_ = distances - np.mean(distances)
            fs = 30.0                  
            order = 4                  
            nyq = 0.5 * fs             
            cutoff_frequencies =  5.0
            normal_cutoff = cutoff_frequencies / nyq  
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, distances_)
            distances_ = filtered_signal
            peaks, _ = find_peaks(distances_, distance=7 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
            troughs, _ = find_peaks(-distances_, distance=7, height=-np.mean(distances_), prominence=np.mean(distances_)/2)

         elif test_type=='ft':
         
            fs = 30.0                  
            order = 4                  
            nyq = 0.5 * fs             
            cutoff_frequencies =  9.0
            normal_cutoff = cutoff_frequencies / nyq  
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, distances_)
            #distances_ = filtered_signal
            peaks, _ = find_peaks(distances_, distance=5 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
            troughs, _ = find_peaks(-distances_, distance=5, height=-np.mean(distances_), prominence=np.mean(distances_)/2)
                


         valid_troughs = []
         for i in range(len(troughs) - 1):
            current_trough = troughs[i]
            next_trough = troughs[i + 1]
            
            peaks_between = peaks[(peaks > current_trough) & (peaks < next_trough)]
            
            if len(peaks_between) > 0:
                if i == 0:  
                    valid_troughs.append(current_trough)
                valid_troughs.append(next_trough)
        
         troughs = np.array(valid_troughs)      
         ############################################################## Compute speed signal
         time_interval = 1 / self.fps
         speed_signal = np.diff(distances_) / time_interval # Speed = Δdistance / Δtime

         ###################################################  amp
        
         
         while len(peaks) > 0 and len(troughs) > 0 and troughs[0] > peaks[0]:
             peaks = peaks[1:]
         
         amplitudes = []
         amp_frame_numbers = []

         for i in range(min(len(peaks), len(troughs))):
             peak = peaks[i]
             valid_troughs = troughs[troughs < peak]
             if len(valid_troughs) == 0:
                 continue
             last_trough = valid_troughs[-1]
             amp = abs(distances_[peak] - distances_[last_trough])
             amplitudes.append(amp)
             amp_frame_numbers.append(peak)

         
         # Compute median and max amplitude
         median_amplitude = np.median(amplitudes)
         max_amplitude = np.max(amplitudes)
         avg_amplitude = np.mean(amplitudes)


         time_points_amp = np.arange(len(amplitudes)).reshape(-1, 1)
         model_amp = LinearRegression()
         model_amp.fit(time_points_amp, amplitudes)
         amp_slope = model_amp.coef_[0]

        ############################################################# Just visualization
         amp_model = LinearRegression()
         amp_frame_numbers = np.array(amp_frame_numbers)
         amp_model.fit(amp_frame_numbers.reshape(-1, 1), amplitudes)
         amp_fitted_line = amp_model.predict(amp_frame_numbers.reshape(-1, 1))
         #amp_linear_slope = amp_model.coef_[0]

         ################################################################# Generate per-cycle speed 
         per_cycle_speed_maxima = []
         per_cycle_speed_avg = []
         per_cycle_speed_avg_frame_numbers = []

         for i in range(len(amplitudes) - 1):
             start_idx = troughs[i]      #
             end_idx = troughs[i + 1]    #     
             window_speed = speed_signal[start_idx:end_idx]  
             
             if len(window_speed) > 0:  
                 per_cycle_speed_maxima.append(np.percentile(np.abs(window_speed), 95))
                 per_cycle_speed_avg.append(np.mean(np.abs(window_speed)))
                 avg_frame = (start_idx + end_idx) // 2
                 per_cycle_speed_avg_frame_numbers.append(avg_frame)
         
         per_cycle_speed_avg = np.array(per_cycle_speed_avg)
         per_cycle_speed_avg_frame_numbers = np.array(per_cycle_speed_avg_frame_numbers).reshape(-1, 1)   
         ############################################################ just visualization 
         speed_avg_model = LinearRegression()
         speed_avg_model.fit(per_cycle_speed_avg_frame_numbers, per_cycle_speed_avg)
         speed_avg_fitted = speed_avg_model.predict(per_cycle_speed_avg_frame_numbers)
         #speed_avg_slope = speed_avg_model.coef_[0]
         ####################################################################

         # Compute the median and max of per-cycle speed maxima
         mean_percycle_max_speed = np.mean(per_cycle_speed_maxima)
         mean_percycle_avg_speed = np.mean(per_cycle_speed_avg)

         avg_speed = np.mean(np.abs(speed_signal))
         time_points_speed = np.arange(len(per_cycle_speed_avg)).reshape(-1, 1)
         model_speed = LinearRegression()
         model_speed.fit(time_points_speed, np.abs(per_cycle_speed_avg))
         speed_slope = model_speed.coef_[0]
         # Compute cycel duration
         tapping_intervals = np.diff(peaks) / self.fps

         median_tapping_interval = np.median(tapping_intervals)
         mean_tapping_interval = np.mean(tapping_intervals)

         time_points_ti = np.arange(len(tapping_intervals)).reshape(-1, 1)
         model_ti = LinearRegression()
         model_ti.fit(time_points_ti, tapping_intervals)
         ti_slope = model_ti.coef_[0]


         time_points_tap = np.arange(len(tapping_intervals))
         tap_slope, tap_intercept = np.polyfit(time_points_tap, tapping_intervals, 1)


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


             
         #########################################################################################
              
         features = {'avg_amplitude': avg_amplitude,
                     'mean_percycle_max_speed':mean_percycle_max_speed,
                     'mean_percycle_avg_speed':mean_percycle_avg_speed,
                     'mean_cycle_duration':mean_tapping_interval,
                     'amp_slope':amp_slope, 
                     'ti_slope':ti_slope,
                     'speed_slope':speed_slope,
                     'cov_tapping_interval':cov_tapping_interval, 
                     'cov_amp':cov_amp, 
                     'cov_per_cycle_speed_maxima':cov_per_cycle_speed_maxima,
                     'cov_per_cycle_speed_avg':cov_per_cycle_speed_avg,
                     'num_interruptions':num_interruptions2,
                     }
         # Convert to DataFrame
         df = pd.DataFrame([features])  # wrap in list to make one row
        
         # Save to CSV
         df.to_csv(os.path.join(self.SAVE_DIR, 'features.csv'), index=False)
          # === PLOTTING ===
                
         fig, axs = plt.subplots(3, 1, figsize=(12, 18), dpi=300)
          
          # === 1. Distance + Amplitudes + Peaks/Troughs ===
         axs[0].scatter(peaks, distances_[peaks], color='red', marker='^', s=20, label='Peaks')
         axs[0].scatter(troughs, distances_[troughs], color='black', marker='v', s=20, label='Troughs')
         axs[0].plot(range(len(distances_)), distances_, color='blue', linewidth=2, label="Distance signal")
         axs[0].plot(amp_frame_numbers, amp_fitted_line, color='red', linestyle='--', linewidth=2,
                      label=f'Amplitude slope: {amp_slope:.2e}')
         axs[0].set_xlabel('Frame number', fontsize=20)
         axs[0].set_ylabel('Distance', fontsize=20)
         axs[0].set_ylim(0, 2)
         axs[0].set_ylabel('Distance', fontsize=20, labelpad=60)
          # Legend for axs[0]
         handles0, labels0 = axs[0].get_legend_handles_labels()
         axs[0].legend(handles0, labels0, loc='upper center', bbox_to_anchor=(0.5, 1.3),
                       bbox_transform=axs[0].transAxes, ncol=len(labels0), fontsize=12, frameon=False, handletextpad=0.5)

          # === 2. Speed signal + per-cycle avg speed ===
         axs[1].plot(range(len(speed_signal)), np.abs(speed_signal), color='darkgreen', linewidth=2, label='Speed signal')
         axs[1].plot(per_cycle_speed_avg_frame_numbers, speed_avg_fitted, color='red', linestyle='--', linewidth=2,
                      label=f'Speed slope: {speed_slope:.2e}')
         axs[1].set_xlabel('Frame number', fontsize=20)
         axs[1].set_ylabel('Speed', fontsize=20)
         axs[1].set_ylim(0, 45)
         axs[1].set_ylabel('Speed', fontsize=20, labelpad=60)

          # Legend for axs[1]
         handles1, labels1 = axs[1].get_legend_handles_labels()
         axs[1].legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.3),
                       bbox_transform=axs[1].transAxes, ncol=len(labels1), fontsize=12, frameon=False, handletextpad=0.5)

          # === 3. Tapping intervals ===
         fitted_tapping_line = tap_slope * time_points_tap + tap_intercept
         axs[2].scatter(time_points_tap, tapping_intervals, color='black', s=50, label="Cycle duration")
         axs[2].plot(time_points_tap, fitted_tapping_line, color='red', linestyle='--', linewidth=2,
                      label=f'Cycle duration Slope: {tap_slope:.2e}')
         axs[2].set_xlabel('Cycle number', fontsize=20)
         axs[2].set_ylabel('Cycle duration', fontsize=20)
         axs[2].set_ylim(0, 1.2)
         axs[2].set_ylabel('Cycle duration', fontsize=20, labelpad=60)

          # Legend for axs[2]
         handles2, labels2 = axs[2].get_legend_handles_labels()
         axs[2].legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.3),
                        bbox_transform=axs[2].transAxes, ncol=len(labels2), fontsize=12, frameon=False, handletextpad=0.5)

          # === Final adjustments ===
         fig.tight_layout(pad=5.0)
         plt.subplots_adjust(top=0.95, hspace=1) 
         plot_save_path = os.path.join(self.SAVE_DIR, 'Results.png')
         plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
         
    def draw_landmarks_on_image(self, rgb_image, detection_result, hand_to_track):
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
          text_y = int(min(y_coordinates) * height) - 10


      return annotated_image

    def preprocess_and_display_video(self):
        """Process video and display frames with calculated distances or angles."""
        hand_to_track = self.config['hand2track']

        cap = cv2.VideoCapture(self.config['video_path'])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)


        normalized_path = os.path.normpath(self.config['video_path'])
        base_name = os.path.basename(normalized_path)
        subfolder_name = os.path.basename(os.path.dirname(normalized_path))
        self.new_filename = f"{subfolder_name}_{base_name}"
        
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.SAVE_DIR = os.path.join(CURRENT_DIR, "results")       
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        
        save_vid_path = os.path.join(self.SAVE_DIR, self.new_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(save_vid_path, fourcc, self.fps, (width, height))

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
    
                                filtered_landmarks.append(detection_result.hand_landmarks[idx])




                detection_result.hand_landmarks = filtered_landmarks

            if detection_result.hand_landmarks:
                detected_frames += 1
                for hand_landmarks in detection_result.hand_landmarks:
                    normalized_keypoints = self._normalize_keypoints_distance(hand_landmarks)
                    index_finger_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.THUMB_TIP]
                    sequence_value = np.sqrt((thumb_tip[0] - index_finger_tip[0]) ** 2 +
                                             (thumb_tip[1] - index_finger_tip[1]) ** 2 +
                                             (thumb_tip[2] - index_finger_tip[2]) ** 2)


                    sequences.append(sequence_value)

                    # Annotate the frame with the label only (no sequence value)
                    annotated_frame = self.draw_landmarks_on_image(frame, detection_result, hand_to_track)
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Annotated Frame', annotated_frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    output.write(annotated_frame)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    frame_keypoints = []
                    for landmark in hand_landmarks:
                        frame_keypoints.append([landmark.x, landmark.y, landmark.z])  # Save original x, y, z coordinates
                    all_keypoints.append(frame_keypoints)  # Shape (21, 3) per frame

        cap.release()
        hand_landmarker.close()
        cv2.destroyAllWindows()
        output.release()
        self._extract_features(sequences)

        
if __name__ == "__main__":

    
    #CONFIG = {
    #    'video_path':  r'//chansey.umcn.nl/diag/Tahereh/Video/Visit 2/POM2VD3156615/On_2L_cropped_square_s.MP4',
    #    'hand2track':'Left' #### Left or Right
    #}
    
    #test = ft_video_analysis(CONFIG)
    #test.preprocess_and_display_video()


    import argparse

    parser = argparse.ArgumentParser(description="Finger Tapping Video Analysis")

    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--hand2track", type=str, choices=["Left", "Right"], default="Left", help="Hand to track")

    args = parser.parse_args()

    CONFIG = {
        'video_path': args.video_path,
        'hand2track': args.hand2track,
    }

    analyzer = ft_video_analysis(CONFIG)
    analyzer.preprocess_and_display_video()


#python ft_video_analysis.py --video_path "C:/Users/Tahereh/video.MP4" --hand2track Right 

