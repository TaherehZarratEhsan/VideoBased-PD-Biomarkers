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


class feature_ext_analysis:
    def __init__(self, config):
        self.config = config
        
    def remove_consecutive_events(self, signal, peaks, troughs):
        """
        Removes the lower peak if two consecutive peaks occur without a trough,
        and similarly removes the higher trough if two consecutive troughs occur without a peak.
        """
        # Sort peaks and troughs to ensure correct order
        events = np.sort(np.concatenate((peaks, troughs)))
        event_types = np.array(['peak' if idx in peaks else 'trough' for idx in events])
    
        cleaned_peaks = set(peaks)
        cleaned_troughs = set(troughs)
    
        i = 0
        while i < len(events) - 1:
            current, next_event = events[i], events[i + 1]
            current_type, next_type = event_types[i], event_types[i + 1]
    
            if current_type == next_type:
                # Consecutive peaks
                if current_type == 'peak':
                    # Remove the peak with lower height
                    if signal[current] < signal[next_event]:
                        cleaned_peaks.discard(current)
                    else:
                        cleaned_peaks.discard(next_event)
                # Consecutive troughs
                else:
                    # Remove the trough with higher value (less negative, shallower trough)
                    if signal[current] > signal[next_event]:
                        cleaned_troughs.discard(current)
                    else:
                        cleaned_troughs.discard(next_event)
                # After removal, update events and event_types
                events = np.sort(np.concatenate((list(cleaned_peaks), list(cleaned_troughs))))
                event_types = np.array(['peak' if idx in cleaned_peaks else 'trough' for idx in events])
                i = 0  # restart to handle new event list correctly
                continue
            i += 1
    
        return np.array(sorted(cleaned_peaks)), np.array(sorted(cleaned_troughs))

        
    def _extract_features(self, distances, fps):
         distances_ = np.array(distances)

         if self.config['test_type']=='la':
            distances_ = distances - np.mean(distances)
            fs = 30.0                  
            order = 4                  
            nyq = 0.5 * fs             
            cutoff_frequencies =  5.0
            normal_cutoff = cutoff_frequencies / nyq  # normalize the cutoff
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, distances_)
            distances_ = filtered_signal
            peaks, _ = find_peaks(distances_, distance=7 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
            troughs, _ = find_peaks(-distances_, distance=7, height=-np.mean(distances_), prominence=np.mean(distances_)/2)

         elif self.config['test_type']=='ft':
         
            fs = 30.0                  
            order = 4                  
            nyq = 0.5 * fs             
            cutoff_frequencies =  9.0
            normal_cutoff = cutoff_frequencies / nyq  # normalize the cutoff
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, distances_)
            distances_ = filtered_signal
            peaks, _ = find_peaks(distances_, distance=5 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
            troughs, _ = find_peaks(-distances_, distance=5, height=-np.mean(distances_), prominence=np.mean(distances_)/2)
                
         
            
         #peaks, troughs = self.remove_consecutive_events(distances_, peaks, troughs)
         ############################################################## Compute speed signal
         time_interval = 1 / fps
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
         tapping_intervals = np.diff(peaks) / fps

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

         # FFT-based maximum magnitude feature (dominant frequency component magnitude)
         n = len(distances_)
         fft_spectrum = np.abs(np.fft.rfft(distances_))
         fft_spectrum_no_dc = fft_spectrum[1:]  # Exclude DC component (index 0)
         max_freq_magnitude = np.max(fft_spectrum_no_dc)

         #  Hjorth Parameters
         first_deriv = np.diff(distances_)
         var_zero = np.var(distances_)
         var_d1 = np.var(first_deriv)
         hjorth_mob = np.sqrt(var_d1 / var_zero) #if var_zero > 0 else 0
             

         #########################################################################################
         
     
         features = {'avg_amplitude': avg_amplitude,
                     'mean_percycle_max_speed':mean_percycle_max_speed,
                     'mean_percycle_avg_speed':mean_percycle_avg_speed,
                     'mean_tapping_interval':mean_tapping_interval,
                     'amp_slope':amp_slope, 
                     'ti_slope':ti_slope,
                     'speed_slope':speed_slope,
                     'cov_tapping_interval':cov_tapping_interval, 
                     'cov_amp':cov_amp, 
                     'cov_per_cycle_speed_maxima':cov_per_cycle_speed_maxima,
                     'cov_per_cycle_speed_avg':cov_per_cycle_speed_avg,
                     'num_interruptions2':num_interruptions2,
                     'hjorth_mob':hjorth_mob, 
                     'max_freq_magnitude':max_freq_magnitude, 
                     }

         feat_name = ['ids', 'video_path', 'label'] + list(features.keys())
  
         
         return list(features.values()),  feat_name


                  
    
    def boxplots_with_anova_brackets(self):
        df = self.data
        cmap = cm.get_cmap('Blues')

        if self.config['test_type']=='ft': 
            column_titles = {
            'avg_amplitude': 'Average amplitude',
            'mean_tapping_interval': 'Average cycle duration',
            'mean_percycle_max_speed': 'CMS',
            'mean_percycle_avg_speed': 'CAS',
            'amp_slope': 'Amplitude slope',
            'ti_slope': 'cycle duration slope',
            'speed_slope': 'Speed slope',
            'cov_tapping_interval': 'COV of tapping intervals',
            'cov_amp': 'COV of amplitude',
            'cov_per_cycle_speed_maxima': 'COV of CMS',
            'cov_per_cycle_speed_avg': 'COV of CAS',
            'num_interruptions2': 'Number of interruptions',
            
        }
        elif     self.config['test_type']=='la' :
            column_titles = {
                'avg_amplitude': 'Average amplitude',
                'mean_tapping_interval': 'Average cycle duration',
                'mean_percycle_max_speed': 'CMS',
                'mean_percycle_avg_speed': 'CAS',
                'amp_slope': 'Amplitude slope',
                'ti_slope': ' Cycle duration slope',
                'speed_slope': 'Speed slope',
                'cov_tapping_interval': 'COV of cycle time slope',
                'cov_amp': 'COV of amplitude',
                'cov_per_cycle_speed_maxima': 'COV of CMS',
                'cov_per_cycle_speed_avg': 'COV of CAS',
                'num_interruptions2': 'Number of interruptions',
            }
        group_titles = [
            'A) Hypokinesia',
            'B) Bradykinesia',
            'C) Combined Hypo- & Bradykinesia',
            'D) Combined Hypo- & Bradykinesia',
            'E) Sequence Effect',
            'F) Sequence Effect',
            'G) Sequence Effect',
            'H) Hesitation-Halts',
            'I) Hesitation-Halts',
            'J) Hesitation-Halts',
            'K) Hesitation-Halts',
            'L) Hesitation-Halts'
        ]
    
        feature_columns = [col for col in column_titles if col in df.columns]
        features_with_labels = df[feature_columns].copy()
        features_with_labels['Label'] = df['label']
    
        def plot_and_test(dataframe, label_set, reference_label, save_suffix):
            from pandas.api.types import CategoricalDtype

            label_type = CategoricalDtype(categories=label_set, ordered=True)
            dataframe['Label'] = dataframe['Label'].astype(label_type)
            
            
            all_results = []
            sns.set(style="whitegrid")
            fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 16))
            axes = axes.flatten()
            


            for i, (column, group) in enumerate(zip(feature_columns, group_titles)):
                ax = axes[i]
                
                sns.boxplot(x='Label', y=column, data=dataframe, ax=ax, color='lightblue', showfliers=False, order=label_set)
                # Apply gradient colors to boxes
                num_boxes = len(label_set)
                for j, patch in enumerate(ax.patches):
                    color_value = (j + 1) / num_boxes  # Normalize to [0,1]
                    patch.set_facecolor(cmap(color_value))
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)
                # ANOVA
                groups = [dataframe[dataframe['Label'] == label][column].dropna().values for label in label_set]
                non_empty_groups = [g for g in groups if len(g) > 1]

                f_stat, anova_p = f_oneway(*non_empty_groups)
                ax.set_title(f"{group} (ANOVA p = {anova_p:.3g})", fontsize=13, fontweight='bold')
    
                all_results.append({
                    'feature': column,
                    'comparison': f'ANOVA',
                    'test': 'ANOVA',
                    'statistic': round(f_stat, 5),
                    'p_value': round(anova_p, 5)
                })
    
                # Positioning for brackets
                tick_pos = {label: pos for pos, label in enumerate(label_set)}

    
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                base_y = y_max - 0.05 * y_range
                bracket_height = 0.04 * y_range
                text_offset = 0.005 * y_range
    
                comparisons = []
                ref_group = dataframe[dataframe['Label'] == reference_label][column].dropna()
    
                for other_label in label_set:
                    if other_label == reference_label:
                        continue
                    other_group = dataframe[dataframe['Label'] == other_label][column].dropna()
    
                    if len(ref_group) < 3 or len(other_group) < 3:
                        all_results.append({
                            'feature': column,
                            'comparison': f"{reference_label} vs {other_label}",
                            'test': 'skipped',
                            'statistic': None,
                            'p_value': None
                        })
                        continue
    
                    normal_ref = shapiro(ref_group).pvalue > 0.05
                    normal_other = shapiro(other_group).pvalue > 0.05
    
                    if normal_ref and normal_other:
                        stat, p = ttest_ind(ref_group, other_group, equal_var=False)
                        test = 't-test'
                    else:
                        stat, p = mannwhitneyu(ref_group, other_group, alternative='two-sided')
                        test = 'Mann-Whitney'
    
                    # Bonferroni correction
                    p = min(p * (len(label_set) - 1), 1.0)
    
                    if p < 0.001:
                        label = '***'
                    elif p < 0.01:
                        label = '**'
                    elif p < 0.05:
                        label = '*'
                    else:
                        label = 'ns'
    
                    all_results.append({
                        'feature': column,
                        'comparison': f"{reference_label} vs {other_label}",
                        'test': test,
                        'statistic': round(stat, 5),
                        'p_value': round(p, 5)
                    })
    
                    x1 = tick_pos[reference_label]
                    x2 = tick_pos[other_label]
                    comparisons.append((abs(x2 - x1), x1, x2, label))
    
                comparisons.sort()
                for idx, (_, x1, x2, label) in enumerate(comparisons):
                    y = base_y + idx * (bracket_height + text_offset * 3)
                    ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], lw=1.5, c='k')
                    ax.text((x1 + x2) / 2, y + bracket_height - text_offset, label,
                            ha='center', va='bottom', fontsize=12, fontweight='bold', color='k')
    
                ax.set_ylabel(column_titles.get(column, column), fontsize=11)
                ax.set_xlabel("UPDRS score", fontsize=11)
                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(True, linestyle='--', linewidth=0.6)
    
            for j in range(len(feature_columns), len(axes)):
                fig.delaxes(axes[j])
    
            plt.tight_layout()
            img_path = os.path.join(self.config['save_path'], f'boxplot_{save_suffix}.png')
            csv_path = os.path.join(self.config['save_path'], f'stats_{save_suffix}.csv')
            plt.savefig(img_path, dpi=300)
            pd.DataFrame(all_results).to_csv(csv_path, index=False)
            print(f"Saved plot: {img_path}")
            print(f"Saved stats: {csv_path}")
    
        ### 1. Always: Plot 0–4
        subset_0_4 = features_with_labels[features_with_labels['Label'].isin([0, 1, 2, 3, 4])]
        plot_and_test(subset_0_4, [0, 1, 2, 3, 4], 0, save_suffix='updrs_0_4')
    
        ### 2. If control included: Also plot control vs 0–4
        if self.config['include_control']:
            subset_control = features_with_labels[features_with_labels['Label'].isin(['Control', 0, 1, 2, 3, 4])]
            plot_and_test(subset_control, ['Control', 0, 1, 2, 3, 4], 'Control', save_suffix='control_vs_updrs')
    
    def boxplots_with_anova_brackets_new(self):
        df = self.data
        cmap = cm.get_cmap('Blues')

        if self.config['test_type']=='ft': 
            column_titles = {
            'avg_amplitude': 'Average amplitude',
            'mean_tapping_interval': 'Average cycle duration',
            'mean_percycle_max_speed': 'CMS',
            'mean_percycle_avg_speed': 'CAS',
            'amp_slope': 'Amplitude slope',
            'ti_slope': 'Cycle duration slope',
            'speed_slope': 'Speed slope',
            'cov_tapping_interval': 'COV of tapping intervals',
            'cov_amp': 'COV of amplitude',
            'cov_per_cycle_speed_maxima': 'COV of CMS',
            'cov_per_cycle_speed_avg': 'COV of CAS',
            'num_interruptions2': 'Number of interruptions',
            'hjorth_mob':'hjorth_mob', 
            'max_freq_magnitude':'max_freq_magnitude', 
            
        }
        elif     self.config['test_type']=='la' :
            column_titles = {
                'avg_amplitude': 'Average amplitude',
                'mean_tapping_interval': 'Average cycle duration',
                'mean_percycle_max_speed': 'CMS',
                'mean_percycle_avg_speed': 'CAS',
                'amp_slope': 'Amplitude slope',
                'ti_slope': 'Cycle duration slope',
                'speed_slope': 'Speed slope',
                'cov_tapping_interval': 'COV of cycle time slope',
                'cov_amp': 'COV of amplitude',
                'cov_per_cycle_speed_maxima': 'COV of CMS',
                'cov_per_cycle_speed_avg': 'COV of CAS',
                'num_interruptions2': 'Number of interruptions',
                'hjorth_mob':'hjorth_mob', 
                'max_freq_magnitude':'max_freq_magnitude', 
                
            }
        group_titles = [
            'Hypokinesia',
            'Bradykinesia',
            'Combined Hypo- & Bradykinesia',
            'Combined Hypo- & Bradykinesia',
            'Sequence Effect',
            'Sequence Effect',
            'Sequence Effect',
            'Hesitation-Halts',
            'Hesitation-Halts',
            'Hesitation-Halts',
            'Hesitation-Halts',
            'Hesitation-Halts',
            'Hesitation-Halts',
            'Hesitation-Halts'
        ]
    
        feature_columns = [col for col in column_titles if col in df.columns]
        features_with_labels = df[feature_columns].copy()
        features_with_labels['Label'] = df['label']
    
        def plot_and_test(dataframe, label_set, reference_label, save_suffix):
            from pandas.api.types import CategoricalDtype

            label_type = CategoricalDtype(categories=label_set, ordered=True)
            dataframe['Label'] = dataframe['Label'].astype(label_type)
            
            
            all_results = []
            sns.set(style="whitegrid")
            fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 16))
            axes = axes.flatten()
            


            for i, (column, group) in enumerate(zip(feature_columns, group_titles)):
                ax = axes[i]
                
                sns.boxplot(x='Label', y=column, data=dataframe, ax=ax, color='lightblue', showfliers=False, order=label_set)
                # Apply gradient colors to boxes
                num_boxes = len(label_set)
                for j, patch in enumerate(ax.patches):
                    color_value = (j + 1) / num_boxes  # Normalize to [0,1]
                    patch.set_facecolor(cmap(color_value))
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)
                # ANOVA
                groups = [dataframe[dataframe['Label'] == label][column].dropna().values for label in label_set]
                non_empty_groups = [g for g in groups if len(g) > 1]

                f_stat, anova_p = f_oneway(*non_empty_groups)
                ax.set_title(f"{group} (ANOVA p = {anova_p:.3g})", fontsize=13, fontweight='bold')
    
                all_results.append({
                    'feature': column,
                    'comparison': f'ANOVA',
                    'test': 'ANOVA',
                    'statistic': round(f_stat, 5),
                    'p_value': round(anova_p, 5)
                })
    
                # Positioning for brackets
                tick_pos = {label: pos for pos, label in enumerate(label_set)}

    
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                base_y = y_max - 0.05 * y_range
                bracket_height = 0.04 * y_range
                text_offset = 0.005 * y_range
    
                comparisons = []
                ref_group = dataframe[dataframe['Label'] == reference_label][column].dropna()
    
                for other_label in label_set:
                    if other_label == reference_label:
                        continue
                    other_group = dataframe[dataframe['Label'] == other_label][column].dropna()
    
                    if len(ref_group) < 3 or len(other_group) < 3:
                        all_results.append({
                            'feature': column,
                            'comparison': f"{reference_label} vs {other_label}",
                            'test': 'skipped',
                            'statistic': None,
                            'p_value': None
                        })
                        continue
    
                    normal_ref = shapiro(ref_group).pvalue > 0.05
                    normal_other = shapiro(other_group).pvalue > 0.05
    
                    if normal_ref and normal_other:
                        stat, p = ttest_ind(ref_group, other_group, equal_var=False)
                        test = 't-test'
                    else:
                        stat, p = mannwhitneyu(ref_group, other_group, alternative='two-sided')
                        test = 'Mann-Whitney'
    
                    # Bonferroni correction
                    p = min(p * (len(label_set) - 1), 1.0)
    
                    if p < 0.001:
                        label = '***'
                    elif p < 0.01:
                        label = '**'
                    elif p < 0.05:
                        label = '*'
                    else:
                        label = 'ns'
    
                    all_results.append({
                        'feature': column,
                        'comparison': f"{reference_label} vs {other_label}",
                        'test': test,
                        'statistic': round(stat, 5),
                        'p_value': round(p, 5)
                    })
    
                    x1 = tick_pos[reference_label]
                    x2 = tick_pos[other_label]
                    comparisons.append((abs(x2 - x1), x1, x2, label))
    
                comparisons.sort()
                for idx, (_, x1, x2, label) in enumerate(comparisons):
                    y = base_y + idx * (bracket_height + text_offset * 3)
                    ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], lw=1.5, c='k')
                    ax.text((x1 + x2) / 2, y + bracket_height - text_offset, label,
                            ha='center', va='bottom', fontsize=12, fontweight='bold', color='k')
    
                ax.set_ylabel(column_titles.get(column, column), fontsize=11)
                ax.set_xlabel("UPDRS score", fontsize=11)
                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(True, linestyle='--', linewidth=0.6)
    
            for j in range(len(feature_columns), len(axes)):
                fig.delaxes(axes[j])
    
            plt.tight_layout()
            img_path = os.path.join(self.config['save_path'], f'boxplot_{save_suffix}.png')
            csv_path = os.path.join(self.config['save_path'], f'stats_{save_suffix}.csv')
            plt.savefig(img_path, dpi=300)
            pd.DataFrame(all_results).to_csv(csv_path, index=False)
            print(f"Saved plot: {img_path}")
            print(f"Saved stats: {csv_path}")
    
        ### 1. Always: Plot 0–4
        subset_0_4 = features_with_labels[features_with_labels['Label'].isin([0, 1, 2, 3, 4])]
        plot_and_test(subset_0_4, [0, 1, 2, 3, 4], 0, save_suffix='updrs_0_4')
    
        ### 2. If control included: Also plot control vs 0–4
        if self.config['include_control']:
            subset_control = features_with_labels[features_with_labels['Label'].isin(['Control', 0, 1, 2, 3, 4])]
            plot_and_test(subset_control, ['Control', 0, 1, 2, 3, 4], 'Control', save_suffix='control_vs_updrs')
        
        
    def clustering_analysis(self): 
       """
       Perform clustering analysis and visualize the results with custom feature titles.
       """
       # Load the CSV file
       self.features_df = self.data
       
       # Define custom titles for the columns
       column_titles = {
           'avg_amplitude': 'Average amplitude (hypokinesia)',
           'mean_tapping_interval': 'Average cycle duration (bradykinesia)',
           'mean_percycle_max_speed': 'CMS (combined hypo- & bradykinesia)',
           'mean_percycle_avg_speed': 'CAS (combined hypo- & bradykinesia)',
           'amp_slope': 'Amplitude slope (sequence effect)',
           'ti_slope': ' Cycle duration slope (sequence effect)',
           'speed_slope': 'Speed slope (sequence effect)',
           'cov_tapping_interval': 'COV of cycle duration (hesitation-halts)',
           'cov_amp': 'COV of amplitude (hesitation-halts)',
           'cov_per_cycle_speed_maxima': 'COV of CMS (hesitation-halts)',
           'cov_per_cycle_speed_avg': 'COV of CAS (hesitation-halts)',
           'num_interruptions2': 'Number of interruptions (hesitation-halts)',
           'hjorth_mob':'Hjorth mobility (hesitation-halts)', 
           'max_freq_magnitude':'Maximum frequency magnitude (hesitation-halts)', 
           
       }
   
       # Exclude non-feature columns
       non_feature_columns = {'ids', 'video_path', 'label', 'hjorth_mob', 'max_freq_magnitude'}
       #non_feature_columns = {'ids', 'video_path', 'label', 'mean_percycle_avg_speed', 'mean_percycle_max_speed'}

       # Use the order from column_titles
       self.feature_columns = [
            col for col in column_titles 
            if col in self.features_df.columns and col not in non_feature_columns
        ]       
       # Subset the data
       data = self.features_df[self.feature_columns]
       
       # Map columns to custom titles in the same order
       self.custom_labels = [column_titles.get(col, col) for col in self.feature_columns]
   
       # Standardize the features
       scaler = StandardScaler()
       self.data_scaled = scaler.fit_transform(data)
       
       ###############################################  pca/ kmeans
       self.applypca()
       # self.findoptimal_clusters()
       # self.apply_kmeans()
       # self.apply_dbscan()
       #################################################################     dbscan
       self.plot_scatter_with_correlation(
            data=self.data,
            label_col='label',
            features=[col for col in column_titles if col in self.data.columns],
            feature_titles=column_titles,
            save_path=self.config['save_path'],
            save_suffix='updrs_spearman'
        )

    def apply_dbscan(self):
       # Apply DBSCAN Clustering on the whole dataset
       dbscan = DBSCAN(eps= 1, min_samples=5, metric='euclidean')
       self.features_df['DBSCAN_Cluster'] = dbscan.fit_predict(self.data_scaled)
       
       # Apply HDBSCAN Clustering on the whole dataset
       hdbscan_clusterer = HDBSCAN(min_cluster_size=20, min_samples=20, metric='cosine')
       self.features_df['HDBSCAN_Cluster'] = hdbscan_clusterer.fit_predict(self.data_scaled)
              
       # Reduce data to first 3 principal components for visualization
       pca = PCA(n_components=3)
       data_pca = pca.fit_transform(self.data_scaled)
       
       data_umap = umap.UMAP(n_components=3, random_state=42).fit_transform(self.data_scaled)
              
       
       # Define distinct colors for clusters
       for method, cluster_column in zip(['DBSCAN', 'HDBSCAN'], ['DBSCAN_Cluster', 'HDBSCAN_Cluster']):
           unique_clusters = np.unique(self.features_df[cluster_column])
           color_palette = sns.color_palette("tab10", len(unique_clusters))  # Generate distinct colors
           cluster_colors = {label: color_palette[i] for i, label in enumerate(unique_clusters)}
       
           # Create a single figure with two subplots
           fig = plt.figure(figsize=(16, 6))
           
           for i, (data_3d, name) in enumerate(zip([data_pca, data_umap], ['PCA', 'UMAP'])):
               ax = fig.add_subplot(1, 2, i + 1, projection='3d')
       
               # Assign distinct colors based on cluster labels
               labels = self.features_df[cluster_column]
               colors = [cluster_colors[label] if label in cluster_colors else 'black' for label in labels]
               scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=colors, s=50)
       
               # Annotate clusters with sample counts
               unique_labels = np.unique(labels)
               for label in unique_labels:
                   cluster_points = data_3d[np.where(labels == label)]
                   if len(cluster_points) > 0:
                       centroid = cluster_points.mean(axis=0)
                       count = len(cluster_points)
                       ax.text(centroid[0], centroid[1], centroid[2], str(count), fontsize=12, weight='bold', color='red')
       
               ax.set_title(f"{method} - {name}", fontsize=16, fontweight='bold')
               ax.set_xlabel(f"{name} Component 1", fontsize=12, fontweight='bold')
               ax.set_ylabel(f"{name} Component 2", fontsize=12, fontweight='bold')
               ax.set_zlabel(f"{name} Component 3", fontsize=12, fontweight='bold')
       
           plt.tight_layout()
           save_path = os.path.join(self.config['save_path'], f'{method.lower()}_pca_umap_3d_clusters.png')
           plt.savefig(save_path, dpi=300)
           plt.show()
           print(f"{method} PCA & UMAP 3D clustering visualization saved to {save_path}")
               

       # Separate Violin plots for DBSCAN and HDBSCAN clusters
       for method, cluster_column in zip(['DBSCAN', 'HDBSCAN'], ['DBSCAN_Cluster', 'HDBSCAN_Cluster']):
           fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 20))
           fig.suptitle(f"Violin Plots for {method} Clusters", fontsize=18, fontweight='bold')
           axes = axes.flatten()
           
           for i, feature in enumerate(self.feature_columns):
               if i < len(axes):
                   ax = axes[i]
                   
                   # Violin Plot for the clustering method
                   sns.violinplot(data=self.features_df, x=cluster_column, y=feature, ax=ax, palette="tab10", inner="quartile")
                   
                   ax.set_xlabel("Cluster")
                   ax.set_ylabel(self.column_titles.get(feature, feature))
                   ax.grid(True)
       
           plt.tight_layout(rect=[0, 0.03, 1, 0.97])
           save_path_violin = os.path.join(self.config['save_path'], f'violin_plot_{method.lower()}_clusters.png')
           plt.savefig(save_path_violin, dpi=300)
           plt.show()
           print(f"Violin plot visualization for {method} saved to {save_path_violin}")
    def applypca(self):
       # Perform PCA with 8 components
       num_components = 12
       pca = PCA(n_components=num_components)
       pca.fit(self.data_scaled)
   
       # Explained variance ratio
       explained_variance_ratio = pca.explained_variance_ratio_
       cumulative_variance = np.cumsum(explained_variance_ratio)
       
       print(f"Total explained variance with {num_components} components: {cumulative_variance[-1]:.4f}")
   
       # Scree plot with cumulative variance
       fig, ax1 = plt.subplots(figsize=(10, 5))
       ax1.bar(range(1, num_components + 1), explained_variance_ratio, color="skyblue", alpha=0.7, label="Individual Explained Variance")
       ax1.set_xlabel('Principal Components')
       ax1.set_ylabel('Explained Variance Ratio')
       ax1.set_title('Scree plot')
       ax1.set_xticks(range(1, num_components + 1))
       ax1.legend(loc="upper left")
   
       ax2 = ax1.twinx()
       ax2.plot(range(1, num_components + 1), cumulative_variance, marker='o', linestyle='-', color='r', label="Cumulative Variance")
       ax2.set_ylabel("Cumulative Explained Variance")
       ax2.set_ylim(0, 1.05)  # Ensures the cumulative variance plot starts from zero
       ax2.legend(loc="upper right")
       scree_path = os.path.join(self.config['save_path'], 'correlation+pca','scree_8pcs.png')
       plt.savefig(scree_path, dpi=300)
       plt.grid(True)
       plt.show()
   
       # Extract PCA loadings (only 8 components)
       loadings_pca = pca.components_.T  # Transpose to match features with PCs
       loadings_pca_df = pd.DataFrame(loadings_pca, index=self.custom_labels, columns=[f'PC{i+1}' for i in range(num_components)])
   
       # PCA Loadings Heatmap (First 8 PCs)
       plt.figure(figsize=(12, 8))
       sns.heatmap(loadings_pca_df, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
       plt.title("PCA Loadings Heatmap")
       plt.xlabel("Principal Components")
       plt.ylabel("Features")
       plt.xticks(rotation=45)
       plt.tight_layout()
       loading_path = os.path.join(self.config['save_path'], 'correlation+pca','loading_pca_8pcs.png')
       plt.savefig(loading_path, dpi=300)
       plt.show()
   
       # Apply Factor Analysis with 8 components
       fa = FactorAnalyzer(method='principal', n_factors=num_components, rotation=None)
       fa.fit(self.data_scaled)
       loadings = fa.loadings_
       loadings_df = pd.DataFrame(loadings, index=self.custom_labels, columns=[f'PC{i+1}' for i in range(num_components)])
   
       # Varimax Rotation for 8 PCs
       rotator = Rotator()
       rotated_loadings = rotator.fit_transform(fa.loadings_)
       rotated_loadings_df = pd.DataFrame(rotated_loadings, index=self.custom_labels, columns=[f'Rotated PC{i+1}' for i in range(num_components)])
   
       # Rotated PCA Loadings Heatmap
       plt.figure(figsize=(12, 8))
       
       # Create heatmap
       ax = sns.heatmap(rotated_loadings_df.iloc[:, :num_components], cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
       
       plt.title("Varimax Rotated PCA Loadings Heatmap")
       plt.xlabel("Rotated Principal Components")
       plt.ylabel("Features")
       
       # Get current x-tick positions and labels
       xticks_positions = ax.get_xticks()
       xticks_labels = [label.get_text() for label in ax.get_xticklabels()]
       
       # Shift tick positions slightly to the left
       shift_amount = 0.1 # Adjust this value as needed
       new_xticks_positions = [pos + shift_amount for pos in xticks_positions]
       
       # Apply new positions
       ax.set_xticks(new_xticks_positions)
       ax.set_xticklabels(xticks_labels, rotation=45, ha='right')  # Maintain rotation
       
       plt.tight_layout()
       
       # Save the figure
       rotated_path = os.path.join(self.config['save_path'],'correlation+pca', 'rotated_8pcs.png')
       plt.savefig(rotated_path, dpi=300)
       plt.show()
    def findoptimal_clusters(self):
       # Range of clusters to test
       k_range = range(2, 11)
       
       # Elbow Method (SSE)
       sse = []
       for k in k_range:
           kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
           kmeans.fit(self.data_scaled)
           sse.append(kmeans.inertia_)
       
       optimal_k_elbow = k_range[np.argmin(np.gradient(sse))]
       plt.figure(figsize=(10, 6))
       plt.plot(k_range, sse, marker='o')
       plt.axvline(optimal_k_elbow, color='red', linestyle='--', label=f'Optimal K={optimal_k_elbow}')
       plt.title("Elbow Method Analysis")
       plt.xlabel("Number of Clusters")
       plt.ylabel("Sum of Squared Errors (SSE)")
       plt.legend()
       plt.grid(True)
       elbow_path = os.path.join(self.config['save_path'], 'elbow_method.png')
       plt.savefig(elbow_path, dpi=300)
       plt.close()
       print(f"Elbow method plot saved to {elbow_path}")
       
       # Silhouette Score Analysis
       silhouette_scores = []
       for k in k_range:
           kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
           labels = kmeans.fit_predict(self.data_scaled)
           silhouette_scores.append(silhouette_score(self.data_scaled, labels))
       
       optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
       plt.figure(figsize=(10, 6))
       plt.plot(k_range, silhouette_scores, marker='o')
       plt.axvline(optimal_k_silhouette, color='red', linestyle='--', label=f'Optimal K={optimal_k_silhouette}')
       plt.title("Silhouette Score Analysis")
       plt.xlabel("Number of Clusters")
       plt.ylabel("Silhouette Score")
       plt.legend()
       plt.grid(True)
       silhouette_path = os.path.join(self.config['save_path'], 'silhouette_analysis.png')
       plt.savefig(silhouette_path, dpi=300)
       plt.close()
       print(f"Silhouette analysis plot saved to {silhouette_path}")
       
       # Davies-Bouldin Index Analysis
       davies_scores = []
       for k in k_range:
           kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
           labels = kmeans.fit_predict(self.data_scaled)
           davies_scores.append(davies_bouldin_score(self.data_scaled, labels))
       
       optimal_k_davies = k_range[np.argmin(davies_scores)]
       plt.figure(figsize=(10, 6))
       plt.plot(k_range, davies_scores, marker='o')
       plt.axvline(optimal_k_davies, color='red', linestyle='--', label=f'Optimal K={optimal_k_davies}')
       plt.title("Davies-Bouldin Index Analysis")
       plt.xlabel("Number of Clusters")
       plt.ylabel("Davies-Bouldin Score")
       plt.legend()
       plt.grid(True)
       davies_path = os.path.join(self.config['save_path'], 'davies_bouldin_analysis.png')
       plt.savefig(davies_path, dpi=300)
       plt.close()
       print(f"Davies-Bouldin analysis plot saved to {davies_path}")
       
       print("Clustering analysis completed using multiple methods.")
    def apply_kmeans(self):
       optimal_clusters = 4
       kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
       kmeans_labels = kmeans.fit_predict(self.data_scaled)
   
       # Assign cluster labels to the original DataFrame
       self.features_df['Cluster'] = kmeans_labels
       
       # Test for statistical significance of features across clusters
       self.test_features_statistical_significance(self.features_df, self.feature_columns)

       ############################################################################# 3D Clustering Visualizations
       # 3D Clustering Visualizations
       data_pca = PCA(n_components=3).fit_transform(self.data_scaled)
       data_tsne = TSNE(n_components=3, random_state=42, perplexity=30, learning_rate=200).fit_transform(self.data_scaled)
       data_umap = umap.UMAP(n_components=3, random_state=42).fit_transform(self.data_scaled)
       
       # Check shapes before visualization
       print(f"PCA Shape: {data_pca.shape}")  # Should be (num_samples, 3)
       print(f"t-SNE Shape: {data_tsne.shape}")  # Should be (num_samples, 3)
       print(f"UMAP Shape: {data_umap.shape}")  # Should be (num_samples, 3)
       
       import matplotlib.colors as mcolors
       
       # Define distinct colors for clusters
       unique_clusters = np.unique(kmeans_labels)
       color_palette = sns.color_palette("tab10", len(unique_clusters))  # Generate distinct colors
       cluster_colors = {label: color_palette[i] for i, label in enumerate(unique_clusters)}
       
       # 3D Clustering Visualizations
       for method, labels, data_3d, name in zip(
           ['K-Means', 'K-Means', 'K-Means'],
           [kmeans_labels, kmeans_labels, kmeans_labels], 
           [data_pca, data_tsne, data_umap], 
           ['pca', 'tsne', 'umap']  # 
       ):

           fig = plt.figure(figsize=(10, 8))
           ax = fig.add_subplot(111, projection='3d')
       
           # Assign distinct colors based on cluster labels
           colors = [cluster_colors[label] for label in labels]
       
           scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=colors, s=50)
       
           # Annotate clusters with sample counts
           unique_labels = np.unique(labels)
           for label in unique_labels:
               cluster_points = data_3d[np.where(labels == label)]  # FIXED: Corrected indexing
               centroid = cluster_points.mean(axis=0)
               count = len(cluster_points)
               ax.text(centroid[0], centroid[1], centroid[2], str(count), fontsize=12, weight='bold', color='red')
       
           ax.set_title(f"{method}-{name.upper()}", fontsize=16, fontweight='bold')
           ax.set_xlabel(f"{name.upper()} Component 1", fontsize=12, fontweight='bold')
           ax.set_ylabel(f"{name.upper()} Component 2", fontsize=12, fontweight='bold')
           ax.set_zlabel(f"{name.upper()} Component 3", fontsize=12, fontweight='bold')
       
           save_path = os.path.join(self.config['save_path'], f'{method.lower()}_{name}_3d_clusters.png')
           plt.tight_layout()
           plt.savefig(save_path, dpi=300)
           plt.close()
           print(f"{method} {name.upper()} 3D clustering visualization saved to {save_path}")

       
       #################################################################################################################################
       # Prepare Cluster Centers for Bar, Radar, and Parallel Coordinate Plots
       cluster_centers_kmeans = pd.DataFrame(kmeans.cluster_centers_, columns=self.custom_labels)

       # Bar Plots for Clusters
       for method, cluster_centers in zip(['K-Means'], [cluster_centers_kmeans]):
           for i, cluster in enumerate(cluster_centers.values):
               plt.figure(figsize=(10, 6))
               plt.bar(self.custom_labels, cluster, color="skyblue")
               plt.xticks(rotation=45, ha="right")
               plt.title(f"{method} - Feature Contributions for Cluster {i}", fontsize=14, fontweight="bold")
               plt.ylabel("Standardized Feature Value", fontsize=12)
               plt.xlabel("Features", fontsize=12)
               save_path_bar = os.path.join(self.config['save_path'], f'{method.lower()}_cluster_{i}_bar_plot.png')
               plt.tight_layout()
               plt.savefig(save_path_bar, dpi=300)
               print(f"Bar plot for {method} Cluster {i} saved to {save_path_bar}")

       # Radar Charts for Each Cluster
       num_vars = len(self.custom_labels)
       angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
       angles += angles[:1]  # Close the loop

       for method, cluster_centers in zip(['K-Means'], [cluster_centers_kmeans]):
           for i, cluster in enumerate(cluster_centers.values):
               values = cluster.tolist()
               values += values[:1]  # Close the loop for radar chart values

               plt.figure(figsize=(6, 6))
               ax = plt.subplot(111, polar=True)
               ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {i}")
               ax.fill(angles, values, alpha=0.25)

               plt.xticks(angles[:-1], self.custom_labels, color="grey", fontsize=10)
               plt.yticks(color="grey", fontsize=8)
               plt.title(f"{method} - Cluster {i}", size=15, color="darkblue", y=1.1)

               save_path_radar = os.path.join(self.config['save_path'], f'{method.lower()}_cluster_{i}_radar_chart.png')
               plt.tight_layout()
               plt.savefig(save_path_radar, dpi=300)
               print(f"Radar chart for {method} Cluster {i} saved to {save_path_radar}")

       # Parallel Coordinate Plots
       for method, cluster_centers in zip(['K-Means'], [cluster_centers_kmeans]):
           melted_data = cluster_centers.melt(ignore_index=False, var_name="Feature", value_name="Value").reset_index()
           melted_data = melted_data.rename(columns={"index": "Cluster"})

           plt.figure(figsize=(12, 6))
           sns.lineplot(data=melted_data, x="Feature", y="Value", hue="Cluster", marker="o", palette="tab10")
           plt.xticks(rotation=45, ha="right")
           plt.title(f"{method} - Parallel Coordinate Plot of Cluster Features", fontsize=16, fontweight="bold")
           plt.ylabel("Standardized Feature Value", fontsize=12)
           plt.xlabel("Features", fontsize=12)
           save_path_parallel = os.path.join(self.config['save_path'], f'{method.lower()}_parallel_coordinate_plot.png')
           plt.tight_layout()
           plt.savefig(save_path_parallel, dpi=300)
           print(f"Parallel coordinate plot for {method} saved to {save_path_parallel}")

   
    def test_features_statistical_significance(self, df, feature_columns):
       """
       Test the statistical significance of features across clusters using the Kruskal-Wallis test.
       """
       from scipy.stats import kruskal
   
       subtype_column = 'Cluster'  # Column for cluster labels
       results = []
   
       print("\n=== Statistical Analysis for Continuous Features ===")
       for feature in feature_columns:
           if feature in df.columns:
               # Perform Kruskal-Wallis test
               groups = [df[df[subtype_column] == cluster][feature].dropna() for cluster in df[subtype_column].unique()]
               stat, p = kruskal(*groups)
               results.append({'Feature': feature, 'Statistic': stat, 'p-value': p})
               print(f"Feature: {feature}, Statistic: {stat:.4f}, p-value: {p:.4f}")
           else:
               print(f"Feature '{feature}' not found in dataset.")
   
       # Convert results to DataFrame and save
       results_df = pd.DataFrame(results)
       results_df['Significant'] = results_df['p-value'] < 0.05   
       # Print significant features
       significant_features = results_df[results_df['Significant']]
       print("\nSignificant Features:")
       print(significant_features[['Feature', 'p-value']])


    def get_feature_columns(self, file_path):
       """
       Dynamically identify feature columns from the CSV file, excluding non-feature columns.

       Parameters:
       - file_path (str): Path to the CSV file.

       Returns:
       - feature_columns (list): List of feature column names.
       """
       # Load the dataset to get column names
       data = pd.read_csv(file_path)

       # Exclude columns like 'id', 'video_path', and 'label' (non-feature columns)
       non_feature_columns = {'ids', 'video_path', 'label'}
       feature_columns = [col for col in data.columns if col not in non_feature_columns]

       return feature_columns

    def process_and_plot_correlation(self):
       """
       Load data, compute correlation matrices, and save heatmaps for Pearson, Spearman, and Kendall correlations
       with custom feature titles, preserving the order defined in column_titles.
       
       Parameters:
       - file_path (str): Path to the CSV file containing the dataset.
       """
       # Define custom titles for the columns
       column_titles = {
           'avg_amplitude': 'Average amplitude (hypokinesia)',
           'mean_tapping_interval': 'Average cycle duration (bradykinesia)',
           'mean_percycle_max_speed': 'CMS (combined hypo- & bradykinesia)',
           'mean_percycle_avg_speed': 'CAS (combined hypo- & bradykinesia)',
           'amp_slope': 'Amplitude slope (sequence effect)',
           'ti_slope': 'Cycle duration slope (sequence effect)',
           'speed_slope': 'Speed slope (sequence effect)',
           'cov_tapping_interval': 'COV of cycle duration (hesitation-halts)',
           'cov_amp': 'COV of amplitude (hesitation-halts)',
           'cov_per_cycle_speed_maxima': 'COV of CMS (hesitation-halts)',
           'cov_per_cycle_speed_avg': 'COV of CAS (hesitation-halts)',
           'num_interruptions2': 'Number of interruptions (hesitation-halts)',
           'hjorth_mob':'Hjorth mobility (hesitation-halts)', 
           'max_freq_magnitude':'Maximum frequency magnitude (hesitation-halts)', 
       }
   
       # Identify the feature columns dynamically
       feature_columns = self.get_feature_columns(self.csv_save_path)
       remove_cols = ['mean_percycle_max_speed', 'mean_percycle_avg_speed']
       remove_cols = ['hjorth_mob', 'max_freq_magnitude']

       feature_columns = [col for col in feature_columns if col not in remove_cols]
        
       data = pd.read_csv(self.csv_save_path)[feature_columns]
   
       # Reorder to match column_titles dictionary
       ordered_feature_columns = [col for col in column_titles if col in data.columns]
       data = data[ordered_feature_columns]
       custom_labels = [column_titles[col] for col in ordered_feature_columns]
   
       # Compute correlation matrices
       correlation_matrices = {
           'pearson': data.corr(method='pearson'),
       }
   
       output_directory = os.path.dirname(self.csv_save_path)
   
       # Plot and save heatmaps
       for method, correlation_matrix in correlation_matrices.items():
           file_name = os.path.join(output_directory,'correlation+pca', f'{method}_correlation_heatmap.png')
           self.plot_and_save_heatmap(
               correlation_matrix,
               f'{method.capitalize()} Correlation Heatmap',
               custom_labels,  # Use custom labels for the heatmap
               file_name
           )

    def plot_and_save_heatmap(self, correlation_matrix, title, labels, file_name):
       """
       Plot and save a heatmap for the given correlation matrix with custom labels.
   
       Parameters:
       - correlation_matrix (DataFrame): Correlation matrix to visualize.
       - title (str): Title for the heatmap.
       - labels (list): Custom axis labels for the heatmap.
       - file_name (str): Path to save the heatmap image.
       """
       plt.figure(figsize=(12, 10))
       sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                   xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Correlation'},   vmin=-1,
        vmax=1)
       plt.title(title, fontsize=14, fontweight='bold')
       plt.tight_layout()
       plt.savefig(file_name, dpi=300)
       plt.show()
    
    def load_data(self, from_save):
        if from_save == False:    
            with open(self.config['annotated_pkl_path'], 'rb') as f:
                annotated_data = pickle.load(f)
            
            if self.config['include_control']:
                with open(self.config['control_path'], 'rb') as f:
                    control_data = pickle.load(f)
            
            combined_video_paths = annotated_data['video_path'] 
            combined_distances = annotated_data['distances'] 
            combined_keypoints = annotated_data['keypoints'] 
            combined_ids = annotated_data['id'] 
            combined_labels = annotated_data['label'] 
            combined_fps = annotated_data['fps'] 
                
                            
            if self.config['include_control']:
                combined_video_paths += control_data['video_path']
                combined_distances += control_data['distances']
                combined_keypoints += control_data['keypoints']
                combined_ids += control_data['id']
                combined_labels += control_data['label']
                combined_fps += control_data['fps']
            
        
        
            ############################################################# === Feature Extraction ===
            all_data = []
            if isinstance(self.config['video_cutting'], int):
                use_limited_duration = True
                video_duration = self.config['video_cutting']
            else:
                use_limited_duration = False
            
            for i in tqdm(range(len(combined_distances))):
                
                                fps = combined_fps[i]
                                if use_limited_duration:
                                    video_length = int(fps*video_duration)
                                    combined_distance=combined_distances[i][0:video_length]
                                    if len(combined_distance) <    video_length:
                                        continue
                                else:
                
                                    combined_distance=combined_distances[i]
                                
                                
                                video_length = len(combined_distance)                   
                                if  video_length>= 4*fps:
                                    print(i, video_length, combined_video_paths[i])
                                    features,  feat_name = self._extract_features(combined_distance,fps)
                                    row = [combined_ids[i], combined_video_paths[i], combined_labels[i]] + features
                                    all_data.append(row)
                
                    
            df = pd.DataFrame(all_data, columns= feat_name)
            df['label'] = df['label'].replace('control', 'Control')             
            filename = f"combined_features_{self.config['video_cutting']}_cutoff.csv"
            self.csv_save_path = os.path.join(self.config['save_path'], filename)
            df.to_csv(self.csv_save_path, index=False)

            self.data = df
        else: 
            filename = f"combined_features_{self.config['video_cutting']}_cutoff.csv"
            self.csv_save_path = os.path.join(self.config['save_path'], filename)
            self.data = pd.read_csv(self.csv_save_path)
            
    def plot_scatter_with_correlation(self, data, label_col, features, feature_titles, save_path, save_suffix='correlation'):
        """
        Plot scatter plots with regression line and show Spearman R and p-value for each feature.
        
        Parameters:
        - data (DataFrame): DataFrame containing the features and label
        - label_col (str): Column name for the UPDRS label
        - features (list): List of feature column names
        - feature_titles (dict): Mapping from feature column to title for the axis
        - save_path (str): Directory to save the plot
        - save_suffix (str): Suffix for file naming
        """
        sns.set(style="whitegrid")
        num_features = len(features)
        ncols = 3
        nrows = (num_features + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
        axes = axes.flatten()
    
        correlation_results = []
    
        for i, feature in enumerate(features):
            ax = axes[i]
            x = data[label_col]
            y = data[feature]
    
            # Spearman correlation
            r, p = spearmanr(x, y, nan_policy='omit')
            label = f"R = {r:.2f}, p = {p:.2g}"
    
            # Scatter plot with regression line
            sns.regplot(x=x, y=y, ax=ax, scatter_kws={'s': 40, 'alpha': 0.7}, line_kws={'color': 'black'})
            ax.set_title(feature_titles.get(feature, feature), fontsize=14, fontweight='bold')
            ax.set_xlabel('UPDRS score', fontsize=12)
            ax.set_ylabel(feature_titles.get(feature, feature), fontsize=12)
            ax.text(0.05, 0.95, label, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))
    
            correlation_results.append({
                'feature': feature,
                'spearman_r': round(r, 4),
                'p_value': round(p, 4)
            })
    
        for j in range(len(features), len(axes)):
            fig.delaxes(axes[j])
    
        plt.tight_layout()
        save_img = os.path.join(save_path, 'correlation+pca', f"correlation_scatter_{save_suffix}.png")
        save_csv = os.path.join(save_path,'correlation+pca', f"correlation_stats_{save_suffix}.csv")
        plt.savefig(save_img, dpi=300)
        pd.DataFrame(correlation_results).to_csv(save_csv, index=False)
        print(f"Saved plot: {save_img}")
        print(f"Saved stats: {save_csv}")
        
if __name__ == "__main__":


    test_type = 'ft'
    
    # Define base path
    base_path = r'//chansey.umcn.nl/diag/Tahereh/new/src1/my HC/id_based split'
    
    # Automatically choose save_path based on test_type
    save_path_lookup = {
        'ft': os.path.join(base_path, 'ft', 'new'),
        'la': os.path.join(base_path, 'leg', 'new'),
    }
    
    CONFIG = {
        'test_type': test_type,
        'save_path': save_path_lookup[test_type],
        'annotated_pkl_path': os.path.join(save_path_lookup[test_type], 'video_keypoints.pkl'),
        'control_path': os.path.join(save_path_lookup[test_type], 'control_keypoints.pkl'),
        'include_control': False,
        'video_cutting': 9.8,
    }
    
    feature_ext_ana = feature_ext_analysis(CONFIG)
    feature_ext_ana.load_data(from_save=True)
    #feature_ext_ana.boxplots_with_anova_brackets_new()    
    #feature_ext_ana.boxplots_with_anova_brackets()    

    feature_ext_ana.process_and_plot_correlation()
    feature_ext_ana.clustering_analysis()

