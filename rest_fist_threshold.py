import pandas as pd
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy import signal
import seaborn as sns


# Goal: Classify between class 1 (rest) vs class 2 (fist) using thresholding

# Set random seed for reproducibility
#np.random.seed(42)
#random.seed(42)

# ------------------------------------- Helper Function 1 ---------------------------------------- #

##Function 1: load_emg_data
# This function randomly selects 30 out of 36 subjects from main folder
# Returns a single pandas DataFrame containing all raw EMG data (time, 8 channels, class) 
#   from all CSV files of the 30 randomly selected subjects, with added 'subject' and 'file' 
#   columns to track each row's origin
def load_emg_data(base_path, n_subjects=30):
 
    base_path = Path(base_path) # Load in base path where the parent folder is located
    
    # Randomly select 30 subjects from 1-36
    all_subjects = list(range(1, 37))
    selected_subjects = list(range(1, 31))
    print(f"Selected subjects: {sorted(selected_subjects)}")
    
    all_data = []
    
    for subject in selected_subjects:
        subject_path = base_path / str(subject)
        
        if subject_path.exists():
            # Get all CSV files in the subject folder
            csv_files = list(subject_path.glob("*.csv"))
            
            for csv_file in csv_files:
                try:
                    # Load the data
                    df = pd.read_csv(csv_file)
                    # Add subject identifier
                    df['subject'] = subject
                    df['file'] = csv_file.name
                    all_data.append(df)
                    print(f"Loaded: Subject {subject} - {csv_file.name}")
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return None
    
# ------------------------------------- Helper Function 2 ---------------------------------------- #
##Function 2: Extract features
# Extract windowed statistical features for class 1 (rest) and class 2 (fist)
# Returns a pandas DataFrame where each row represents features from one extracted window.
def extract_features(data, window_size=100, step_size=50):
 
    features_list = []   # Stores stats for rest and fist gestures

     # Separate the data into rest and fist dataframes
    df_rest = data[data['class'] == 1]  # Rest dataframe
    df_fist = data[data['class'] == 2]  # Fist dataframe

    # Process each class separately
    for class_label, class_df in [(1, df_rest), (2, df_fist)]:
        
        # Process each subject-file group
        grouped = class_df.groupby(['subject', 'file'])
        
        # Iterate through the dataframe by groups
        for (subject, file_name), group in grouped:
            # Get EMG channels (assuming columns like channel1, channel2, etc.)
            emg_columns = [col for col in group.columns if 'channel' in col.lower()]
            
            if not emg_columns:
                print(f"No channel columns found in {file_name}")
                continue
            
            # Process windows, overlap 50 samples
            for i in range(0, len(group) - window_size, step_size):
                window = group.iloc[i:i+window_size]
                
                window_class = class_label  # Class 1 (Rest) or 2 (Fist)

                # Calculate features for each channel
                mav_features = []
                rms_features = []
                var_features = []
                wl_features = []
                
                for channel in emg_columns:
                    channel_data = window[channel].values
                    
                    # Mean Absolute Value (MAV)
                    mav = np.mean(np.abs(channel_data))
                    mav_features.append(mav)
                    
                    # Root Mean Square (RMS)
                    rms = np.sqrt(np.mean(channel_data**2))
                    rms_features.append(rms)
                    
                    # Variance
                    var = np.var(channel_data)
                    var_features.append(var)

                    # Waveform Length
                    wl = np.sum(np.abs(np.diff(channel_data)))
                    wl_features.append(wl)
                
                # Store features
                features_list.append({
                    'subject': subject,
                    'file': file_name,
                    'class': window_class,
                    'mav_mean': np.mean(mav_features),  # Average MAV across channels
                    'mav_max': np.max(mav_features),    # Max MAV across channels
                    'rms_mean': np.mean(rms_features),  # Average RMS across channels
                    'rms_max': np.max(rms_features),    # Max RMS across channels
                    'var_mean': np.mean(var_features),  # Average variance
                    'wl_mean': np.mean(wl_features),    # Average waveform length across channels
                    'mav_all': mav_features,            # Keep all channel MAVs
                    'rms_all': rms_features,            # Keep all channel RMSs
                    'wl_all': wl_features               # Waveform length per channel

                })
        
    return pd.DataFrame(features_list)  # List that stores features from each window

# ------------------------------------- Helper Function 3 ---------------------------------------- #
## Function 3: Find the optimal threshold
#   Find optimal threshold using ROC curve analysis
def find_optimal_threshold(features_df, feature_name, plot=True):
  
    # Separate classes
    rest_values = features_df[features_df['class'] == 1][feature_name].values # All values in the rest class for the specific feature
    fist_values = features_df[features_df['class'] == 2][feature_name].values # All values in the fist class for the specific feature
    
    # ------------- Start ROC Curve Threshold Analysis ------------------------------
 
    # Prepare data for ROC curve, label 0 for rest and 1 for fist
    y_true = np.concatenate([np.zeros(len(rest_values)), np.ones(len(fist_values))])  # Arrays of label names [0, 0, ... , 0, 1, 1, ... 1]
    y_scores = np.concatenate([rest_values, fist_values]) # Array of feature values (mav in this case) [rest feature values, fist feature values]
    
    # Calculate ROC curve
    # Input parameter
    #  y_true: All the labels (0: rest, 1: fist)
    #  y_score: All the feature values
    # roc_curve function performs:
    #  Takes all values and tests it as a potential threshold value
    # Output values:
    #  fpr: How many false alarms at each threshold (Fraction of REST incorrectly called FIST)
    #  tpr: Fraction of FIST correctly identified
    #  thresholds: ALl tested thresholds from inf to -inf sorted from highest to lowest
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Area under ROC curve
    # AUC = 1.0 is perfect classifier, AUC = 0.5 is random guessing, AUC < 0.5 is worse than random guessing 
    # Tells how well the feature seperates the two classes
    roc_auc = auc(fpr, tpr) 
    
    # Find optimal threshold (Youden's J statistic)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate accuracy at optimal threshold
    predictions = (y_scores > optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, predictions) # Used for feature selection

    # ----------------------------------------------------------------------------------
    
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Distribution of features
        axes[0].hist(rest_values, alpha=0.5, label='Rest (Class 1)', bins=30, color='blue')
        axes[0].hist(fist_values, alpha=0.5, label='Fist (Class 2)', bins=30, color='red')
        axes[0].axvline(optimal_threshold, color='black', linestyle='--', 
                       label=f'Threshold: {optimal_threshold:.8f}')
        axes[0].set_xlabel(f'{feature_name}')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {feature_name}')
        axes[0].legend()
        
        # Plot 2: ROC Curve
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.5f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                       label=f'Optimal point')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        
        # Plot 3: Box plot
        data_for_box = pd.DataFrame({
            'Class': ['Rest']*len(rest_values) + ['Fist']*len(fist_values),
            'Value': np.concatenate([rest_values, fist_values])
        })
        axes[2].boxplot([rest_values, fist_values], labels=['Rest', 'Fist'])
        axes[2].axhline(optimal_threshold, color='red', linestyle='--', 
                       label=f'Threshold: {optimal_threshold:.8f}')
        axes[2].set_ylabel(f'{feature_name}')
        axes[2].set_title('Box Plot Comparison')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        'threshold': optimal_threshold,
        'auc': roc_auc,
        'accuracy': accuracy,
        'sensitivity': tpr[optimal_idx],
        'specificity': 1 - fpr[optimal_idx]
    }

# ------------------------------------- Helper Function 4 ---------------------------------------- #
# Function 4: Compare different EMG features to identify which provides the best classification accuracy.
#  Input: features_df where each row is the statiscal values of each window
#  Output: return results_df with all stats for each feature, and sort in terms of accuracy

def compare_features(features_df):
    """
    Compare different features to find the best one
    """
    feature_columns = ['mav_mean', 'mav_max', 'rms_mean', 'rms_max', 'var_mean', 'wl_mean']
    results = []
    
    for feature in feature_columns:
        result = find_optimal_threshold(features_df, feature, plot=False)
        result['feature'] = feature
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*60)
    print("Feature Comparison Results:")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, results_df['auc'], width, label='AUC', alpha=0.8)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Score')
    ax.set_title('Feature Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['feature'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ------------------------------------- Main Function Execution ---------------------------------------- #
if __name__ == "__main__":
    # Base parent path
    base_path = "/Users/beckercheng/Desktop/ECE 4905/emg-haptic-prototype/EMG Dataset/EMG_data_for_gestures-master_csv"
    
    # Load EMG dataset for processing
    print("Loading EMG data from 30 random subjects...") 
    data = load_emg_data(base_path, n_subjects=30)  
    
    if data is not None:
        print(f"\nTotal data shape: {data.shape}")
        print(f"Classes in data: {data['class'].unique()}")
        print(f"Class distribution:\n{data['class'].value_counts()}")
        
        # Use window size = 200
        window_size = 200
        step_size  = window_size // 2  # → 100
        
        print(f"\n{'='*60}")
        print(f"Extracting features using window size: {window_size}")
        print('='*60)

        # Extract features where each row stores the stats of one window    
        features = extract_features(data, window_size=window_size, step_size=step_size)
            
        if not features.empty:
            print(f"Extracted {len(features)} feature windows")
            print(f"Class distribution in features:\n{features['class'].value_counts()}")
                
            # Compare different features and return a dataframe with features that is sorted 
            # in terms of highest accruacy
            results_df = compare_features(features)
                
            # Show detailed results for best feature
            best_feature = results_df.iloc[0]['feature']  # Get the feature that produces the most accurate threshold
            print(f"\n{'='*60}")
            print(f"Detailed results for best feature: {best_feature}")
            print('='*60)

            # Determine the stats and threshold based on the best feature selected    
            optimal_result = find_optimal_threshold(features, best_feature, plot=True)
                
            print(f"\nOptimal Threshold: {optimal_result['threshold']:.8f}")
            print(f"Accuracy: {optimal_result['accuracy']:.2%}")
            print(f"Sensitivity: {optimal_result['sensitivity']:.2%}")
            print(f"Specificity: {optimal_result['specificity']:.2%}")
            print(f"AUC: {optimal_result['auc']:.3f}")
        
        # Save the best threshold for deployment
        print("\n" + "="*60)
        print("RECOMMENDED SETTINGS BASED ON DATASET:")
        print("="*60)
        print(f"1. Use window size: {window_size} samples")
        print(f"2. Use feature: {best_feature} averaged across channels")
        print(f"3. Threshold: {optimal_result['threshold']:.8f}")
        print(f"4. Classification rule: if MAV > {optimal_result['threshold']:.8f} → Fist, else → Rest")
        
    else:
        print("Failed to load data. Please check your file path and format.")

# # --------------------- Main Function Execution for multiple window sizes  -------------------------------- #
# if __name__ == "__main__":
#     # Base parent path
#     base_path = "/Users/beckercheng/Desktop/ECE 3906/emg-haptic-prototype/EMG_data_for_gestures-master_csv"
    
#     # Load EMG dataset for processing
#     print("Loading EMG data from 30 random subjects...") 
#     data = load_emg_data(base_path, n_subjects=30)  
    
#     if data is not None:
#         print(f"\nTotal data shape: {data.shape}")
#         print(f"Classes in data: {data['class'].unique()}")
#         print(f"Class distribution:\n{data['class'].value_counts()}")
        
#         # Extract features with different window sizes to test
#         window_sizes = [50, 100, 200]
        
#         for window_size in window_sizes:
#             print(f"\n{'='*60}")
#             print(f"Testing window size: {window_size}")
#             print('='*60)
            
#             features = extract_features(data, window_size=window_size, step_size=window_size//2)
            
#             if not features.empty:
#                 print(f"Extracted {len(features)} feature windows")
#                 print(f"Class distribution in features:\n{features['class'].value_counts()}")
                
#                 # Compare different features
#                 results_df = compare_features(features)
                
#                 # Show detailed results for the best feature
#                 best_feature = results_df.iloc[0]['feature']
#                 print(f"\n{'='*60}")
#                 print(f"Detailed results for best feature: {best_feature}")
#                 print('='*60)
                
#                 optimal_result = find_optimal_threshold(features, best_feature, plot=True)
                
#                 print(f"\nOptimal Threshold: {optimal_result['threshold']:.8f}")
#                 print(f"Accuracy: {optimal_result['accuracy']:.2%}")
#                 print(f"Sensitivity: {optimal_result['sensitivity']:.2%}")
#                 print(f"Specificity: {optimal_result['specificity']:.2%}")
#                 print(f"AUC: {optimal_result['auc']:.3f}")
        
#         # Save the best threshold for deployment
#         print("\n" + "="*60)
#         print("RECOMMENDED SETTINGS BASED ON DATASET:")
#         print("="*60)
#         print(f"1. Use window size: 100 samples")
#         print(f"2. Use feature: Mean Absolute Value (MAV) averaged across channels")
#         print(f"3. Threshold: {optimal_result['threshold']:.8f}")
#         print(f"4. Classification rule: if MAV > {optimal_result['threshold']:.8f} → Fist, else → Rest")
        
#     else:
#         print("Failed to load data. Please check your file path and format.")