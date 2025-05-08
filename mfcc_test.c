#mfcc test

import os
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
import noisereduce as nr

# Paths for test data
test_folder_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug/'
test_wav_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug_wav/'
test_mfcc_image_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug_mfcc_images/'
# 設定儲存路徑
accuracy_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/mugaccuracy3/accuracy_crnn_mug_62.txt'
confusion_matrix_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/mugconfusionmatrix3/confusion_matrix_crnn_mug_62.png'

# 確保目標資料夾存在
os.makedirs(os.path.dirname(accuracy_save_path), exist_ok=True)
os.makedirs(os.path.dirname(confusion_matrix_save_path), exist_ok=True)
# Ensure necessary folders exist
os.makedirs(test_wav_folder, exist_ok=True)
os.makedirs(test_mfcc_image_folder, exist_ok=True)

# Step 1: Convert audio files in test dataset to WAV
def convert_test_audio_to_wav(audio_path, wav_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
        print(f"Converted {audio_path} to {wav_path}")
    except Exception as e:
        print(f"Error converting {audio_path} to WAV: {e}")

# Step 2: Extract MFCC and save feature images
def save_mfcc_image(file_path, save_path, sr=16000, n_mfcc=40, max_pad_len=400):
    # Load audio
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    
    # Apply noise reduction
    try:
        y_reduced = nr.reduce_noise(y=y, sr=sr)
        print(f"Noise reduction applied to {file_path} for MFCC image")
        y = y_reduced  # Use the noise-reduced signal
    except Exception as e:
        print(f"Error applying noise reduction to {file_path} for MFCC image: {e}")
        
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Padding or trimming MFCC
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    # Save MFCC image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved MFCC image for {file_path} to {save_path}")

# Step 3: Load and process test data
test_wav_files = []
test_labels = []
label_mapping = {"watermug": 0, "applemug": 1, "teamug": 2}

for label in label_mapping.keys():
    label_path = os.path.join(test_folder_path, label)
    print(f"Checking folder: {label_path}")
    
    if not os.path.exists(label_path):
        print(f"Folder for label '{label}' does not exist. Skipping.")
        continue

    # Create directories for WAV and MFCC images
    wav_label_path = os.path.join(test_wav_folder, label)
    mfcc_label_path = os.path.join(test_mfcc_image_folder, label)
    
    os.makedirs(wav_label_path, exist_ok=True)
    os.makedirs(mfcc_label_path, exist_ok=True)

    for file in os.listdir(label_path):
        if file.endswith(('.mp3', '.m4a')):
            audio_path = os.path.join(label_path, file)
            wav_path = os.path.join(wav_label_path, file.replace('.mp3', '.wav').replace('.m4a', '.wav'))
            mfcc_image_path = os.path.join(mfcc_label_path, file.replace('.mp3', '.png').replace('.m4a', '.png'))
            
            # Convert to WAV and save MFCC image
            convert_test_audio_to_wav(audio_path, wav_path)

            # Load the WAV file for further processing
            y, sr = librosa.load(wav_path, sr=None)  # Load with original sample rate

            # Apply noise reduction
            try:
                y_reduced = nr.reduce_noise(y=y, sr=sr)  # 使用 noisereduce 进行降噪
                print(f"Noise reduction applied to {file}")
                y = y_reduced  # 将降噪后的信号赋值给 y
            except Exception as e:
                print(f"Error applying noise reduction to {file}: {e}")

            # Save MFCC image (existing function)
            save_mfcc_image(wav_path, mfcc_image_path)
            
            test_wav_files.append(wav_path)  # Append WAV file path
            test_labels.append(label)          # Append label name

if not test_wav_files or not test_labels:
    raise ValueError("No WAV files or labels were generated from the test dataset. Check your dataset.")

# Step 4: Extract MFCC features for testing
def extract_mfcc(file_path, sr=16000, n_mfcc=40, max_pad_len=400):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc

test_features = [extract_mfcc(wav_file) for wav_file in test_wav_files]
test_features = np.array(test_features)

# One-hot encode test labels
encoder = LabelEncoder()
encoder.fit(list(label_mapping.keys()))  # Fit the encoder on the label mapping keys
encoded_test_labels = encoder.transform(test_labels)  # Transform using the correct labels (strings)
categorical_test_labels = to_categorical(encoded_test_labels)
print(f"complete training")
