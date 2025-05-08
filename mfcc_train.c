#mfcc train
import os
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from tensorflow.keras.regularizers import l2
import noisereduce as nr  # 导入 noisereduce 库

# Paths
folder_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug2/'
wav_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug2_wav/'
mfcc_image_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug2_mfcc_images/'
time_domain_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug2_time_images/'
frequency_domain_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug2_frequency_images/'
confusion_matrix_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/mfcc_crnn_juicemug2_confusion_matrices/'

# Ensure necessary folders exist
os.makedirs(wav_folder, exist_ok=True)
os.makedirs(mfcc_image_folder, exist_ok=True)
os.makedirs(time_domain_folder, exist_ok=True)
os.makedirs(frequency_domain_folder, exist_ok=True)
os.makedirs(confusion_matrix_folder, exist_ok=True)

def generate_time_domain_signal(frequency=5, duration=1, sample_rate=100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

# Step 2: Save time domain signal
def save_time_domain_signal(signal, sr, filename):
    t = np.arange(len(signal)) / sr  # Time axis in seconds
    plt.figure()
    plt.plot(t, signal)
    plt.title('Time Domain Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig(os.path.join(time_domain_folder, filename))
    plt.close()

def compute_frequency_domain(signal, sample_rate):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    return xf[:N//2], np.abs(yf[:N//2])  # 返回正频率部分

def save_frequency_domain_signal(xf, yf, filename):
    plt.figure()
    plt.plot(xf, yf)
    plt.title('Frequency Domain Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.savefig(os.path.join(frequency_domain_folder, filename))
    plt.close()


# Step 1: Convert audio files to WAV
def convert_audio_to_wav(audio_path, wav_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
        print(f"Converted {audio_path} to {wav_path}")
    except Exception as e:
        print(f"Error converting {audio_path} to WAV: {e}")

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

# {"watertrain": 0, "appletrain": 1, "coldtea": 2}
# Convert all audio files to WAV and generate MFCC images
wav_files = []
labels = []
label_mapping = {"watermug2": 0, "applemug2": 1, "teamug2": 2}

# 在处理每个音频文件时，加载音频并提取时域和频域信号
for label in label_mapping.keys():
    label_path = os.path.join(folder_path, label)
    
    if not os.path.exists(label_path):
        print(f"Folder for label '{label}' does not exist. Skipping.")
        continue

    wav_label_path = os.path.join(wav_folder, label)
    mfcc_label_path = os.path.join(mfcc_image_folder, label)
    
    os.makedirs(wav_label_path, exist_ok=True)
    os.makedirs(mfcc_label_path, exist_ok=True)

    for file in os.listdir(label_path):
        if file.endswith(('.mp3', '.m4a')):
            audio_path = os.path.join(label_path, file)
            wav_path = os.path.join(wav_label_path, file.replace('.mp3', '.wav').replace('.m4a', '.wav'))
            mfcc_image_path = os.path.join(mfcc_label_path, file.replace('.mp3', '.png').replace('.m4a', '.png'))
            
            # Convert to WAV
            convert_audio_to_wav(audio_path, wav_path)

            # Load the WAV file for further processing
            y, sr = librosa.load(wav_path, sr=None)  # Load with original sample rate

            # Apply noise reduction
            try:
                y_reduced = nr.reduce_noise(y=y, sr=sr)  # 使用 noisereduce 进行降噪
                print(f"Noise reduction applied to {file}")
                y = y_reduced  # 将降噪后的信号赋值给 y
            except Exception as e:
                print(f"Error applying noise reduction to {file}: {e}")

            # Save time domain signal
            save_time_domain_signal(y, sr, f'time_domain_{file.replace(".mp3", "").replace(".m4a", "")}.png')

            # Compute and save frequency domain signal
            xf, yf = compute_frequency_domain(y, sr)
            save_frequency_domain_signal(xf, yf, f'frequency_domain_{file.replace(".mp3", "").replace(".m4a", "")}.png')

            # Save MFCC image (existing function)
            save_mfcc_image(wav_path, mfcc_image_path)

            wav_files.append(wav_path)
            labels.append(label)

if not wav_files or not labels:
    raise ValueError("No WAV files or labels were generated. Check your dataset.")

# Step 3: Extract MFCC features for training
def extract_mfcc(file_path, sr=16000, n_mfcc=40, max_pad_len=400):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

features = [extract_mfcc(wav_file) for wav_file in wav_files]
features = np.array(features)

# One-hot encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# 確保所有類別都被正確處理
assert len(np.unique(encoded_labels)) == len(label_mapping), "Some classes are missing in the labels."

categorical_labels = to_categorical(encoded_labels, num_classes=len(label_mapping))

print("Categorical labels shape:", categorical_labels.shape)
