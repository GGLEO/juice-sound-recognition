#crnn
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
folder_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1/'
wav_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_wav/'
mfcc_image_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_mfcc_images/'
time_domain_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_time_images/'
frequency_domain_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_frequency_images/'
confusion_matrix_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/mfcc_crnn_train1_confusion_matrices/'

# Ensure necessary folders exist
os.makedirs(wav_folder, exist_ok=True)
os.makedirs(mfcc_image_folder, exist_ok=True)
os.makedirs(time_domain_folder, exist_ok=True)
os.makedirs(frequency_domain_folder, exist_ok=True)
os.makedirs(confusion_matrix_folder, exist_ok=True)

# Step 1: Build the CRNN model
def build_crnn(input_shape, num_classes):
    # Enable mixed precision for Apple Silicon
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    inputs = tf.keras.Input(shape=input_shape)
    
    # 卷積層（特徵提取）
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Dropout(0.3)(x)  # 降低過擬合風險

    # 雙向遞歸層（時序建模）
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2))(x)

    # 全連接層
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # 避免過擬合
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # 輸出層（分類）
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # 建立模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define input shape and number of classes
input_shape = features.shape[1:]  # (40, 400)
num_classes = len(np.unique(encoded_labels))

# Build the model with the correct number of output classes
model = build_crnn(input_shape, num_classes)

# Callbacks for training
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model using all data (no validation split in this case)
history = model.fit(
    features,
    categorical_labels,
    batch_size=8,
    epochs=50,
    callbacks=[reduce_lr, early_stopping]
)

# Step 3: Save the trained model in .keras format
model.save('finaltrain_mfcc_crnn_water_model62.keras')
print("Model has been saved as 'finaltrain_mfcc_crnn_water_model62.keras'")
