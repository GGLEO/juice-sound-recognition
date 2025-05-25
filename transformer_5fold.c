#transformer
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd

# Paths
folder_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1/'
wav_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_wav/'
mfcc_image_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_mfcc_images/'
time_domain_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_time_images/'
frequency_domain_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/train1_frequency_images/'

# 定義儲存結果的資料夾
results_folder = '/Users/leechienju/new5_23/transformer_5fold/22/mel'
os.makedirs(results_folder, exist_ok=True)
os.makedirs(os.path.join(results_folder, 'confusion_matrices'), exist_ok=True)
os.makedirs(os.path.join(results_folder, 'roc_curves'), exist_ok=True)
os.makedirs(os.path.join(results_folder, 'classification_reports'), exist_ok=True)
os.makedirs(os.path.join(results_folder, 'models'), exist_ok=True)
best_model_path = "./best_mel_transformer_model_5fold22.keras"
short_label_map = {"watertrain": "w", "appletrain": "a", "coldtea": "t"}

# Ensure necessary folders exist
os.makedirs(wav_folder, exist_ok=True)
os.makedirs(mfcc_image_folder, exist_ok=True)
os.makedirs(time_domain_folder, exist_ok=True)
os.makedirs(frequency_domain_folder, exist_ok=True)

# Step 1: Build the transformer model
def build_transformer(input_shape, num_classes):
    tf.keras.mixed_precision.set_global_policy("float32")
    inputs = tf.keras.Input(shape=input_shape)
    pos_encoding = tf.keras.layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1], dtype=tf.float32)(tf.range(input_shape[0]))
    x = inputs + pos_encoding
    
    for _ in range(3):  # Transformer blocks
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_shape[1])
        ])
        x = tf.keras.layers.Add()([x, ffn(x)])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)   
    # 建立模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

X = features
y = encoded_labels

# Define input shape and number of classes
input_shape = features.shape[1:]  # (40, 400)
num_classes = len(np.unique(encoded_labels))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 儲存每一折的結果
fold_metrics = []
conf_matrices = []
roc_curves = []

best_f1 = 0
best_model = None
best_fold = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Processing fold {fold}...")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    
    model = build_transformer(input_shape, num_classes)
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=8,
        callbacks=[reduce_lr, early_stopping],
        verbose=0
    )
    
    # 預測
    y_pred_proba = model.predict(X_val)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 分類報告
    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_folder, 'classification_reports', f'fold_{fold}_report.csv'))
    
    # 混淆矩陣
    cm = confusion_matrix(y_val, y_pred)
    conf_matrices.append(cm)
    
    # 繪製混淆矩陣
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_label_map.values(),
                yticklabels=short_label_map.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(os.path.join(results_folder, 'confusion_matrices', f'fold_{fold}_confusion_matrix.png'))
    plt.close()
    
    # ROC 曲線
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_cat[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 繪製 ROC 曲線
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_folder, 'roc_curves', f'fold_{fold}_roc_curve.png'))
    plt.close()
    
    # 儲存模型
    model.save(os.path.join(results_folder, 'models', f'fold_{fold}_model.keras'))
    
    # 計算 F1-score
    f1 = report['weighted avg']['f1-score']
    fold_metrics.append({
        'fold': fold,
        'f1_score': f1,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall']
    })
    
    # 更新最佳模型
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_fold = fold

# 計算平均性能指標
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(os.path.join(results_folder, 'average_metrics.csv'), index=False)

# 計算平均混淆矩陣
avg_cm = np.mean(conf_matrices, axis=0)

# 繪製平均混淆矩陣
plt.figure(figsize=(6, 4))
sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=short_label_map.values(),
            yticklabels=short_label_map.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Average Confusion Matrix')
plt.savefig(os.path.join(results_folder, 'confusion_matrices', 'average_confusion_matrix.png'))
plt.close()

# 繪製平均 ROC 曲線
# 注意：這裡僅為示意，實際計算平均 ROC 曲線需對齊所有折的 fpr
# 並在相同的 fpr 上插值 tpr，然後計算平均 tpr
best_model.save(best_model_path)
print(f"Best model from fold {best_fold} saved to {best_model_path}")
