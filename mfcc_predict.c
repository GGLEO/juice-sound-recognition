#mfcc 測試集

import os
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
import noisereduce as nr

# Paths for test data
test_folder_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicetest3/'
test_wav_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicetest3_wav/'
test_mfcc_image_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicetest3_mfcc_images/'
# 設定儲存路徑
accuracy_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/new5_14/accuracy_mfcc_transformer_model62_juicetest3.txt'
confusion_matrix_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/new5_14/confusion_matrix_mfcc_transformer_model62_juicetest3.png'
# 定義分類報告儲存路徑
classification_report_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/new5_14/classification_report_mfcc_transformer_model62_juicetest3.txt'

# 確保目標資料夾存在
os.makedirs(os.path.dirname(accuracy_save_path), exist_ok=True)
os.makedirs(os.path.dirname(confusion_matrix_save_path), exist_ok=True)
# Ensure necessary folders exist
os.makedirs(test_wav_folder, exist_ok=True)
os.makedirs(test_mfcc_image_folder, exist_ok=True)
os.makedirs(os.path.dirname(classification_report_path), exist_ok=True)

# Load the best model
final_model_path = 'finaltrain_mfcc_transformer_model62.keras'
# best_model = tf.keras.models.load_model(final_model_path, custom_objects={'Adam': Adam})
best_model = tf.keras.models.load_model(final_model_path, compile=False)
best_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss='categorical_crossentropy',  # or your specific loss
                   metrics=['accuracy'])  # or your specific metrics

print("Best model loaded successfully.")


# Step 5: Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(test_features, categorical_test_labels, verbose=0)
# 儲存準確率
with open(accuracy_save_path, 'w') as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
print(f"Test accuracy saved to {accuracy_save_path}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Step 6: Confusion Matrix for Test Data
#{"watermug": 0, "applemug": 1, "teamug": 2}
y_pred = best_model.predict(test_features)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(categorical_test_labels, axis=1)

# 定義縮寫對應表
short_label_map = {"watertest3": "w", "appletest3": "a", "teatest3": "t"}

# 簡化標籤
class_names_abbr = [short_label_map[label] for label in encoder.classes_]
y_true_abbr = [short_label_map[encoder.classes_[i]] for i in y_true_labels]
y_pred_abbr = [short_label_map[encoder.classes_[i]] for i in y_pred_labels]

# 重新計算混淆矩陣（使用縮寫）
cm_abbr = confusion_matrix(y_true_abbr, y_pred_abbr, labels=class_names_abbr)

# 畫出改良後的混淆矩陣
plt.figure(figsize=(8, 6), dpi=300)  # 更高解析度

# 建議在圖示後加 layout
plt.tight_layout()

sns.heatmap(cm_abbr,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names_abbr,
            yticklabels=class_names_abbr,
            #linewidths=1,       # 加粗邊框線
            linecolor='black'   # 設定邊框顏色
           )

plt.xlabel('Identified Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
#plt.title('Confusion Matrix (w: water, a: apple, t: tea)', fontsize=16)

# ✅ 調整刻度字體大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ✅ 在圖下方加入完整說明
label_description = "w: watertest3   |   a: appletest3   |   t: teatest3"
plt.figtext(0.5, -0.05, label_description, wrap=True, horizontalalignment='center', fontsize=10)

# 儲存與顯示圖表
plt.tight_layout()
plt.savefig(confusion_matrix_save_path)
plt.show()
plt.close()
print(f"Confusion matrix saved to {confusion_matrix_save_path}")


# 取得分類報告（字串形式）
report_str = classification_report(y_true_abbr, y_pred_abbr, labels=class_names_abbr, digits=4)

# 印出到終端
print("Classification Report (Abbreviated Labels):")
print(report_str)

# 寫入到檔案
with open(classification_report_path, 'w') as f:
    f.write("Classification Report (Abbreviated Labels)\n")
    f.write("Label Mapping: w = watertest3, a = appletest3, t = teatest3\n\n")
    f.write(report_str)

print(f"Classification report saved to {classification_report_path}")
