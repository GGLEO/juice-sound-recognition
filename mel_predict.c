#mel 測試集

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
test_mel_image_folder = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicemug_mel_images/'
# 設定儲存路徑
accuracy_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/mugaccuracy/accuracy_mel_cnn_mug_42.txt'
confusion_matrix_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/mugconfusionmatrix/confusion_matrix_mel_cnn_mug_42.png'

# 確保目標資料夾存在
os.makedirs(os.path.dirname(accuracy_save_path), exist_ok=True)
os.makedirs(os.path.dirname(confusion_matrix_save_path), exist_ok=True)
# Ensure necessary folders exist
os.makedirs(test_wav_folder, exist_ok=True)
os.makedirs(test_mfcc_image_folder, exist_ok=True)
os.makedirs(test_mel_image_folder, exist_ok=True)

# Load the best model
final_model_path = 'finaltrain_mel_cnn_water_model42.keras'
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
y_pred = best_model.predict(test_features)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(categorical_test_labels, axis=1)

cm_test = confusion_matrix(y_true_labels, y_pred_labels)
print(f"Confusion Matrix for Test Data:\n{cm_test}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Test Data')
plt.savefig(confusion_matrix_save_path)
plt.show()
plt.close()
print(f"Confusion matrix saved to {confusion_matrix_save_path}")

