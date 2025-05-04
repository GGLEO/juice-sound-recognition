## ast test
import os
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import pandas as pd
from datetime import datetime
from torch.nn import CrossEntropyLoss

# ========= 模型與標籤 =========
model_path = "./ast_audio_model61"
label_names = ["watertrain", "appletrain", "teatrain"]
mp3_root = "/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicetest3/"
wav_root = "/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/juicetest3wav/"

# ========= MP3 轉換成 WAV =========
for folder in os.listdir(mp3_root):
    src_folder = os.path.join(mp3_root, folder)
    if not os.path.isdir(src_folder):
        continue  # 跳過非資料夾項目（例如 .DS_Store）
    dst_folder = os.path.join(wav_root, folder)
    os.makedirs(dst_folder, exist_ok=True)
    for file in os.listdir(src_folder):
        if file.endswith(".mp3"):
            mp3_path = os.path.join(src_folder, file)
            wav_path = os.path.join(dst_folder, file.replace(".mp3", ".wav"))
            try:
                sound = AudioSegment.from_mp3(mp3_path)
                sound.export(wav_path, format="wav")
            except Exception as e:
                print(f"❌ Failed to convert {mp3_path}: {e}")

folder_path = wav_root

# ========= 載入模型 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
extractor = AutoFeatureExtractor.from_pretrained(model_path)

# ========= 音訊預處理 =========
def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    return inputs

# ========= 單檔預測 =========
def predict(file_path):
    inputs = preprocess_audio(file_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([predicted_id]).to(device))
    return label_names[predicted_id], loss.item()

# ========= 從檔名提取真實標籤 =========
def extract_true_label(filename):
    label_map = {
        "appletest3": "appletrain",
        "watertest3": "watertrain",
        "teatest3": "teatrain"
    }

    label = filename.split('_')[0]
    true_label = label_map.get(label, "unknown")
    return true_label

# ========= 批次預測 + 混淆矩陣繪製 =========
def evaluate_folder(folder_path, accuracy_save_path, confusion_matrix_save_path):
    y_true = []
    y_pred = []
    file_losses = []
    print("\n🔍 開始預測...\n")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                try:
                    true_label = extract_true_label(file)
                    pred_label, loss = predict(file_path)
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    file_losses.append(loss)
                    print(f"{file:<25} | True: {true_label:<12} ➜ Pred: {pred_label} | Loss: {loss:.4f}")
                except Exception as e:
                    print(f"⚠️ 無法處理 {file}: {e}")
    
    # ➤ 準確率與平均損失
    acc = accuracy_score(y_true, y_pred)
    avg_loss = np.mean(file_losses)

    print(f"\n✅ 準確率：{acc * 100:.2f}%")
    print(f"📉 平均損失：{avg_loss:.4f}")


    # ➤ 儲存準確率到檔案
    with open(accuracy_save_path, 'w') as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n\n")
    
    # ➤ 混淆矩陣
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # ➤ 儲存混淆矩陣圖片
    plt.savefig(confusion_matrix_save_path)
    plt.show()
    plt.close()

    # ➤ 儲存 CSV 分類報告
    report_dict = classification_report(y_true, y_pred, labels=label_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    report_csv_path = accuracy_save_path.replace('.txt', '_report.csv')
    df_report.to_csv(report_csv_path)

    # ➤ 繪製 precision、recall、f1-score 柱狀圖
    metrics = ['precision', 'recall', 'f1-score']
    df_plot = df_report.loc[label_names, metrics]

    plt.figure(figsize=(8, 6))
    df_plot.plot(kind='bar', ylim=(0, 1), colormap='viridis')
    plt.title("Classification Report")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # ➤ 儲存圖
    report_bar_path = accuracy_save_path.replace('.txt', '_report_bar.png')
    plt.tight_layout()
    plt.savefig(report_bar_path)
    plt.show()
    plt.close()

# ========= 主程式 =========
if __name__ == "__main__":
    accuracy_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/newacc/accuracy_mel_ast_61_juicetest3.txt'
    confusion_matrix_save_path = '/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/newcfmt/confusion_matrix_mel_ast_61_juicetest3.png'

    evaluate_folder(folder_path, accuracy_save_path, confusion_matrix_save_path)
