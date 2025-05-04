## ast train
import os
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch
from pydub import AudioSegment
print(torch.backends.mps.is_available())  # True 表示支援
print(f"Using device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")

# ========== 登入 Hugging Face ==========
from huggingface_hub import login
login(token="")


# ========== 資料夾與標籤設定 ==========
label_mapping = {"waterglass340390": 0, "appleglass340390": 1, "teaglass340390": 2}
mp3_root = "/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/new/glass340390/"
wav_root = "/Users/leechienju/Library/Mobile Documents/com~apple~CloudDocs/Documents/juice/new/glass340390wav2/"

# MP3 轉換成 WAV
# MP3 轉換成 WAV，僅保留 5~10 秒區段
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
                start_ms = 5 * 1000  # 5 秒
                end_ms = 10 * 1000   # 10 秒
                if len(sound) >= end_ms:
                    sound = sound[start_ms:end_ms]
                    sound.export(wav_path, format="wav")
                else:
                    print(f"⚠️ 音檔太短，已略過：{mp3_path}")
            except Exception as e:
                print(f"❌ Failed to convert {mp3_path}: {e}")


folder_path = wav_root

# ========== 載入 AST 特徵擷取器與模型 ==========
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model_ast = AutoModelForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(label_mapping),
    ignore_mismatched_sizes=True
)

# ========== 音檔轉 AST 輸入 ==========
def extract_input_tensor(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # AST 模型需要 16kHz 音檔
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs['input_values'][0]
    except Exception as e:
        print(f"❌ 無法讀取 {file_path}：{e}")
        return None

# ========== 收集資料 ==========
wav_files = []
labels = []

for label_name, label_id in label_mapping.items():
    label_dir = os.path.join(folder_path, label_name)
    if not os.path.exists(label_dir):
        print(f"❗ 資料夾不存在: {label_dir}")
        continue

    for fname in os.listdir(label_dir):
        if fname.endswith(".mp3") or fname.endswith(".wav"):
            fpath = os.path.join(label_dir, fname)
            wav_files.append(fpath)
            labels.append(label_id)

# 過濾掉讀取失敗的檔案
X_list = []
labels_clean = []
for f, l in zip(wav_files, labels):
    tensor = extract_input_tensor(f)
    if tensor is not None:
        X_list.append(tensor)  # shape (num_files, seq_len)
        labels_clean.append(l)

X_tensor = torch.stack(X_list)  # PyTorch tensor
y_tensor = torch.tensor(labels_clean)

# 分割訓練與驗證集
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor.numpy(), test_size=0.2, stratify=y_tensor.numpy(), random_state=42)

# 再轉換回 PyTorch tensor
X_train_tensor = torch.tensor(X_train)
X_val_tensor = torch.tensor(X_val)
y_train_tensor = torch.tensor(y_train)
y_val_tensor = torch.tensor(y_val)

# 自定義 Dataset 類別
class AudioDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {'input_values': self.encodings[idx], 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

train_dataset = AudioDataset(X_train_tensor, y_train_tensor)
val_dataset = AudioDataset(X_val_tensor, y_val_tensor)
# train_dataset = AudioDataset(X_tensor, y_tensor)

# 訓練參數設定
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=50,
    load_best_model_at_end=True,
    no_cuda=True  # ✅ 強制使用 CPU，避免使用 MPS（Metal）
    # use_cpu=True,  # ✅ 新寫法，取代 no_cuda
    # use_mps_device=True,  # 或者開啟這行，用 Mac GPU
)



trainer = Trainer(
    model=model_ast,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 開始訓練
trainer.train()

# 儲存模型
output_dir = "./ast_audio_model_glass340390wav2_2"
model_ast.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
print(f"✅ 模型已儲存至 {output_dir}")
