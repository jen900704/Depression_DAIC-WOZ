import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2Processor, 
    Trainer, 
    TrainingArguments,
    Wav2Vec2Config,
    PreTrainedModel
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, mean_squared_error
from safetensors.torch import load_file

# ================= 🎯 設定區 (已自動填入) =================
MODEL_PATH = "./content/models/scenario_A/checkpoint-6100"
CSV_PATH = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"
# ==========================================================

# 1. 定義模型結構 (保持與訓練時一致)
class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()
    
    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean": return torch.mean(hidden_states, dim=1)
        elif mode == "sum": return torch.sum(hidden_states, dim=1)
        elif mode == "max": return torch.max(hidden_states, dim=1)[0]
        else: raise Exception("Unknown pooling mode")

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits

# 2. 定義 Dataset (V9版防呆邏輯)
class DAICDataset(Dataset):
    def __init__(self, csv_path, audio_root, processor, split='test'):
        self.df = pd.read_csv(csv_path)
        # 這裡我們不做 filter，直接假設 CSV 就是我們要的測試集
        self.audio_root = audio_root
        self.processor = processor
        
    def __len__(self):
        return len(self.df)

    def get_valid_path(self, filename):
        raw_base = str(filename).strip()
        clean_base = raw_base[:-4] if raw_base.lower().endswith('.wav') else raw_base
        attempts = [f"{clean_base}.wav", f"{clean_base}"]
        if "_" in clean_base and "_0_" not in clean_base:
            p = clean_base.split("_", 1)
            attempts.append(f"{p[0]}_0_{p[1]}.wav")
        if "_0_" in clean_base:
            attempts.append(clean_base.replace("_0_", "_") + ".wav")
        for att in attempts:
            full_path = os.path.join(self.audio_root, att)
            if os.path.exists(full_path): return full_path
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.get_valid_path(row['path'])
        
        # 標籤欄位判定
        label_col = 'class_2' if 'class_2' in row else 'label'
        label = int(row[label_col])

        if not audio_path:
            return {"input_values": torch.zeros(16000), "labels": label}

        speech, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)
        speech = speech.squeeze()

        MAX_LENGTH = 16000 * 10 
        
        inputs = self.processor(
            speech, 
            sampling_rate=16000, 
            max_length=MAX_LENGTH,
            truncation=True, 
            padding="max_length", 
            return_tensors="pt",
            return_attention_mask=True
        )

        input_tensor = inputs['input_values'][0]
        if 'attention_mask' in inputs:
            mask_tensor = inputs['attention_mask'][0]
        else:
            mask_tensor = torch.ones_like(input_tensor, dtype=torch.long)

        return {
            "input_values": input_tensor,
            "attention_mask": mask_tensor,
            "labels": label
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def run_scenario_A_eval():
    print(f"🔬 啟動 Scenario A (Clean) 科學評估: {MODEL_PATH}")
    
    config = Wav2Vec2Config.from_pretrained(MODEL_PATH)
    if not hasattr(config, 'pooling_mode'): config.pooling_mode = 'mean'
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForSpeechClassification.from_pretrained(MODEL_PATH, config=config)
    
    # 強制載入權重
    safetensors_path = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print("✅ 權重載入成功")
    
    test_dataset = DAICDataset(CSV_PATH, AUDIO_ROOT, processor, split='test')
    print(f"📊 Scenario A 測試集樣本數: {len(test_dataset)}")

    training_args = TrainingArguments(
        output_dir="./scenario_A_eval_output",
        per_device_eval_batch_size=8,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    print("⚡ 開始執行預測...")
    output = trainer.predict(test_dataset)
    
    preds = np.argmax(output.predictions, axis=1)
    labels = output.label_ids
    
    # 計算所有指標
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rmse = mean_squared_error(labels, preds, squared=False)
    
    probs = torch.nn.functional.softmax(torch.tensor(output.predictions), dim=1)[:, 1].numpy()
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.5 # 如果只有一類

    print("\n" + "="*50)
    print("📊 SCENARIO A (SUBJECT-INDEPENDENT) REPORT")
    print("="*50)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 35)
    print(f"{'Accuracy':<20} | {acc:.4f}")
    print(f"{'Precision':<20} | {precision:.4f}")
    print(f"{'Recall':<20} | {recall:.4f}")
    print(f"{'F1-Score':<20} | {f1:.4f}")
    print(f"{'AUC (ROC)':<20} | {auc:.4f}")
    print(f"{'Specificity':<20} | {specificity:.4f}")
    print(f"{'RMSE':<20} | {rmse:.4f}")
    print("-" * 35)
    print("🧮 Confusion Matrix:")
    print(cm)
    print(f"(TN={tn}, FP={fp}, FN={fn}, TP={tp})")
    print("="*50)

if __name__ == "__main__":
    run_scenario_A_eval()