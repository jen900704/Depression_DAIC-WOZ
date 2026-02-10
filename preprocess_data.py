import os, yaml, argparse, torchaudio, torch
import pandas as pd
from transformers import AutoConfig, Wav2Vec2Processor
from datasets import Dataset
from nested_array_catcher import nested_array_catcher

CHUNK_SECONDS = 15
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

def get_valid_path(filename):
    """暴力匹配：嘗試所有可能的檔名格式"""
    raw_base = str(filename).strip()
    # 移除可能存在的 .wav 方便統一處理
    clean_base = raw_base[:-4] if raw_base.lower().endswith('.wav') else raw_base
    
    # 建立嘗試清單
    attempts = [
        f"{clean_base}.wav",           # 原始格式: 484_19.wav
        f"{clean_base}",               # 無副檔名
    ]
    
    # 如果是 484_19 格式，嘗試 484_0_19
    if "_" in clean_base and "_0_" not in clean_base:
        p = clean_base.split("_", 1)
        attempts.append(f"{p[0]}_0_{p[1]}.wav")
        
    # 如果是 484_0_19 格式，嘗試 484_19
    if "_0_" in clean_base:
        attempts.append(clean_base.replace("_0_", "_") + ".wav")

    for att in attempts:
        full_path = os.path.join(AUDIO_ROOT, att)
        if os.path.exists(full_path):
            return full_path
    return None

def training_data(configuration):
    csv_path = configuration['csv_path']
    print(f"📖 讀取 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df['split'] = df['split'].astype(str).str.lower().str.strip()
    
    train_ds = Dataset.from_pandas(df[df['split'] == 'train'].reset_index(drop=True))
    test_ds = Dataset.from_pandas(df[df['split'] == 'test'].reset_index(drop=True))
    
    print(f"📊 初始筆數: 訓練 {len(train_ds)}, 測試 {len(test_ds)}")
    return train_ds, test_ds, "path", "class_2", ["non", "dep"], 2

def load_processor(configuration, label_list, num_labels):
    name = configuration['processor_name_or_path']
    config = AutoConfig.from_pretrained(name, num_labels=num_labels, cache_dir=configuration['cache_dir'])
    setattr(config, 'pooling_mode', configuration['pooling_mode'])
    processor = Wav2Vec2Processor.from_pretrained(name, cache_dir=configuration['cache_dir'])
    return config, processor, processor.feature_extractor.sampling_rate

def preprocess_data(configuration, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list):
    def label_to_id(val):
        s = str(val).lower().strip()
        return 1 if s in ['1', '1.0', 'dep'] else 0

    def preprocess_function(batch):
        new_batch = {"input_values": [], "labels": []}
        for path, label_val in zip(batch[input_column], batch[output_column]):
            full_path = get_valid_path(os.path.basename(path))
            if not full_path: continue
            
            try:
                speech_array, sr = torchaudio.load(full_path)
                resampler = torchaudio.transforms.Resample(sr, target_sampling_rate)
                speech = resampler(speech_array).squeeze()
                
                MAX_SAMPLES = target_sampling_rate * 25
                if len(speech) > MAX_SAMPLES: speech = speech[:MAX_SAMPLES]
                
                if len(speech) > target_sampling_rate:
                    new_batch["input_values"].append(speech.numpy())
                    new_batch["labels"].append(label_to_id(label_val))
            except: continue

        for i in range(len(new_batch['input_values'])):
            new_batch['input_values'][i] = nested_array_catcher(new_batch['input_values'][i])
        return new_batch

    features_path = configuration['output_dir'] + '/features'
    print(f"⚡ 特徵提取開始 (讀取目錄: {AUDIO_ROOT})")
    
    # 🚨 使用單線程確保 100% 穩定，並可觀察進度
    train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=2, num_proc=1, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, batch_size=2, num_proc=1, remove_columns=eval_dataset.column_names)
    
    train_dataset.save_to_disk(features_path + "/train_dataset")
    eval_dataset.save_to_disk(features_path + "/eval_dataset")
    
    print(f"✅ 完成！有效訓練樣本: {len(train_dataset)}")
    return train_dataset, eval_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path')
    args = parser.parse_args()
    with open(args.config) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    train_dataset, eval_dataset, input_column, output_column, label_list, num_labels = training_data(configuration)
    config, processor, target_sampling_rate = load_processor(configuration, label_list, num_labels)
    preprocess_data(configuration, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list)