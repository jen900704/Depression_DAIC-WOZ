import yaml, os, subprocess, datetime, shutil
import pandas as pd

train_script = "train.py"
base_config_file = "train_m5_v9.yaml" 

tasks = [
    # {"id": "A", ...},  <-- 註解掉 A，它已經完成了
    {"id": "B", "name": "Scenario_B_Monitoring", "path": "./experiment_sisman_scientific/scenario_B_monitoring"}
]

def run_task(task):
    print(f"\n[{datetime.datetime.now()}] 🚀 啟動任務: {task['name']}")
    
    # 1. 清理
    cache_path = os.path.join(os.getcwd(), "cache_huggingface")
    if os.path.exists(cache_path): shutil.rmtree(cache_path)
    
    # 2. 生成 CSV
    abs_data_path = os.path.abspath(task['path'])
    df_train = pd.read_csv(os.path.join(abs_data_path, "train.csv"))
    df_test = pd.read_csv(os.path.join(abs_data_path, "test.csv"))
    df_train['split'], df_test['split'] = 'train', 'test'
    df_merged = pd.concat([df_train, df_test], ignore_index=True)
    
    merged_file = os.path.join(abs_data_path, "merged_final.csv")
    df_merged.to_csv(merged_file, index=False)

    # 3. 強制生成全參數 Config (A100 續傳版)
    with open(base_config_file, 'r') as f: config = yaml.safe_load(f)
    
    config.update({
        'csv_path': merged_file,
        'output_dir': f"./content/models/scenario_{task['id']}",
        'cache_dir': cache_path,
        'input_column': 'path', 'output_column': 'class_2',
        'freeze_feature_extractor': True,
        'gradient_accumulation_steps': 1,
        
        # 🚀 A100 優化：加大 Batch Size 以提升速度
        'per_device_train_batch_size': 8, 
        'per_device_eval_batch_size': 8,
        
        'num_train_epochs': 20,
        'save_total_limit': 2,
        'logging_steps': 10, 'save_steps': 500, 'eval_steps': 500,
        'evaluation_strategy': 'steps', 
        
        # 🚀 關鍵修改：開啟 FP16 加速 + 啟動續傳模式
        'fp16': True, 
        'resume_from_checkpoint': True, 
        
        'report_to': 'none'
    })

    current_config = f"night_config_{task['id']}.yaml"
    with open(current_config, 'w') as f: yaml.dump(config, f)

    # 4. 執行
    log_name = f"log_{task['id']}.txt"
    print(f"🔥 執行中... 請監控 {log_name}")
    with open(log_name, "w") as outfile:
        subprocess.run(f"python -u {train_script} --config {current_config}", shell=True, stdout=outfile, stderr=subprocess.STDOUT)

for t in tasks:
    run_task(t)