import pandas as pd
import os

# 設定根目錄
ROOT = "./experiment_sisman_scientific"

scenarios = {
    "Scenario A (Screening)": os.path.join(ROOT, "scenario_A_screening"),
    "Scenario B (Monitoring)": os.path.join(ROOT, "scenario_B_monitoring")
}

def get_ids(df):
    # 從路徑解析 ID: .../301_0_10.wav -> 301
    return set(df['path'].apply(lambda x: os.path.basename(str(x)).split('_')[0]))

print("========================================")
print("      📊 資料切分與病人 ID 檢查      ")
print("========================================")

for name, folder in scenarios.items():
    print(f"\n🔍 檢查: {name}")
    train_path = os.path.join(folder, "train.csv")
    test_path = os.path.join(folder, "test.csv")
    
    if not os.path.exists(train_path):
        print("   ❌ 找不到檔案")
        continue
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    ids_train = get_ids(df_train)
    ids_test = get_ids(df_test)
    
    overlap = ids_train.intersection(ids_test)
    
    print(f"   📂 訓練集人數: {len(ids_train)}")
    print(f"   📂 測試集人數: {len(ids_test)}")
    print(f"   🔗 重疊人數 (Leakage): {len(overlap)}")
    
    if len(overlap) == 0:
        print("   ✅ 狀態: Subject Independent (乾淨)")
        # 印出前幾位病人 ID 證明不一樣
        print(f"      Train 範例: {sorted(list(ids_train))[:5]}...")
        print(f"      Test  範例: {sorted(list(ids_test))[:5]}...")
    else:
        print("   ⚠️ 狀態: Subject Leakage (混雜)")
        print(f"      重疊範例: {sorted(list(overlap))[:5]}...")

print("\n========================================")