import pandas as pd
import os

# 設定根目錄
ROOT = "./experiment_sisman_scientific"

scenarios = {
    "Scenario A (Screening)": os.path.join(ROOT, "scenario_A_screening"),
    "Scenario B (Monitoring)": os.path.join(ROOT, "scenario_B_monitoring")
}

print("========================================")
print("      🎧 音檔 (Segment) 重疊檢查      ")
print("========================================")

for name, folder in scenarios.items():
    print(f"\n🔍 檢查: {name}")
    train_path = os.path.join(folder, "train.csv")
    test_path = os.path.join(folder, "test.csv")
    
    if not os.path.exists(train_path):
        print("   ❌ 找不到 CSV 檔案")
        continue
        
    # 讀取 CSV
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 提取檔名 (去除路徑，只看檔名，例如 303_1.wav)
    # 使用 strip() 去除可能存在的空白
    files_train = set(df_train['path'].apply(lambda x: str(x).strip()))
    files_test = set(df_test['path'].apply(lambda x: str(x).strip()))
    
    # 計算交集 (重疊)
    overlap = files_train.intersection(files_test)
    
    print(f"   📂 Train 樣本數: {len(files_train)}")
    print(f"   📂 Test  樣本數: {len(files_test)}")
    print(f"   🔗 音檔重疊數: {len(overlap)}")
    
    if len(overlap) == 0:
        print("   ✅ 完美！沒有任何音檔被重複使用。")
        print("      (代表模型沒看過這些特定的句子，它是認出了『人的聲音』)")
    else:
        print(f"   ❌ 警告！發現 {len(overlap)} 個重複音檔！")
        print(f"      範例: {list(overlap)[:3]}")

print("\n========================================")