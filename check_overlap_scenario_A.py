import pandas as pd
import os

# ================= 設定區 (請修改這裡！) =================
# 指向 Scenario A 的 train 和 test CSV
# 假設你有分開的檔案，或者一個大檔案
TRAIN_CSV = "./experiment_sisman_scientific/scenario_A/splits/train.csv" # <--- 修改
TEST_CSV = "./experiment_sisman_scientific/scenario_A/splits/test.csv"   # <--- 修改
# ==========================================================

def get_participant_id(filename):
    base = os.path.basename(str(filename))
    return base.split('_')[0]

def check_overlap_A():
    print("🚀 正在檢查 Scenario A Speaker Overlap (泛化驗證)...")
    
    try:
        df_train = pd.read_csv(TRAIN_CSV)
        df_test = pd.read_csv(TEST_CSV)
    except Exception as e:
        print(f"❌ 讀取檔案失敗: {e}")
        return

    # 提取 ID
    train_ids = set(df_train['path'].apply(get_participant_id))
    test_ids = set(df_test['path'].apply(get_participant_id))
    
    print(f"   - 訓練集人數: {len(train_ids)}")
    print(f"   - 測試集人數: {len(test_ids)}")
    
    # 計算交集
    overlap = train_ids.intersection(test_ids)
    
    print("\n" + "="*40)
    print("📊 SCENARIO A LEAKAGE ANALYSIS")
    print("="*40)
    print(f"🔴 Overlapping Speakers: {len(overlap)}")
    
    if len(overlap) == 0:
        print("✅ 完美！Overlap 為 0。")
        print("   這證明了這是嚴格的 Subject-Independent Split。")
        print("   測試集裡的每一個病人，模型在訓練時都沒見過。")
    else:
        print(f"⚠️ 警告！發現 {len(overlap)} 個重疊病人：{overlap}")
    print("="*40)

if __name__ == "__main__":
    check_overlap_A()