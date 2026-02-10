import pandas as pd
import os

# ================= 設定區 =================
# 直接指向單一的 CSV 檔案
SPLIT_CSV = "./experiment_sisman_scientific/scenario_B_monitoring/splits/dirty_random_split.csv"
# =========================================

def get_participant_id(filename):
    """
    從檔名解析病人 ID。
    假設檔名格式為: 301_AUDIO_0_1.wav -> ID: 301
    """
    base = os.path.basename(str(filename))
    return base.split('_')[0]

def check_overlap():
    print("🚀 正在檢查 Train/Test Speaker Overlap (資料洩漏驗證)...")
    
    if not os.path.exists(SPLIT_CSV):
        print(f"❌ 找不到檔案: {SPLIT_CSV}")
        return

    # 讀取完整數據
    df = pd.read_csv(SPLIT_CSV)
    
    # 根據 'split' 欄位切分
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    
    print(f"   - 總樣本數: {len(df)}")
    print(f"   - 訓練集樣本數: {len(df_train)}")
    print(f"   - 測試集樣本數: {len(df_test)}")
    
    # 提取 ID
    train_ids = set(df_train['path'].apply(get_participant_id))
    test_ids = set(df_test['path'].apply(get_participant_id))
    
    print(f"   - 訓練集人數 (Unique Speakers): {len(train_ids)}")
    print(f"   - 測試集人數 (Unique Speakers): {len(test_ids)}")
    
    # 計算交集
    overlap = train_ids.intersection(test_ids)
    overlap_count = len(overlap)
    
    # 計算重疊率
    leakage_rate = (overlap_count / len(test_ids)) * 100
    
    print("\n" + "="*40)
    print("📊 LEAKAGE ANALYSIS RESULT")
    print("="*40)
    print(f"🔴 Overlapping Speakers: {overlap_count}")
    print(f"🔴 Leakage Rate: {leakage_rate:.2f}%")
    print("="*40)
    
    if leakage_rate > 90:
        print("✅ 證實：Speaker Identity 幾乎完全洩漏。")
        print("   這解釋了為什麼模型可以達到 100% 準確率 (它在認人)。")
    else:
        print("⚠️ 重疊率較低，可能分割方式不同。")

if __name__ == "__main__":
    check_overlap()