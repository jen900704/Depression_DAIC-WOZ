import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ================= 設定區 =================
source_csv = "all_m5_clean_v_final.csv"  
output_base = "./experiment_sisman_scientific"
# =========================================

print("🚀 [V5] 開始建構「科學隨機抽樣」對照實驗...")

# 1. 讀取與前置處理
if not os.path.exists(source_csv):
    print(f"❌ 找不到來源檔案: {source_csv}"); exit()

df = pd.read_csv(source_csv, sep=None, engine='python')

# 確保路徑與ID
path_col = next((c for c in df.columns if c.lower() in ['path', 'file_path', 'filename']), None)
if not path_col: print("❌ 找不到路徑欄位"); exit()
df['participant_id'] = df[path_col].apply(lambda x: os.path.basename(str(x)).split('_')[0])

# 確保 Label
label_col = next((c for c in df.columns if c.lower() in ['class_2', 'label', 'target']), None)
if not label_col: print("❌ 找不到 Label"); exit()
df = df.dropna(subset=[label_col])

print(f"👥 總人數: {df['participant_id'].nunique()}")
print(f"📊 總資料: {len(df)}")

# ==========================================
# 🧠 核心邏輯：雙盲隨機抽樣
# ==========================================

# 第一步：選出 20% 的人當作「測試對象」(Test Subjects)
# 這些人是我們這次實驗的主角，我們要觀察「有沒有洩漏」對預測這群人有多大影響
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_people_idx, test_people_idx = next(gss.split(df, groups=df['participant_id']))

df_background = df.iloc[train_people_idx].copy() # 背景路人 (永遠在訓練集)
df_test_subjects = df.iloc[test_people_idx].copy() # 測試對象 (我們要對他們做手腳)

print(f"\n🔒 鎖定測試對象: {df_test_subjects['participant_id'].nunique()} 人")

# 第二步：針對測試對象，進行「隨機打散切分」 (Random Shuffle Split)
# 這避免了「前半段/後半段」的時間偏差。
# 我們將每位測試對象的錄音檔隨機分成 50% 歷史資料 (Leakage) 和 50% 當下資料 (Test)
split_leak_history, split_exam_target = train_test_split(
    df_test_subjects, 
    test_size=0.5, 
    random_state=42, 
    stratify=df_test_subjects['participant_id'] # 關鍵：確保每個人都被均勻切分
)

print(f"   📂 歷史資料 (Leak Source): {len(split_leak_history)} 筆 (隨機抽樣)")
print(f"   📂 當下資料 (Fixed Exam):  {len(split_exam_target)} 筆 (隨機抽樣)")

# ==========================================
# 🧪 實驗 A: 初診篩檢 (Scenario: First Visit)
# ==========================================
print("\n[實驗 A] 初診篩檢 (Strict No Leakage)...")
save_dir_A = os.path.join(output_base, "scenario_A_screening")
os.makedirs(save_dir_A, exist_ok=True)

# 訓練集 = 只有背景路人
train_A = df_background.copy()
# 測試集 = 當下資料
test_A = split_exam_target.copy()

# 存檔
path_c, label_c = path_col, label_col
train_A.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_A, "train.csv"), index=False)
test_A.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_A, "test.csv"), index=False)

print(f"   👉 Train: {len(train_A)} | Test: {len(test_A)}")
print(f"   👉 邏輯: 模型完全沒聽過這 {df_test_subjects['participant_id'].nunique()} 位病人的聲音。")


# ==========================================
# 🧪 實驗 B: 長期監測 (Scenario: Longitudinal Monitoring)
# ==========================================
print("\n[實驗 B] 長期監測 (With Historical Leakage)...")
save_dir_B = os.path.join(output_base, "scenario_B_monitoring")
os.makedirs(save_dir_B, exist_ok=True)

# 訓練集 = 背景路人 + 測試對象的歷史資料 (Leakage)
train_B = pd.concat([df_background, split_leak_history])
# 測試集 = 當下資料 (絕對跟 A 一模一樣)
test_B = split_exam_target.copy()

# 存檔
train_B.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_B, "train.csv"), index=False)
test_B.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_B, "test.csv"), index=False)

print(f"   👉 Train: {len(train_B)} | Test: {len(test_B)}")
print(f"   👉 邏輯: 模型在訓練時聽過了這 {df_test_subjects['participant_id'].nunique()} 位病人『其他的』錄音檔。")
print(f"   ✨ 科學性: Test Set 完全固定，唯一的變因是 Training Set 是否包含該人的歷史資訊。")

print("\n✅ 科學對照數據生成完畢！")
print(f"資料夾: {output_base}")