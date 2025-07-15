import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import gc

# ======================
# 저장 경로
# ======================
save_folder = r'C:\Users\idle9\Desktop\Naver MYBOX\AISLab\실험\SCI\AWID3-MSA_Graph_Detection\Temporal-Dynamic-Graph-Snapshot\Feature_Auto_Extraction\Feature_list'
os.makedirs(save_folder, exist_ok=True)

# ======================
# 데이터셋 폴더와 폴더별 확장자
# ======================
dataset_folder = r'C:\Users\idle9\Desktop\Naver MYBOX\AISLab\Dataset\AWID3_Dataset_CSV\CSV'
folders = {
    '1.Deauth': '.csv',
    '2.Disas': '.csv',
    '3.(Re)Assoc': '.csv',
    '4.Rogue_AP': '.csv',
    '5.Krack': '.csv',
    '6.Kr00k': '.csv',
    '7.SSH': '.csv',
    '8.Botnet': '.csv',
    '9.Malware': '.csv',
    '10.SQL_Injection': '.csv',
    '11.SSDP': '.csv',
    '12.Evil_Twin': '.csv',
    '13.Website_spoofing': '.csv'
}

# ======================
# 레이블 매핑 딕셔너리
# ======================
label_mapping = {
    'Normal': 0,
    'Deauth': 1,
    'Disas': 2,
    '(Re)Assoc': 3,
    'RogueAP': 4,
    'Krack': 5,
    'Kr00k': 6,
    'kr00k': 6,
    'SSH': 7,
    'Botnet': 8,
    'Malware': 9,
    'SQL_Injection': 10,
    'SSDP': 11,
    'SDDP': 11,
    'Evil_Twin': 12,
    'Website_spoofing': 13
}

PROTOCOL_KEYWORDS = [
    "subtype", "flags", "opcode", "type", "reason", "duration", "capabilities",
    "handshake", "control", "protected", "retry", "pwrmgt", "timestamp"
]

def get_protocol_control_fields(df):
    protocol_fields = [col for col in df.columns if any(key in col for key in PROTOCOL_KEYWORDS)]
    return protocol_fields

def get_top_variance_fields(df, n=5, ignore_fields=None):
    ignore_fields = ignore_fields or []
    candidate_cols = [col for col in df.columns if col not in ignore_fields and pd.api.types.is_numeric_dtype(df[col])]
    if not candidate_cols:
        return []
    variances = df[candidate_cols].var().sort_values(ascending=False)
    return list(variances.head(n).index)

def get_top_mutual_info_fields(df, label_col="label", n=5, ignore_fields=None):
    ignore_fields = ignore_fields or []
    candidate_cols = [col for col in df.columns if col not in ignore_fields and pd.api.types.is_numeric_dtype(df[col])]
    if not candidate_cols or label_col not in df.columns:
        return []
    X = df[candidate_cols].fillna(0)
    y = df[label_col]
    mi = mutual_info_classif(X, y)
    top_idx = np.argsort(mi)[::-1][:n]
    return [candidate_cols[i] for i in top_idx]

def get_top_feature_importance_fields(df, label_col="label", n=5, ignore_fields=None):
    ignore_fields = ignore_fields or []
    candidate_cols = [col for col in df.columns if col not in ignore_fields and pd.api.types.is_numeric_dtype(df[col])]
    if not candidate_cols or label_col not in df.columns:
        return []
    X = df[candidate_cols].fillna(0)
    y = df[label_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n]
    return [candidate_cols[i] for i in top_idx]

def filter_to_control_fields(field_list):
    return [f for f in field_list if any(key in f for key in PROTOCOL_KEYWORDS)]

def auto_select_trigger_fields(df, method="protocol", top_n=5, label_col="label"):
    ignore_fields = ["label", "Label", "frame.time_epoch", "wlan.sa", "wlan.da", "tcp.srcport", "tcp.dstport", "ip.proto"]
    results = {}

    if method in ("protocol", "all"):
        proto_fields = get_protocol_control_fields(df)
        results["protocol"] = proto_fields[:top_n]

    if method in ("variance", "all"):
        var_fields = get_top_variance_fields(df, n=top_n, ignore_fields=ignore_fields)
        results["variance"] = var_fields

    if method in ("mutual_info", "all"):
        mi_fields = get_top_mutual_info_fields(df, label_col=label_col, n=top_n, ignore_fields=ignore_fields)
        results["mutual_info"] = mi_fields

    if method in ("feature_importance", "all"):
        fi_fields = get_top_feature_importance_fields(df, label_col=label_col, n=top_n, ignore_fields=ignore_fields)
        results["feature_importance"] = fi_fields

    # 각 방법별로 control 필드만 남기는 필터 (옵션)
    for key in ["variance", "mutual_info", "feature_importance"]:
        if key in results:
            results[key + "_control"] = filter_to_control_fields(results[key])
    
    if method == "all":
        return results
    else:
        if method in ["variance", "mutual_info", "feature_importance"]:
            return results.get(method + "_control", results.get(method, []))
        else:
            return results.get(method, [])

# ======================
# 폴더(공격)별 전체 파일을 합쳐서 trigger feature 자동 추출 및 파일로 저장
# ======================

folder_trigger_fields = {}

for folder, ext in folders.items():
    folder_path = os.path.join(dataset_folder, folder)
    dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(ext):
            file_path = os.path.join(folder_path, filename)
            print(f"[{folder}] Reading file: {filename}")
            # CSV 읽기 (메모리/타입 경고 방지)
            df = pd.read_csv(file_path, low_memory=False)
            # label 매핑/변환(숫자, NaN 제거)
            label_col = None
            if 'Label' in df.columns:
                df['Label'] = df['Label'].map(label_mapping)
                df = df[df['Label'].notna()]
                label_col = 'Label'
            elif 'label' in df.columns:
                df['label'] = df['label'].map(label_mapping)
                df = df[df['label'].notna()]
                label_col = 'label'
            dfs.append(df)
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        auto_result = auto_select_trigger_fields(all_df, method="all", top_n=5, label_col=label_col)
        folder_trigger_fields[folder] = auto_result
        print(f"\n[{folder}] Trigger feature candidates (자동 추출):")
        for method, field_list in auto_result.items():
            print(f"  {method}: {field_list}")

        # ▶ 저장 부분
        save_file = os.path.join(save_folder, f"{folder}_trigger_features.txt")
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(f"Trigger feature candidates for {folder}\n\n")
            for method, field_list in auto_result.items():
                f.write(f"[{method}]\n")
                for feat in field_list:
                    f.write(f"{feat}\n")
                f.write("\n")
        print(f"저장 완료: {save_file}")

        del all_df
        del dfs
        gc.collect()
    else:
        print(f"[{folder}] No CSV file found.")


# --------------------
# (option) 전체 데이터 누적 자동 추출도 가능
# --------------------
# data = pd.DataFrame()
# for folder, ext in folders.items():
#     folder_path = os.path.join(dataset_folder, folder)
#     for filename in os.listdir(folder_path):
#         if filename.endswith(ext):
#             file_path = os.path.join(folder_path, filename)
#             data1 = pd.read_csv(file_path)
#             data = pd.concat([data, data1])
# if not data.empty:
#     label_col = "Label" if "Label" in data.columns else "label" if "label" in data.columns else None
#     auto_result_all = auto_select_trigger_fields(data, method="all", top_n=5, label_col=label_col)
#     print("\n[전체 데이터] Trigger feature candidates (자동 추출):")
#     for method, field_list in auto_result_all.items():
#         print(f"  {method}: {field_list}")
