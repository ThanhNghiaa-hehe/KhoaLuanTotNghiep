import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# (Sửa lại giống với đường dẫn bạn đã thiết lập trong run_all_models.py)
# Ví dụ: RESULTS_DIR = "/content/drive/MyDrive/KetQuaKhoaLuan"
# =============================================================================
RESULTS_DIR = "./results"

OUTPUT_DIR = os.path.join(RESULTS_DIR, "ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]

def load_predictions():
    """Tải tất cả các logits từ các mô hình đã được huấn luyện."""
    search_path = os.path.join(RESULTS_DIR, "*", "")
    model_dirs = glob.glob(search_path)
    model_logits = {}
    y_true = None
    
    for d in model_dirs:
        logits_path = os.path.join(d, "test_logits.npy")
        y_true_path = os.path.join(d, "test_y_true.npy")
        
        if os.path.exists(logits_path) and os.path.exists(y_true_path):
            model_name = os.path.basename(os.path.normpath(d))
            # Bỏ qua các thư mục không phải transformer (như ensemble, baseline)
            if model_name in ["ensemble", "baseline_svm", "baseline_lr", "confusion_analysis", "attention_viz"]:
                continue
                
            logits = np.load(logits_path)
            model_logits[model_name] = logits
            
            if y_true is None:
                y_true = np.load(y_true_path)
                
    return model_logits, y_true

def evaluate_and_save(y_true, y_pred, name):
    report_str = classification_report(y_true, y_pred, labels=list(range(7)), target_names=LABEL_NAMES, digits=4)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n[Ensemble: {name}] Macro-F1: {macro_f1:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, f"report_{name.replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Ensemble Method: {name}\n\n")
        f.write(report_str)
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Confusion Matrix\nEnsemble: {name}", fontsize=13, fontweight="bold")
    plt.ylabel("Nhãn thực tế")
    plt.xlabel("Nhãn dự đoán")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{name.replace(' ', '_')}.png"), dpi=150)
    plt.close()
    return macro_f1

def majority_voting(model_logits):
    """Mỗi mô hình bầu chọn 1 nhãn, nhãn nào nhiều vote nhất thì thắng."""
    all_preds = []
    for name, logits in model_logits.items():
        preds = np.argmax(logits, axis=-1)
        all_preds.append(preds)
    
    all_preds = np.array(all_preds) # shape: (num_models, num_samples)
    final_preds = []
    for i in range(all_preds.shape[1]):
        votes = all_preds[:, i]
        counts = np.bincount(votes)
        final_preds.append(np.argmax(counts))
        
    return np.array(final_preds)

def average_logits(model_logits):
    """Tính trung bình logits (Soft Voting)."""
    logits_list = list(model_logits.values())
    avg_logits = np.mean(logits_list, axis=0)
    return np.argmax(avg_logits, axis=-1)

def stacking(model_logits, y_true):
    """Sử dụng Logistic Regression để học cách kết hợp các logits."""
    # Nối tất cả logits thành feature vector
    # shape: (num_samples, num_models * 7)
    X = np.concatenate(list(model_logits.values()), axis=1)
    
    # Do chúng ta chỉ có test set để chạy ensemble, chúng ta sử dụng StratifiedKFold để train/predict
    # Trong môi trường thực tế, Stacking nên được train trên Validation set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_preds = np.zeros_like(y_true)
    
    for train_idx, test_idx in skf.split(X, y_true):
        X_tr, y_tr = X[train_idx], y_true[train_idx]
        X_te = X[test_idx]
        
        lr = LogisticRegression(max_iter=1000, multi_class='multinomial')
        lr.fit(X_tr, y_tr)
        final_preds[test_idx] = lr.predict(X_te)
        
    return final_preds

if __name__ == "__main__":
    print("="*60)
    print("BẮT ĐẦU ENSEMBLE LEARNING")
    print("="*60)
    
    model_logits, y_true = load_predictions()
    
    if not model_logits:
        print("❌ Không tìm thấy dữ liệu dự đoán (test_logits.npy) của bất kỳ mô hình nào!")
        print("Vui lòng chạy file test.py ít nhất 2 lần với các mô hình khác nhau trước.")
        exit()
        
    print(f"Đã tải dự đoán từ {len(model_logits)} mô hình:")
    for name in model_logits.keys():
        print(f" - {name}")
        
    print("\n1. Đánh giá Majority Voting...")
    preds_mv = majority_voting(model_logits)
    evaluate_and_save(y_true, preds_mv, "Majority Voting")
    
    print("\n2. Đánh giá Average Logits (Soft Voting)...")
    preds_al = average_logits(model_logits)
    evaluate_and_save(y_true, preds_al, "Average Logits")
    
    print("\n3. Đánh giá Stacking (Meta-Classifier)...")
    preds_stack = stacking(model_logits, y_true)
    evaluate_and_save(y_true, preds_stack, "Stacking LR")
    
    print(f"\n✅ HOÀN TẤT! Kết quả được lưu tại: {OUTPUT_DIR}")
