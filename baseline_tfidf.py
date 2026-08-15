import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from underthesea import word_tokenize

RANDOM_STATE = 42

OUTPUT_DIR_SVM = "./results/baseline_svm"
OUTPUT_DIR_LR = "./results/baseline_lr"
os.makedirs(OUTPUT_DIR_SVM, exist_ok=True)
os.makedirs(OUTPUT_DIR_LR, exist_ok=True)

LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}

print("\n[1] Đang tải dataset tridm/UIT-VSMEC ...")
raw = load_dataset("tridm/UIT-VSMEC")
frames = [split_data.to_pandas() for split_data in raw.values()]
full_df = pd.concat(frames, ignore_index=True)

full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
full_df["labels"] = full_df["labels"].map(label2id)
full_df.dropna(subset=["labels"], inplace=True)
full_df["labels"] = full_df["labels"].astype(int)

# Chia giống hệ thống chính (80/20) để test set hoàn toàn giống nhau
train_val_df, test_df = train_test_split(
    full_df, test_size=0.20,
    stratify=full_df["labels"], random_state=RANDOM_STATE
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.125,        # 0.125 × 80% = 10% tổng
    stratify=train_val_df["labels"], random_state=RANDOM_STATE
)

# Gộp Train và Val thành một tập chung cho ML truyền thống (không cần tập Val riêng để early stopping)
train_ml_df = pd.concat([train_df, val_df])

print(f"Số mẫu Train dùng cho ML: {len(train_ml_df)}")
print(f"Số mẫu Test: {len(test_df)}")

print("\n[2] Tiền xử lý (Word Segmentation) ...")
def preprocess_text(text: str) -> str:
    return word_tokenize(text, format="text")

X_train = train_ml_df["text"].apply(preprocess_text).tolist()
y_train = train_ml_df["labels"].tolist()
X_test = test_df["text"].apply(preprocess_text).tolist()
y_test = test_df["labels"].tolist()

print("\n[3] Trích xuất đặc trưng TF-IDF ...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

def evaluate_and_save(model, name, out_dir):
    print(f"\n[4] Huấn luyện và đánh giá mô hình {name} ...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    report_str = classification_report(y_test, y_pred, labels=list(range(7)), target_names=LABEL_NAMES, digits=4)
    print(report_str)
    
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {name} + TF-IDF\n\n")
        f.write(report_str)
        
    cm = confusion_matrix(y_test, y_pred, labels=list(range(7)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Confusion Matrix\n{name} + TF-IDF", fontsize=13, fontweight="bold")
    plt.ylabel("Nhãn thực tế")
    plt.xlabel("Nhãn dự đoán")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    
    print(f"✅ Đã lưu kết quả tại: {out_dir}")

# SVM
svm_model = LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
evaluate_and_save(svm_model, "Linear SVM", OUTPUT_DIR_SVM)

# Logistic Regression
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, multi_class='multinomial')
evaluate_and_save(lr_model, "Logistic Regression", OUTPUT_DIR_LR)

print("\n✅ HOÀN TẤT BASELINE TRUYỀN THỐNG!")
