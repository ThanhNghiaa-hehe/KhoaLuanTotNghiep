import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Thử import wordcloud, nếu chưa có sẽ in ra cảnh báo
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
RESULTS_DIR = "./results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "confusion_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def draw_confidence_histogram(model_dirs):
    """Vẽ biểu đồ phân phối độ tự tin của các mô hình khi đoán đúng và đoán sai."""
    print("\n📊 Đang tạo Confidence Histogram...")
    
    correct_confidences = []
    incorrect_confidences = []
    
    for d in model_dirs:
        logits_path = os.path.join(d, "test_logits.npy")
        y_true_path = os.path.join(d, "test_y_true.npy")
        
        if os.path.exists(logits_path) and os.path.exists(y_true_path):
            logits = np.load(logits_path)
            y_true = np.load(y_true_path)
            
            probs = softmax(logits)
            confidences = np.max(probs, axis=1)
            y_pred = np.argmax(logits, axis=1)
            
            is_correct = (y_pred == y_true)
            
            correct_confidences.extend(confidences[is_correct])
            incorrect_confidences.extend(confidences[~is_correct])
            
    if not correct_confidences:
        return
        
    plt.figure(figsize=(10, 6))
    sns.histplot(correct_confidences, color="green", label="Đoán Đúng", kde=True, stat="density", alpha=0.5, binwidth=0.05)
    sns.histplot(incorrect_confidences, color="red", label="Đoán Sai", kde=True, stat="density", alpha=0.5, binwidth=0.05)
    
    plt.title("Phân phối Độ Tự tin (Confidence Distribution)", fontsize=14, fontweight="bold")
    plt.xlabel("Độ tự tin (Xác suất lớn nhất)", fontsize=12)
    plt.ylabel("Mật độ", fontsize=12)
    plt.legend()
    plt.xlim(0, 1)
    
    out_path = os.path.join(OUTPUT_DIR, "error_confidence_histogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu Confidence Histogram: {out_path}")

def generate_wordcloud(text, title, filename):
    if not HAS_WORDCLOUD:
        print(f"⚠️ Chưa cài đặt wordcloud. Bỏ qua vẽ WordCloud cho {title}. (Chạy lệnh: !pip install wordcloud)")
        return
        
    if not text.strip():
        return
        
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100, colormap="inferno").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã lưu Word Cloud: {out_path}")

def analyze_hard_samples():
    print("="*60)
    print("PHÂN TÍCH LỖI NÂNG CAO (CONFUSION ANALYSIS)")
    print("="*60)
    
    search_path = os.path.join(RESULTS_DIR, "*", "error_analysis.csv")
    error_files = glob.glob(search_path)
    
    model_dirs = glob.glob(os.path.join(RESULTS_DIR, "*/"))
    # Filter only base models
    model_dirs = [d for d in model_dirs if os.path.basename(os.path.normpath(d)) not in ["ensemble", "baseline_svm", "baseline_lr", "confusion_analysis", "attention_viz"]]
    
    if len(error_files) == 0:
        print("❌ Không tìm thấy file error_analysis.csv nào. Hãy chạy test.py hoặc run_all_models.py trước.")
        return
        
    print(f"Đã tìm thấy {len(error_files)} file phân tích lỗi từ các mô hình.")
    
    all_errors = []
    
    for file_path in error_files:
        model_name = os.path.basename(os.path.dirname(file_path))
        if model_name in ["ensemble", "baseline_svm", "baseline_lr", "confusion_analysis", "attention_viz"]:
            continue
            
        df = pd.read_csv(file_path)
        df["Mô hình"] = model_name
        all_errors.append(df)
        
    if len(all_errors) == 0:
        print("❌ Không có dữ liệu lỗi hợp lệ.")
        return
        
    combined_df = pd.concat(all_errors, ignore_index=True)
    
    # 1. HARD SAMPLES
    grouped = combined_df.groupby("Câu văn thực tế").agg(
        Số_lần_sai=("Mô hình", "count"),
        Chi_tiết_lỗi=("Mô hình", lambda x: " | ".join([f"{m} đoán [{p}]" for m, p in zip(x, combined_df.loc[x.index, 'Nhãn dự đoán'])]))
    ).reset_index()
    grouped.rename(columns={"Số_lần_sai": "Số lần sai", "Chi_tiết_lỗi": "Chi tiết lỗi"}, inplace=True)
    
    true_labels_df = combined_df[["Câu văn thực tế", "Nhãn thực tế"]].drop_duplicates()
    
    hard_samples = pd.merge(grouped, true_labels_df, on="Câu văn thực tế")
    # Sắp xếp lại thứ tự cột cho đẹp mắt
    hard_samples = hard_samples[["Câu văn thực tế", "Nhãn thực tế", "Số lần sai", "Chi tiết lỗi"]]
    hard_samples = hard_samples.sort_values(by="Số lần sai", ascending=False).reset_index(drop=True)
    
    # Nâng điều kiện Hard Samples lên >= 4 mô hình (hoặc tối đa số file nếu ít hơn)
    threshold = min(4, len(error_files))
    hardest_df = hard_samples[hard_samples["Số lần sai"] >= threshold]
    
    out_path = os.path.join(OUTPUT_DIR, "hard_samples.csv")
    hardest_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Kết quả] Tìm thấy {len(hardest_df)} mẫu siêu khó (sai ở ít nhất {threshold} mô hình).")
    print(f"✅ Đã lưu danh sách Hard Samples tại: {out_path}")
    
    # 2. CONFUSION RATE HEATMAP
    print("\n📊 Đang tạo Confusion Rate Heatmap...")
    pair_counts = combined_df.groupby(["Nhãn thực tế", "Nhãn dự đoán"]).size().reset_index(name="Tổng số lần nhầm")
    pair_counts = pair_counts.sort_values("Tổng số lần nhầm", ascending=False).reset_index(drop=True)
    
    out_pair_path = os.path.join(OUTPUT_DIR, "global_confusion_pairs.csv")
    pair_counts.to_csv(out_pair_path, index=False, encoding="utf-8-sig")
    
    # Tạo ma trận Pivot
    pivot_table = pair_counts.pivot(index="Nhãn thực tế", columns="Nhãn dự đoán", values="Tổng số lần nhầm").fillna(0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="Reds", linewidths=0.5)
    plt.title("Ma trận Tỷ lệ Nhầm lẫn Tổng hợp (Cross-model Confusion)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Nhãn Thực tế", fontsize=12)
    plt.xlabel("Nhãn Dự đoán Sai", fontsize=12)
    
    heatmap_path = os.path.join(OUTPUT_DIR, "global_confusion_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu Confusion Heatmap: {heatmap_path}")
    
    # 3. WORD CLOUD CHO CẶP NHẦM NHIỀU NHẤT
    if not pair_counts.empty:
        top_pair = pair_counts.iloc[0]
        true_label = top_pair["Nhãn thực tế"]
        pred_label = top_pair["Nhãn dự đoán"]
        
        # Lấy tất cả các câu bị nhầm trong cặp này
        subset_texts = combined_df[(combined_df["Nhãn thực tế"] == true_label) & (combined_df["Nhãn dự đoán"] == pred_label)]["Câu văn thực tế"].tolist()
        combined_text = " ".join(subset_texts)
        
        generate_wordcloud(
            combined_text, 
            f"Word Cloud: Thực tế [{true_label}] bị nhầm thành [{pred_label}]", 
            "wordcloud_top_confusion.png"
        )

    # 4. CONFIDENCE HISTOGRAM
    draw_confidence_histogram(model_dirs)
    
    print("\n" + "="*60)
    print("✅ PHÂN TÍCH LỖI NÂNG CAO ĐÃ HOÀN TẤT!")
    print(f"📂 Xem kết quả tại: {OUTPUT_DIR}/")

if __name__ == "__main__":
    analyze_hard_samples()
