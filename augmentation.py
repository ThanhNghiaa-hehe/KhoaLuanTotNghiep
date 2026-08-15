import os
import time
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

def back_translate(text, source='vi', intermediate='en'):
    """Dịch câu từ VI -> EN -> VI để tạo câu mới đồng nghĩa nhưng khác cấu trúc."""
    try:
        # VI -> EN
        translated = GoogleTranslator(source=source, target=intermediate).translate(text)
        time.sleep(0.5) # Tránh bị rate-limit
        # EN -> VI
        back_translated = GoogleTranslator(source=intermediate, target=source).translate(translated)
        time.sleep(0.5)
        return back_translated
    except Exception as e:
        print(f"Lỗi dịch câu '{text}': {e}")
        return text # Nếu lỗi thì trả về câu gốc

def augment_minority_classes(train_df, target_count=800, save_path="./TaiLieu/augmented_train.csv"):
    """
    Tăng cường dữ liệu cho các nhãn thiểu số (Fear, Surprise, Anger) lên mức target_count.
    Sử dụng Back Translation. Nếu đã có file save_path thì load lên thay vì dịch lại (tiết kiệm thời gian).
    """
    if os.path.exists(save_path):
        print(f"[Augmentation] Đã tìm thấy file '{save_path}', tiến hành load trực tiếp...")
        return pd.read_csv(save_path)
        
    print("[Augmentation] Không tìm thấy file lưu tạm, bắt đầu Back Translation (cần Internet)...")
    
    # 0: Enjoyment, 1: Sadness, 2: Anger, 3: Fear, 4: Disgust, 5: Surprise, 6: Other
    label_counts = train_df['labels'].value_counts()
    
    augmented_rows = []
    
    for label in [2, 3, 5]: # Chỉ augment Anger(2), Fear(3), Surprise(5)
        current_count = label_counts.get(label, 0)
        num_to_augment = target_count - current_count
        
        if num_to_augment <= 0:
            continue
            
        print(f"  + Đang augment nhãn {label} (Cần thêm {num_to_augment} mẫu)...")
        
        # Lấy các mẫu của nhãn này
        minority_df = train_df[train_df['labels'] == label]
        
        # Lặp lại nếu số lượng cần thêm lớn hơn số lượng hiện có
        samples_to_augment = minority_df.sample(n=num_to_augment, replace=True, random_state=42)
        
        for idx, row in tqdm(samples_to_augment.iterrows(), total=len(samples_to_augment)):
            orig_text = row['text']
            new_text = back_translate(orig_text)
            
            # Chỉ thêm nếu câu mới khác câu cũ
            if new_text.strip().lower() != orig_text.strip().lower():
                augmented_rows.append({'text': new_text, 'labels': label})
                
    if len(augmented_rows) > 0:
        aug_df = pd.DataFrame(augmented_rows)
        # Gộp với train_df gốc
        final_train_df = pd.concat([train_df, aug_df], ignore_index=True)
        # Xáo trộn lại
        final_train_df = final_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Lưu lại để lần sau chạy không phải dịch lại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_train_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"[Augmentation] Đã tạo và lưu {len(augmented_rows)} mẫu mới tại {save_path}")
        return final_train_df
    else:
        print("[Augmentation] Không có mẫu mới nào được tạo.")
        print("[Augmentation] Không có mẫu mới nào được tạo.")
        return train_df

if __name__ == "__main__":
    print("=====================================================")
    print(" BẮT ĐẦU CHẠY AUGMENTATION (TẠO DỮ LIỆU ĐỘC LẬP)")
    print("=====================================================")
    
    # 1. Tải bộ dữ liệu gốc để chuẩn bị
    try:
        from datasets import load_dataset
        from sklearn.model_selection import train_test_split
        
        print("[1] Đang tải dataset tridm/UIT-VSMEC từ HuggingFace...")
        raw = load_dataset("tridm/UIT-VSMEC")
        frames = [split_data.to_pandas() for split_data in raw.values()]
        full_df = pd.concat(frames, ignore_index=True)
        full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
        
        LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
        label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
        full_df["labels"] = full_df["labels"].map(label2id)
        full_df.dropna(subset=["labels"], inplace=True)
        full_df["labels"] = full_df["labels"].astype(int)
        
        # Chia train/test y như trong run_all_models.py để đảm bảo khớp dữ liệu
        train_val_df, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["labels"], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df["labels"], random_state=42)
        
        print(f"[2] Đã chuẩn bị xong tập Train ({len(train_df)} mẫu). Bắt đầu dịch thuật...")
        # 2. Gọi hàm Augmentation
        augmented_train_df = augment_minority_classes(train_df, target_count=800, save_path="./TaiLieu/augmented_train.csv")
        
        print("\n🎉 HOÀN TẤT! File dữ liệu tăng cường đã được lưu tại ./TaiLieu/augmented_train.csv")
        print("Bây giờ bạn có thể chạy file run_all_models.py một cách nhanh chóng!")
        
    except ImportError as e:
        print(f"⚠️ Lỗi: Thiếu thư viện. Vui lòng cài đặt bằng lệnh: pip install datasets scikit-learn deep-translator pandas")
        print(f"Chi tiết lỗi: {e}")
