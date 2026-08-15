import os
import time
import random
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
from underthesea import word_tokenize

def back_translate_advanced(text, source='vi', intermediate_langs=['en', 'fr', 'ja', 'ko']):
    """
    Dịch câu từ VI -> NGÔN NGỮ TRUNG GIAN -> VI.
    Chọn ngẫu nhiên một ngôn ngữ trung gian (Anh, Pháp, Nhật, Hàn) 
    để tạo ra sự đa dạng tối đa về ngữ pháp và từ vựng.
    """
    try:
        # Chọn ngẫu nhiên 1 ngôn ngữ trung gian
        intermediate = random.choice(intermediate_langs)
        
        # VI -> Intermediate
        translated = GoogleTranslator(source=source, target=intermediate).translate(text)
        time.sleep(0.5) # Tránh bị block API
        
        # Intermediate -> VI
        back_translated = GoogleTranslator(source=intermediate, target=source).translate(translated)
        time.sleep(0.5)
        return back_translated
    except Exception as e:
        return text # Nếu lỗi thì trả về câu gốc để đảm bảo an toàn

def apply_eda(text):
    """
    Xáo trộn dữ liệu nhanh bằng Kỹ thuật EDA (Random Swap hoặc Random Deletion)
    Dùng underthesea để tháo rời từ ghép Tiếng Việt nhằm xáo trộn an toàn.
    """
    try:
        # Tách từ, giữ nguyên từ ghép tiếng Việt (như 'vui_vẻ' sẽ là 1 từ)
        words = word_tokenize(text)
        
        # Nếu câu quá ngắn (< 3 từ), không nên xáo trộn
        if len(words) < 3:
            return text
            
        # Tung đồng xu: 50% Random Swap, 50% Random Deletion
        if random.random() < 0.5:
            # Random Swap (Đổi chỗ 2 từ ngẫu nhiên)
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        else:
            # Random Deletion (Xóa 1 từ ngẫu nhiên khỏi câu)
            idx_to_delete = random.randint(0, len(words) - 1)
            words.pop(idx_to_delete)
            
        # Ghép lại thành câu
        return " ".join(words)
    except Exception as e:
        return text

def is_valid_augmentation(orig_text, new_text):
    """
    BỘ LỌC NHIỄU (Data Cleansing): 
    Kiểm tra xem câu mới sinh ra có đạt chất lượng không, tránh đưa rác vào mô hình.
    """
    orig_text = str(orig_text).strip().lower()
    new_text = str(new_text).strip().lower()
    
    # 1. Loại bỏ câu bị trùng lặp y chang
    if new_text == orig_text:
        return False
        
    # 2. Loại bỏ câu quá ngắn (rác do Google dịch lỗi thành 1 chữ như "hmm", "ờ")
    if len(new_text.split()) < 3 and len(orig_text.split()) >= 3:
        return False
        
    # 3. Loại bỏ câu có độ dài bất thường (ngắn hơn 30% hoặc dài gấp 3 lần câu gốc)
    orig_len = len(orig_text)
    new_len = len(new_text)
    if new_len < (orig_len * 0.3) or new_len > (orig_len * 3):
        return False
        
    return True

def augment_minority_classes_upgrade(train_df, target_count=1000, save_path="./TaiLieu/augmented_train_upgrade.csv"):
    """
    Tăng cường dữ liệu bằng phiên bản nâng cấp (Đa dạng ngôn ngữ + Lọc nhiễu kỹ càng + Chống trùng lặp tuyệt đối + EDA).
    """
    if os.path.exists(save_path):
        print(f"[Augmentation Upgrade] Đã tìm thấy file '{save_path}', tiến hành load trực tiếp...")
        return pd.read_csv(save_path)
        
    print(f"[Augmentation Upgrade] Bắt đầu Hybrid Augmentation (Mục tiêu: {target_count} mẫu/nhãn)...")
    
    # 0: Enjoyment, 1: Sadness, 2: Anger, 3: Fear, 4: Disgust, 5: Surprise, 6: Other
    label_counts = train_df['labels'].value_counts()
    augmented_rows = []
    
    # BỘ NHỚ TOÀN CỤC CHỐNG TRÙNG LẶP: Khởi tạo bằng toàn bộ câu gốc trong tập Train
    global_seen_texts = set(train_df['text'].str.strip().str.lower().tolist())
    
    for label in [2, 3, 5]: # Chỉ augment Anger(2), Fear(3), Surprise(5)
        current_count = label_counts.get(label, 0)
        num_to_augment = target_count - current_count
        
        if num_to_augment <= 0:
            continue
            
        print(f"  + Đang augment nhãn {label} (Cần bổ sung thêm {num_to_augment} mẫu chất lượng cao)...")
        minority_df = train_df[train_df['labels'] == label]
        
        # Lấy mẫu thừa ra (gấp 3 lần) vì Bộ lọc chống trùng lặp giờ rất gắt, sẽ gạch bỏ rất nhiều câu
        samples_to_augment = minority_df.sample(n=num_to_augment * 3, replace=True, random_state=42)
        
        success_count = 0
        pbar = tqdm(total=num_to_augment, desc=f"Nhãn {label}")
        
        for idx, row in samples_to_augment.iterrows():
            if success_count >= num_to_augment:
                break # Đủ KPI thì dừng lại
                
            orig_text = row['text']
            
            # Hybrid Augmentation: 70% Back-Translation Đa Ngôn Ngữ, 30% EDA
            if random.random() < 0.7:
                new_text = back_translate_advanced(orig_text)
            else:
                new_text = apply_eda(orig_text)
                
            new_text_lower = str(new_text).strip().lower()
            
            # Kiểm tra: 1. Đạt chuẩn bộ lọc nhiễu VÀ 2. CHƯA TỪNG xuất hiện trong lịch sử
            if is_valid_augmentation(orig_text, new_text) and (new_text_lower not in global_seen_texts):
                augmented_rows.append({'text': new_text, 'labels': label})
                global_seen_texts.add(new_text_lower) # Ghi nhớ câu này để không cho trùng ở tương lai
                success_count += 1
                pbar.update(1)
                
        pbar.close()
                
    if len(augmented_rows) > 0:
        aug_df = pd.DataFrame(augmented_rows)
        # Gộp với train_df gốc
        final_train_df = pd.concat([train_df, aug_df], ignore_index=True)
        # Xáo trộn lại toàn bộ
        final_train_df = final_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Lưu file mới
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_train_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n[Augmentation Upgrade] Đã tạo và lưu {len(augmented_rows)} mẫu mới SẠCH tại {save_path}")
        return final_train_df
    else:
        print("[Augmentation Upgrade] Không có mẫu mới nào đạt chuẩn được tạo.")
        return train_df

if __name__ == "__main__":
    print("=====================================================")
    print(" BẮT ĐẦU CHẠY AUGMENTATION UPGRADE (PHIÊN BẢN PRO)")
    print("=====================================================")
    
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
        
        # Chia train/test đảm bảo khớp với code chính
        train_val_df, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["labels"], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df["labels"], random_state=42)
        
        print(f"[2] Đã chuẩn bị xong tập Train ({len(train_df)} mẫu). Bắt đầu chiến dịch Hybrid Augmentation...")
        
        # Đã nâng target_count lên 1000 mẫu
        augmented_train_df = augment_minority_classes_upgrade(
            train_df, 
            target_count=1000, 
            save_path="./TaiLieu/augmented_train_upgrade.csv"
        )
        
        print("\n🎉 HOÀN TẤT! File dữ liệu tăng cường phiên bản PRO đã được lưu.")
        print("Tên file: ./TaiLieu/augmented_train_upgrade.csv")
        print("--> GHI CHÚ: Hãy mở file run_all_models.py và đổi đường dẫn đọc csv sang file này nhé!")
        
    except ImportError as e:
        print(f"⚠️ Lỗi: Thiếu thư viện. Vui lòng cài đặt bằng lệnh:")
        print("pip install datasets scikit-learn deep-translator pandas tqdm underthesea")
