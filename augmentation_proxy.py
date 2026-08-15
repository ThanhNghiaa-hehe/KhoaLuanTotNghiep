import os
import re
import time
import pandas as pd
from tqdm import tqdm

# SỬ DỤNG OPENAI SDK ĐỂ KẾT NỐI VỚI NGUỒN KHÁC (BÊN THỨ 3)
from openai import OpenAI

# =========================================================================
# CẤU HÌNH API CỦA WEB BÊN THỨ 3
# =========================================================================
API_KEY = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"
BASE_URL = "ĐIỀN_BASE_URL_VÀO_ĐÂY" # Ví dụ: "https://api.tên-trang-web.com/v1"
MODEL_NAME = "GLM-4.7-Flash" # Đổi tên model theo ý muốn

client = None

LABEL_MAP = {
    0: "Enjoyment (Thích thú, vui vẻ, hạnh phúc)",
    1: "Sadness (Buồn bã, thất vọng)",
    2: "Anger (Tức giận, phẫn nộ, chửi thề)",
    3: "Fear (Sợ hãi, hoang mang, lo lắng)",
    4: "Disgust (Kinh tởm, khinh bỉ, chê bai)",
    5: "Surprise (Bất ngờ, ngạc nhiên, ngỡ ngàng)",
    6: "Other (Khác / Trung tính)"
}

def generate_proxy_batch(label_id, example_sentences, num_generate=20):
    label_desc = LABEL_MAP[label_id]
    examples_text = "\n".join([f"- {text}" for text in example_sentences])
    
    prompt = f"""Bạn là một chuyên gia ngôn ngữ học và một người dùng mạng xã hội GenZ tại Việt Nam.
Tôi đang xây dựng bộ dữ liệu AI để nhận diện cảm xúc. Dưới đây là {len(example_sentences)} câu bình luận mẫu mang cảm xúc: {label_desc}

Các câu mẫu:
{examples_text}

NHIỆM VỤ CỦA BẠN:
Hãy sáng tác ra đúng {num_generate} câu bình luận MỚI HOÀN TOÀN dựa trên phong cách, từ lóng và cách hành văn mạng xã hội của các câu mẫu trên. 
Yêu cầu bắt buộc:
1. Thể hiện RÕ RÀNG cảm xúc: {label_desc}.
2. Các câu phải khác biệt nhau về từ vựng và ngữ cảnh. Tuyệt đối không lặp lại câu mẫu.
3. Độ dài tương đương các câu mẫu (ngắn gọn, từ 3 đến 20 từ).
4. Chỉ trả về danh sách {num_generate} câu, mỗi câu trên 1 dòng, KHÔNG gạch đầu dòng, KHÔNG đánh số thứ tự, KHÔNG giải thích gì thêm.
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7 
        )
        
        result_text = response.choices[0].message.content
        
        # Xử lý text rác 
        raw_lines = [line.strip() for line in result_text.strip().split('\n') if line.strip()]
        generated_texts = [re.sub(r'^[\d]+[.)\-]\s*', '', line).lstrip('-*').strip() 
                           for line in raw_lines]
        
        # GLM-4.7-Flash giới hạn 10 Request / Phút
        time.sleep(6) 
        
        return generated_texts
    except Exception as e:
        print(f"\n[Lỗi API] {e}")
        time.sleep(10)
        return []

def augment_with_proxy(train_df, target_count=500, save_path="./TaiLieu/augmented_train_proxy.csv"):
    if API_KEY == "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY" or BASE_URL == "ĐIỀN_BASE_URL_VÀO_ĐÂY":
        print("❌ LỖI: Bạn chưa điền API Key hoặc Base URL ở dòng 12!")
        return train_df
    
    global client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
        
    if os.path.exists(save_path):
        print(f"[Proxy Augment] Đã tìm thấy file '{save_path}', tiến hành load trực tiếp...")
        return pd.read_csv(save_path)
        
    print(f"[Proxy Augment] Bắt đầu gọi {MODEL_NAME} (Mục tiêu: {target_count} mẫu/nhãn)...")
    
    label_counts = train_df['labels'].value_counts()
    augmented_rows = []
    
    stats = {
        'success': {2: 0, 3: 0, 5: 0},
        'rejected': {2: 0, 3: 0, 5: 0}
    }
    
    global_seen_texts = set(train_df['text'].str.strip().str.lower().tolist())
    
    for label in [2, 3, 5]: 
        current_count = label_counts.get(label, 0)
        num_to_augment = target_count - current_count
        
        if num_to_augment <= 0:
            continue
            
        print(f"\n  + Đang nhờ {MODEL_NAME} sáng tác nhãn {LABEL_MAP[label]} (Cần thêm {num_to_augment} câu)...")
        minority_df = train_df[train_df['labels'] == label]
        
        success_count = 0
        failed_retries = 0 
        pbar = tqdm(total=num_to_augment, desc=f"Nhãn {label}")
        
        while success_count < num_to_augment:
            if failed_retries > 50:
                print(f"\n⚠️ CẢNH BÁO: Đã gọi API lỗi quá 50 lần cho nhãn {label}. Bỏ qua để tránh treo máy!")
                break
                
            examples = minority_df.sample(n=min(5, len(minority_df)))['text'].tolist()
            
            # Đặt 20 câu/lần gọi để tối ưu chất lượng
            num_request = min(20, num_to_augment - success_count + 5) 
            
            new_sentences = generate_proxy_batch(label, examples, num_generate=num_request)
            
            if len(new_sentences) == 0:
                failed_retries += 1
                continue
            
            added_any = False
            for text in new_sentences:
                if success_count >= num_to_augment:
                    break
                
                text_lower = str(text).strip().lower()
                word_count = len(text_lower.split())
                
                if 3 <= word_count <= 40 and text_lower not in global_seen_texts:
                    if not text_lower.startswith(("dạ", "chào", "đây là", "tất nhiên", "dưới đây", "chắc chắn")):
                        augmented_rows.append({'text': text, 'labels': label})
                        global_seen_texts.add(text_lower)
                        success_count += 1
                        stats['success'][label] += 1
                        pbar.update(1)
                        added_any = True
                    else:
                        stats['rejected'][label] += 1
                else:
                    stats['rejected'][label] += 1
                        
            if not added_any:
                failed_retries += 1
            else:
                failed_retries = 0 
                
        pbar.close()
        
        if len(augmented_rows) > 0:
            pd.DataFrame(augmented_rows).to_csv(save_path.replace('.csv', '_checkpoint.csv'), index=False, encoding='utf-8-sig')
            print(f"  -> Đã lưu nháp an toàn (Checkpoint) {len(augmented_rows)} câu...")
                
    if len(augmented_rows) > 0:
        aug_df = pd.DataFrame(augmented_rows)
        final_train_df = pd.concat([train_df, aug_df], ignore_index=True)
        final_train_df = final_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_train_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f" BÁO CÁO THỐNG KÊ DATA AUGMENTATION ({MODEL_NAME})")
        print("="*50)
        total_success = 0
        total_rejected = 0
        for lbl in [2, 3, 5]:
            print(f"- Nhãn {LABEL_MAP[lbl]}:")
            print(f"  + Sinh thành công: {stats['success'][lbl]} câu")
            print(f"  + Lọc bỏ (rác/trùng): {stats['rejected'][lbl]} câu")
            total_success += stats['success'][lbl]
            total_rejected += stats['rejected'][lbl]
            
        print("-" * 50)
        print(f"Tổng hợp: Bổ sung {total_success} câu tinh khiết (Loại bỏ {total_rejected} câu rác).")
        print(f"Đã lưu dữ liệu tại: {save_path}")
        print("="*50 + "\n")
        
        return final_train_df
    else:
        print("\n[Proxy Augment] Không có câu nào được tạo ra.")
        return train_df

if __name__ == "__main__":
    print("=====================================================")
    print(f" BẮT ĐẦU CHẠY AUGMENTATION BẰNG {MODEL_NAME} (BÊN THỨ 3)")
    print("=====================================================")
    
    try:
        from datasets import load_dataset
        from sklearn.model_selection import train_test_split
        
        print("[1] Đang tải dataset tridm/UIT-VSMEC từ HuggingFace...")
        raw = load_dataset("tridm/UIT-VSMEC")
        frames = [split_data.to_pandas() for split_data in raw.values()]
        full_df = pd.concat(frames, ignore_index=True)
        full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
        
        LABEL_NAMES_LIST = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
        label2id = {name: idx for idx, name in enumerate(LABEL_NAMES_LIST)}
        full_df["labels"] = full_df["labels"].map(label2id)
        full_df.dropna(subset=["labels"], inplace=True)
        full_df["labels"] = full_df["labels"].astype(int)
        
        train_val_df, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["labels"], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df["labels"], random_state=42)
        
        print(f"[2] Bắt đầu chiến dịch Sáng tác nội dung...")
        
        # CHẠY CHO MỐC 600 ĐỂ CÀY ĐIỂM F1 > 63%
        augmented_train_df = augment_with_proxy(
            train_df, 
            target_count=600, 
            save_path="./TaiLieu/augmented_train_proxy.csv"
        )
        
    except ImportError as e:
        print(f"⚠️ Lỗi: Thiếu thư viện. Hãy qua Colab mở 1 code cell và chạy lệnh sau:")
        print("!pip install openai datasets scikit-learn pandas tqdm")
