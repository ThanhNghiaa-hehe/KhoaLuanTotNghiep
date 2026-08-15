import os
import re
import time
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# BƯỚC 1: DÁN API KEY CỦA BẠN VÀO ĐÂY
API_KEY = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"

# Lưu ý: genai.configure() và model sẽ được khởi tạo bên trong hàm augment_with_gemini()
# để tránh crash ngay khi import file mà chưa điền API Key.
model = None

LABEL_MAP = {
    0: "Enjoyment (Thích thú, vui vẻ, hạnh phúc)",
    1: "Sadness (Buồn bã, thất vọng)",
    2: "Anger (Tức giận, phẫn nộ, chửi thề)",
    3: "Fear (Sợ hãi, hoang mang, lo lắng)",
    4: "Disgust (Kinh tởm, khinh bỉ, chê bai)",
    5: "Surprise (Bất ngờ, ngạc nhiên, ngỡ ngàng)",
    6: "Other (Khác / Trung tính)"
}

def generate_gemini_batch(label_id, example_sentences, num_generate=20):
    """
    Gọi API Gemini để sáng tác ra 1 lô (20 câu) dựa trên 5 câu ví dụ gốc.
    Việc gọi lô lớn giúp tiết kiệm số lần Request, vượt qua hạn mức Free Tier dễ dàng.
    """
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
        # Tắt toàn bộ bộ lọc an toàn để cho phép sinh từ lóng/chửi thề (nhãn Anger)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        # Xử lý text rác (nếu AI cố tình gạch đầu dòng hoặc đánh số)
        # Dùng regex để xóa chính xác số thứ tự đầu dòng (vd: "1.", "10.", "20.")
        # Tránh dùng lstrip() vì nó sẽ cắt nhầm số trong nội dung câu
        raw_lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        generated_texts = [re.sub(r'^[\d]+[.)\-]\s*', '', line).lstrip('-*').strip() 
                           for line in raw_lines]
        
        # TÍNH NĂNG CHỐNG BAN (Rate Limit):
        # Với gemini-1.5-flash và API Key mới, Google cho phép 15 Request / Phút.
        # Ta sleep 5 giây là quá đủ để chạy nhanh mà không bị chặn.
        time.sleep(5) 
        
        return generated_texts
    except Exception as e:
        print(f"\n[Lỗi API] {e}")
        time.sleep(10) # Đợi lâu hơn nếu lỡ bị cấm do mạng lag
        return []

def augment_with_gemini(train_df, target_count=500, save_path="./TaiLieu/augmented_train_genai.csv"):
    """
    Vòng lặp chính điều phối quá trình sinh dữ liệu.
    """
    if API_KEY == "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY":
        print("❌ LỖI: Bạn chưa điền API Key vào file code! Hãy điền API_KEY ở dòng 10.")
        return train_df
    
    # Khởi tạo Gemini tại đây (sau khi đã chắc chắn API Key hợp lệ)
    global model
    genai.configure(api_key=API_KEY)
    # Trở về chân ái gemini-1.5-flash: Ổn định, miễn phí rộng rãi, và hỗ trợ API Key mới
    model = genai.GenerativeModel('gemini-1.5-flash')
        
    if os.path.exists(save_path):
        print(f"[GenAI Augment] Đã tìm thấy file '{save_path}', tiến hành load trực tiếp...")
        return pd.read_csv(save_path)
        
    print(f"[GenAI Augment] Bắt đầu gọi Google Gemini (Mục tiêu: {target_count} mẫu/nhãn)...")
    print("⏳ Máy đang chạy tự động, thời gian dự kiến: ~5 phút. Vui lòng không tắt màn hình...")
    
    label_counts = train_df['labels'].value_counts()
    augmented_rows = []
    
    # Biến thống kê báo cáo cho Khóa luận
    stats = {
        'success': {2: 0, 3: 0, 5: 0},
        'rejected': {2: 0, 3: 0, 5: 0}
    }
    
    # BỘ NHỚ TOÀN CỤC: Giúp ngăn chặn việc Gemini sinh ra 2 câu giống hệt nhau
    global_seen_texts = set(train_df['text'].str.strip().str.lower().tolist())
    
    for label in [2, 3, 5]: # Xử lý 3 nhãn thiểu số: Anger, Fear, Surprise
        current_count = label_counts.get(label, 0)
        num_to_augment = target_count - current_count
        
        if num_to_augment <= 0:
            continue
            
        print(f"\n  + Đang nhờ Gemini sáng tác nhãn {LABEL_MAP[label]} (Cần thêm {num_to_augment} câu)...")
        minority_df = train_df[train_df['labels'] == label]
        
        success_count = 0
        failed_retries = 0 # Bộ đếm chống lặp vô hạn
        pbar = tqdm(total=num_to_augment, desc=f"Nhãn {label}")
        
        while success_count < num_to_augment:
            if failed_retries > 50:
                print(f"\n⚠️ CẢNH BÁO: Đã gọi API lỗi quá 50 lần cho nhãn {label}. Bỏ qua để tránh treo máy!")
                break
                
            # Bốc 5 câu ngẫu nhiên từ tập gốc làm "cảm hứng văn phong" cho Gemini
            examples = minority_df.sample(n=min(5, len(minority_df)))['text'].tolist()
            
            # Đã có API Key mới với hạn mức xịn, ta giảm số câu xuống 20/lần
            # để đảm bảo CHẤT LƯỢNG NGỮ NGHĨA CAO NHẤT (tránh Semantic Fatigue)
            num_request = min(20, num_to_augment - success_count + 5) 
            
            new_sentences = generate_gemini_batch(label, examples, num_generate=num_request)
            
            if len(new_sentences) == 0:
                failed_retries += 1
                continue
            
            added_any = False
            for text in new_sentences:
                if success_count >= num_to_augment:
                    break
                
                text_lower = str(text).strip().lower()
                word_count = len(text_lower.split())
                
                # Bộ lọc rác AI: Chỉ nhận câu từ 3 -> 40 chữ VÀ CHƯA TỪNG XUẤT HIỆN
                if 3 <= word_count <= 40 and text_lower not in global_seen_texts:
                    # Chặn thêm ảo giác AI (Câu chào hỏi)
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
                        
            # Nếu Gemini sinh ra toàn câu trùng lặp hoặc rác, tính là 1 lần failed
            if not added_any:
                failed_retries += 1
            else:
                failed_retries = 0 # Trả lại bộ đếm nếu có tiến triển
                
        pbar.close()
        
        # [TÍNH NĂNG MỚI: AUTO-SAVE CHECKPOINT]
        # Lưu nháp ngay lập tức sau khi xong mỗi nhãn. 
        # Nếu đang chạy nhãn Surprise mà API chết, ta vẫn còn giữ được Anger và Fear.
        if len(augmented_rows) > 0:
            pd.DataFrame(augmented_rows).to_csv(save_path.replace('.csv', '_checkpoint.csv'), index=False, encoding='utf-8-sig')
            print(f"  -> Đã lưu nháp an toàn (Checkpoint) {len(augmented_rows)} câu...")
                
    if len(augmented_rows) > 0:
        aug_df = pd.DataFrame(augmented_rows)
        # Gộp dữ liệu Gemini sáng tác vào tập Train gốc
        final_train_df = pd.concat([train_df, aug_df], ignore_index=True)
        final_train_df = final_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_train_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(" BÁO CÁO THỐNG KÊ DATA AUGMENTATION (GEN AI)")
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
        print("\n[GenAI Augment] Không có câu nào được tạo ra.")
        return train_df

if __name__ == "__main__":
    print("=====================================================")
    print(" BẮT ĐẦU CHẠY AUGMENTATION BẰNG GOOGLE GEMINI")
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
        
        # Chia tách chuẩn với code train
        train_val_df, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["labels"], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df["labels"], random_state=42)
        
        print(f"[2] Bắt đầu chiến dịch Sáng tác nội dung (Generative AI)...")
        
        # ĐẨY TARGET LÊN 600 ĐỂ TÌM KIẾM SỰ ĐỘT PHÁ (MỤC TIÊU >63%)
        augmented_train_df = augment_with_gemini(
            train_df, 
            target_count=600, 
            save_path="./TaiLieu/augmented_train_genai.csv"
        )
        
    except ImportError as e:
        print(f"⚠️ Lỗi: Thiếu thư viện. Hãy qua Colab mở 1 code cell và chạy lệnh sau:")
        print("!pip install google-generativeai datasets scikit-learn pandas tqdm")
