import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer
)

MODEL_NAME = "vinai/phobert-base-v2"
OUTPUT_DIR = "./models/phobert-dapt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Giả lập: Đường dẫn tới file text chứa dữ liệu mạng xã hội chưa gán nhãn (1 câu / dòng)
UNLABELED_DATA_PATH = "./TaiLieu/unlabeled_social_media.txt"

def run_dapt():
    print("="*60)
    print("DOMAIN-ADAPTIVE PRE-TRAINING (DAPT) VỚI MASKED LANGUAGE MODELING")
    print("="*60)
    
    if not os.path.exists(UNLABELED_DATA_PATH):
        print(f"❌ Không tìm thấy file dữ liệu: {UNLABELED_DATA_PATH}")
        print("Tạo file mẫu để demo...")
        os.makedirs(os.path.dirname(UNLABELED_DATA_PATH), exist_ok=True)
        with open(UNLABELED_DATA_PATH, "w", encoding="utf-8") as f:
            f.write("Trời ơi hôm nay đi làm vui quá kkkk\n")
            f.write("Thằng lol này ngáo vãi chưởng\n")
            f.write("Mình đang cảm thấy rất hoang mang về tương lai\n")
            
    # Đọc dữ liệu
    with open(UNLABELED_DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    print(f"Đã tải {len(lines)} câu dữ liệu không gán nhãn.")
    
    # Dataset
    dataset = Dataset.from_dict({"text": lines})
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Preprocess
    def tokenize_function(examples):
        # Word segment nếu cần, ở đây giả sử text đã được xử lý hoặc model tự lo
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Data Collator cho MLM (Mặc định mask 15% token)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Model
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3, # Thông thường DAPT chạy 3-10 epochs tùy data size
        per_device_train_batch_size=16,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=2e-5,
        fp16=False # Đổi thành True nếu dùng Colab T4
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )
    
    print("\nBắt đầu Pre-training...")
    trainer.train()
    
    # Lưu model hoàn chỉnh
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Đã lưu model DAPT tại: {OUTPUT_DIR}")
    print("Sau khi DAPT xong, bạn có thể đổi MODEL_NAME = './models/phobert-dapt' trong test.py để fine-tune tiếp.")

if __name__ == "__main__":
    run_dapt()
