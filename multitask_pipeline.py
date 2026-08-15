import os
import numpy as np
import pandas as pd
import torch
import torch.serialization
torch.serialization.add_safe_globals([set])
if getattr(torch, "_is_patched_load", False) is False:
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
    torch._is_patched_load = True
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, EarlyStoppingCallback, AutoModel
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- ĐỊNH NGHĨA MODEL TẠI CHỖ ĐỂ DỄ CHẠY TRÊN COLAB/KAGGLE ---
class MultiTaskTransformer(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.num_emotion_labels = 7
        self.num_sentiment_labels = 3
        
        # SỬA LỖI CHÍNH XÁC 100%: Dùng from_pretrained để load trọng số đã học
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.transformer.config
        
        # Two classification heads
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.emotion_classifier = torch.nn.Linear(self.config.hidden_size, self.num_emotion_labels)
        self.sentiment_classifier = torch.nn.Linear(self.config.hidden_size, self.num_sentiment_labels)
        
        # Loss functions
        self.emotion_loss_fct = torch.nn.CrossEntropyLoss()
        self.sentiment_loss_fct = torch.nn.CrossEntropyLoss()
        
        # Trọng số cho sentiment task (thường thấp hơn main task)
        self.sentiment_lambda = 0.3

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,          # labels = emotion_labels
        sentiment_labels=None, # sentiment_labels = auxiliary labels
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        # Sử dụng output của thẻ [CLS] cho sequence classification
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback nếu model không có pooler (như RoBERTa)
            pooled_output = outputs[0][:, 0, :]

        pooled_output = self.dropout(pooled_output)
        
        # Tính logits cho cả 2 task
        emotion_logits = self.emotion_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)

        total_loss = None
        if labels is not None and sentiment_labels is not None:
            emotion_loss = self.emotion_loss_fct(emotion_logits.view(-1, self.num_emotion_labels), labels.view(-1))
            sentiment_loss = self.sentiment_loss_fct(sentiment_logits.view(-1, self.num_sentiment_labels), sentiment_labels.view(-1))
            
            # Kết hợp loss
            total_loss = emotion_loss + self.sentiment_lambda * sentiment_loss

        if not return_dict:
            output = (emotion_logits, sentiment_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        output_dict = {
            "loss": total_loss,
            "logits": emotion_logits, # Trả về main logits để Trainer đánh giá metric
            "sentiment_logits": sentiment_logits,
        }
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            output_dict["hidden_states"] = outputs.hidden_states
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            output_dict["attentions"] = outputs.attentions
            
        return output_dict


# --- CONFIG ---
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
OUTPUT_DIR = f"./results/multitask_{MODEL_NAME.replace('/', '_')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}

def map_emotion_to_sentiment(emotion_id):
    """
    0: Enjoyment -> 0 (Positive)
    1, 2, 3, 4: Sadness, Anger, Fear, Disgust -> 1 (Negative)
    5, 6: Surprise, Other -> 2 (Neutral)
    """
    if emotion_id == 0:
        return 0
    elif emotion_id in [1, 2, 3, 4]:
        return 1
    else:
        return 2

print("="*60)
print("BẮT ĐẦU PIPELINE MULTI-TASK LEARNING")
print("="*60)

# 1. Load Data
from datasets import load_dataset
raw = load_dataset("tridm/UIT-VSMEC")
frames = [split_data.to_pandas() for split_data in raw.values()]
full_df = pd.concat(frames, ignore_index=True)
full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
full_df["labels"] = full_df["labels"].map(label2id)
full_df.dropna(subset=["labels"], inplace=True)
full_df["labels"] = full_df["labels"].astype(int)

# Thêm nhãn phụ (Sentiment)
full_df["sentiment_labels"] = full_df["labels"].apply(map_emotion_to_sentiment)

print("Phân phối nhãn Sentiment:")
print("  0 (Positive):", len(full_df[full_df['sentiment_labels'] == 0]))
print("  1 (Negative):", len(full_df[full_df['sentiment_labels'] == 1]))
print("  2 (Neutral) :", len(full_df[full_df['sentiment_labels'] == 2]))

# Split data
train_val_df, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["labels"], random_state=RANDOM_STATE)
train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df["labels"], random_state=RANDOM_STATE)

# Preprocess
def preprocess_text(text: str) -> str:
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")

train_df["text"] = train_df["text"].apply(preprocess_text)
val_df["text"] = val_df["text"].apply(preprocess_text)
test_df["text"] = test_df["text"].apply(preprocess_text)

# Chuyển sang Dataset
def df_to_hf(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "text": df["text"].tolist(),
        "labels": df["labels"].tolist(),
        "sentiment_labels": df["sentiment_labels"].tolist(),
    })

dataset = DatasetDict({
    "train": df_to_hf(train_df),
    "validation": df_to_hf(val_df),
    "test": df_to_hf(test_df),
})

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch")

# Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    # eval_pred.predictions là tuple (emotion_logits, sentiment_logits) do model trả về 2 keys
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    
    # eval_pred.label_ids là tuple (labels, sentiment_labels) do forward nhận 2 tham số labels
    labels = eval_pred.label_ids[0] if isinstance(eval_pred.label_ids, tuple) else eval_pred.label_ids
    
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "macro_f1": macro_f1["f1"]}

# Khởi tạo mô hình Multi-Task
print(f"\nKhởi tạo mô hình Multi-Task từ {MODEL_NAME}...")
model = MultiTaskTransformer(MODEL_NAME).float()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1", # Trainer tự thêm eval_
    save_total_limit=1,
    report_to="none",
    fp16=False, bf16=False,
    remove_unused_columns=False, # Quan trọng: Giữ lại sentiment_labels
)

class MultiTaskTrainer(Trainer):
    # Ghi đè compute_loss để báo cho HF Trainer cách lấy loss từ dictionary trả về
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\nBắt đầu huấn luyện...")
trainer.train()

print("\nĐánh giá trên tập TEST...")
test_out = trainer.predict(tokenized_dataset["test"])
print(f"✅ Độ chính xác tổng quát (Accuracy): {test_out.metrics['test_accuracy']:.4f}")
print(f"✅ Trung bình Macro-F1: {test_out.metrics['test_macro_f1']:.4f}\n")

# Bóc tách tuple predictions (vì model MultiTask trả về cả 2 logits)
emotion_logits, sentiment_logits = test_out.predictions

# Lấy nhãn thực tế từ kết quả predict (Trainer đã tự động convert sang numpy array)
emotion_labels, sentiment_labels_true = test_out.label_ids

# Tính dự đoán
emotion_preds = np.argmax(emotion_logits, axis=-1)
sentiment_preds = np.argmax(sentiment_logits, axis=-1)

from sklearn.metrics import classification_report

print("="*60)
print("BÁO CÁO NHIỆM VỤ CHÍNH (7 CẢM XÚC)")
print("="*60)
emotion_report_str = classification_report(emotion_labels, emotion_preds, target_names=LABEL_NAMES, digits=4)
print(emotion_report_str)

print("="*60)
print("BÁO CÁO NHIỆM VỤ PHỤ (3 SẮC THÁI)")
print("="*60)
sentiment_names = ["Positive (0)", "Negative (1)", "Neutral (2)"]
sentiment_report_str = classification_report(sentiment_labels_true, sentiment_preds, target_names=sentiment_names, digits=4)
print(sentiment_report_str)

# =============================================================================
# LƯU TRỮ DỮ LIỆU ĐỂ BÁO CÁO (TRÁNH BỊ MẤT KHI TẮT COLAB)
# =============================================================================
print("\n💾 Đang xuất các file dữ liệu thô...")

# 1. Lưu Classification Report ra file Text
with open(os.path.join(OUTPUT_DIR, "classification_report_emotion.txt"), "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_NAME} (Multi-Task)\n\n" + emotion_report_str)
with open(os.path.join(OUTPUT_DIR, "classification_report_sentiment.txt"), "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_NAME} (Multi-Task)\n\n" + sentiment_report_str)

# 2. Lưu mảng Logits & Labels để sau này có thể dùng làm Ensemble hoặc ROC/AUC
np.save(os.path.join(OUTPUT_DIR, "test_emotion_logits.npy"), emotion_logits)
np.save(os.path.join(OUTPUT_DIR, "test_emotion_labels.npy"), emotion_labels)
np.save(os.path.join(OUTPUT_DIR, "test_sentiment_logits.npy"), sentiment_logits)

# 3. Xuất Error Analysis CSV (Cực kỳ quan trọng để chèn vào báo cáo Khóa luận)
test_texts = test_df["text"].tolist()
y_true_names = [LABEL_NAMES[i] for i in emotion_labels]
y_pred_names = [LABEL_NAMES[i] for i in emotion_preds]

result_df = pd.DataFrame({
    "Câu văn thực tế": test_texts,
    "Nhãn thực tế (Cảm xúc)": y_true_names,
    "Nhãn dự đoán (Cảm xúc)": y_pred_names,
    "Đúng/Sai": ["✓" if t == p else "✗" for t, p in zip(y_true_names, y_pred_names)],
})
error_df = result_df[result_df["Đúng/Sai"] == "✗"].drop(columns=["Đúng/Sai"]).reset_index(drop=True)
csv_path = os.path.join(OUTPUT_DIR, "error_analysis_emotion.csv")
error_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Đã lưu {len(error_df)} câu bị đoán sai vào: {csv_path}")

# =============================================================================
# VẼ BIỂU ĐỒ TRỰC QUAN (CONFUSION MATRIX & F1 BAR CHART)
# =============================================================================
print("\n🎨 Đang vẽ và lưu các biểu đồ đánh giá...")

# 1. Confusion Matrix cho 7 Cảm xúc
cm_emotion = confusion_matrix(emotion_labels, emotion_preds, labels=list(range(7)))
plt.figure(figsize=(9, 7))
sns.heatmap(cm_emotion, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, linewidths=0.5)
plt.title(f"Confusion Matrix (Emotion)\n{MODEL_NAME} - Multi-Task", fontsize=13, fontweight="bold", pad=12)
plt.ylabel("Nhãn Thực Tế")
plt.xlabel("Nhãn Dự Đoán")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_emotion.png")
plt.savefig(cm_path, dpi=150)
plt.close()

# 2. F1-Score Bar Chart cho 7 Cảm xúc
f1_per_class = f1_score(emotion_labels, emotion_preds, labels=list(range(7)), average=None)
macro_f1 = f1_score(emotion_labels, emotion_preds, average="macro")
sorted_idx = np.argsort(f1_per_class)[::-1]
sorted_names = [LABEL_NAMES[i] for i in sorted_idx]
sorted_f1 = f1_per_class[sorted_idx]
colors = plt.cm.RdYlGn(sorted_f1)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(sorted_names, sorted_f1, color=colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, sorted_f1):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=10)
ax.axvline(macro_f1, color="navy", linestyle="--", linewidth=1.5, label=f"Macro-F1 = {macro_f1:.4f}")
ax.set_xlim(0, 1.08)
ax.set_xlabel("F1-Score", fontsize=12)
ax.set_title(f"F1-Score theo từng nhãn cảm xúc\n{MODEL_NAME} - Multi-Task", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
f1_chart_path = os.path.join(OUTPUT_DIR, "f1_per_class_emotion.png")
plt.savefig(f1_chart_path, dpi=150)
plt.close()

# 3. Confusion Matrix cho 3 Sắc thái (Sentiment)
cm_sentiment = confusion_matrix(sentiment_labels_true, sentiment_preds, labels=[0, 1, 2])
plt.figure(figsize=(7, 5))
sns.heatmap(cm_sentiment, annot=True, fmt="d", cmap="Oranges", xticklabels=sentiment_names, yticklabels=sentiment_names, linewidths=0.5)
plt.title(f"Confusion Matrix (Sentiment)\n{MODEL_NAME} - Multi-Task", fontsize=13, fontweight="bold", pad=12)
plt.ylabel("Thực Tế")
plt.xlabel("Dự Đoán")
plt.tight_layout()
cm_sentiment_path = os.path.join(OUTPUT_DIR, "confusion_matrix_sentiment.png")
plt.savefig(cm_sentiment_path, dpi=150)
plt.close()

# 4. F1-Score Bar Chart cho 3 Sắc thái (Sentiment)
f1_per_class_sent = f1_score(sentiment_labels_true, sentiment_preds, labels=[0, 1, 2], average=None)
macro_f1_sent = f1_score(sentiment_labels_true, sentiment_preds, average="macro")
sorted_idx_sent = np.argsort(f1_per_class_sent)[::-1]
sorted_names_sent = [sentiment_names[i] for i in sorted_idx_sent]
sorted_f1_sent = f1_per_class_sent[sorted_idx_sent]
colors_sent = plt.cm.autumn_r(sorted_f1_sent) # Dùng dải màu cam/vàng/đỏ cho phân loại Sentiment

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(sorted_names_sent, sorted_f1_sent, color=colors_sent, edgecolor="white", height=0.5)
for bar, val in zip(bars, sorted_f1_sent):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=10)
ax.axvline(macro_f1_sent, color="brown", linestyle="--", linewidth=1.5, label=f"Macro-F1 = {macro_f1_sent:.4f}")
ax.set_xlim(0, 1.08)
ax.set_xlabel("F1-Score", fontsize=12)
ax.set_title(f"F1-Score theo Sắc thái\n{MODEL_NAME} - Multi-Task", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
f1_sent_chart_path = os.path.join(OUTPUT_DIR, "f1_per_class_sentiment.png")
plt.savefig(f1_sent_chart_path, dpi=150)
plt.close()

print(f"✅ Đã lưu Confusion Matrix (Emotion): {cm_path}")
print(f"✅ Đã lưu F1 Bar Chart (Emotion): {f1_chart_path}")
print(f"✅ Đã lưu Confusion Matrix (Sentiment): {cm_sentiment_path}")
print(f"✅ Đã lưu F1 Bar Chart (Sentiment): {f1_sent_chart_path}")
print(f"\n📂 Mô hình Multi-Task đã được lưu tại: {OUTPUT_DIR}")
