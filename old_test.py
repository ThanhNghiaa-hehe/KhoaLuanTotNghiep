# =============================================================================
# FULL PIPELINE — Nhận diện cảm xúc tiếng Việt (UIT-VSMEC)
# Model chính: ViDeBERTa — tối ưu để chạy LOCAL trên RTX 2050 4GB VRAM
# Khóa luận: "Khảo sát các mô hình tiền huấn luyện cho bài toán
#             nhận diện cảm xúc tiếng Việt trên mạng xã hội"
# =============================================================================
#
# Cài đặt local trong VS Code / Terminal:
#   python -m venv .venv
#   .venv\Scripts\activate              # Windows PowerShell: .\.venv\Scripts\Activate.ps1
#   pip install --upgrade pip
#   pip install -r requirements.txt
#
# Chạy:
#   python test.py
#
# Ghi chú quan trọng:
# - RTX 2050 chỉ có 4GB VRAM, vì vậy batch vật lý nhỏ nhưng dùng
#   gradient_accumulation_steps để giữ effective batch size = 16.
# - ViDeBERTa dùng PyVi word segmentation theo hướng dẫn repo gốc.
# - classification_report PHẢI truyền labels=LABEL_NAMES để không bị lệch tên nhãn.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 0 — CỐ ĐỊNH SEED (phải đặt TRƯỚC KHI import bất kỳ thứ gì)
# ─────────────────────────────────────────────────────────────────────────────
import os

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import torch

RANDOM_STATE = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_STATE)
print(f"✅ Seed đã được cố định: {RANDOM_STATE}")


# ─────────────────────────────────────────────────────────────────────────────
# THƯ VIỆN
# ─────────────────────────────────────────────────────────────────────────────
import inspect
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)


# =============================================================================
# CẤU HÌNH — CHỈ CẦN THAY ĐỔI Ở ĐÂY KHI ĐỔI MODEL
# =============================================================================

MODEL_NAME = "FPTAI/velectra-base-discriminator-cased"
# MODEL_NAME = "vinai/phobert-base-v2"
# MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "xlm-roberta-base"
# MODEL_NAME = "FPTAI/vibert-base-cased"
# MODEL_NAME = "FPTAI/velectra-base-discriminator-cased"

# ── Hyperparameter chuẩn trong khóa luận ─────────────────────────────────
MAX_LENGTH = 128
EFFECTIVE_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01

# ── Preset tối ưu riêng cho ViDeBERTa ─────────────────────────────────────
# RTX 2050 4GB: batch vật lý nhỏ, tích lũy gradient để đạt effective batch=16.
# Weighted CE nhẹ hơn Focal Loss; mục tiêu là bắt buộc học đủ 7 nhãn, không sập
# về các lớp đa số như lần chạy Macro-F1 ~0.19.
VIDEBERTA_LEARNING_RATE = 3e-5
VIDEBERTA_EPOCHS = 15
VIDEBERTA_WARMUP_RATIO = 0.10
VIDEBERTA_PHYSICAL_BATCH_SIZE = 4
VIDEBERTA_GRAD_ACCUM = max(1, EFFECTIVE_BATCH_SIZE // VIDEBERTA_PHYSICAL_BATCH_SIZE)
VIDEBERTA_LABEL_SMOOTHING = 0.05
VIDEBERTA_USE_CLASS_WEIGHTS = True

# ── Cấu hình chuẩn cho các model khác nếu cần chạy lại ────────────────────
STANDARD_LEARNING_RATE = 2e-5
STANDARD_EPOCHS = 5
STANDARD_WARMUP_STEPS = 150
STANDARD_PHYSICAL_BATCH_SIZE = 16
STANDARD_GRAD_ACCUM = 1

# ── Tự động phát hiện model type ─────────────────────────────────────────
IS_PHOBERT = "phobert" in MODEL_NAME.lower()
IS_VIDEBERTA = "videberta" in MODEL_NAME.lower() or "deberta-v3" in MODEL_NAME.lower()

if IS_VIDEBERTA:
    LEARNING_RATE = VIDEBERTA_LEARNING_RATE
    EPOCHS = VIDEBERTA_EPOCHS
    BATCH_SIZE = VIDEBERTA_PHYSICAL_BATCH_SIZE
    GRADIENT_ACCUMULATION_STEPS = VIDEBERTA_GRAD_ACCUM
    WARMUP_RATIO = VIDEBERTA_WARMUP_RATIO
    WARMUP_STEPS = 0
else:
    LEARNING_RATE = STANDARD_LEARNING_RATE
    EPOCHS = STANDARD_EPOCHS
    BATCH_SIZE = STANDARD_PHYSICAL_BATCH_SIZE
    GRADIENT_ACCUMULATION_STEPS = STANDARD_GRAD_ACCUM
    WARMUP_RATIO = 0.0
    WARMUP_STEPS = STANDARD_WARMUP_STEPS

# ── Precision phù hợp local RTX 2050 ──────────────────────────────────────
# ViDeBERTa từng dễ lỗi NaN/Half-Float khi fp16; ưu tiên fp32 + gradient checkpointing.
# Nếu vẫn OOM trên RTX 2050, giảm VIDEBERTA_PHYSICAL_BATCH_SIZE từ 4 xuống 2.
use_fp16 = torch.cuda.is_available() and not IS_VIDEBERTA
use_bf16 = False

print(f"\n📌 Model      : {MODEL_NAME}")
print(f"   PhoBERT    : {IS_PHOBERT}")
print(f"   ViDeBERTa  : {IS_VIDEBERTA}")
print(f"   CUDA       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU        : {torch.cuda.get_device_name(0)}")
print(f"   fp16={use_fp16}, bf16={use_bf16}")
print(f"   batch/device={BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION_STEPS}, effective_batch={BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

# ── Thư mục lưu kết quả ───────────────────────────────────────────────────
OUTPUT_DIR = f"./results/{MODEL_NAME.replace('/', '_')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# BLOCK 1 — DATA PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("BLOCK 1 — DATA PREPARATION")
print("=" * 70)

# ── 1.1 Ánh xạ nhãn ──────────────────────────────────────────────────────
LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]

label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}

print("label2id :", label2id)
print("id2label :", id2label)

# ── 1.2 Tải & gộp dataset ────────────────────────────────────────────────
print("\n[1] Đang tải dataset tridm/UIT-VSMEC ...")
raw = load_dataset("tridm/UIT-VSMEC")

frames = [split_data.to_pandas() for split_data in raw.values()]
full_df = pd.concat(frames, ignore_index=True)

full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
full_df["labels"] = full_df["labels"].map(label2id)
full_df.dropna(subset=["labels"], inplace=True)
full_df["labels"] = full_df["labels"].astype(int)

print(f"Tổng số mẫu sau khi gộp: {len(full_df)}")
print("Phân phối nhãn:\n", full_df["labels"].value_counts().sort_index())

# ── 1.3 Chia lại 70 / 10 / 20 (stratified) ──────────────────────────────
train_val_df, test_df = train_test_split(
    full_df,
    test_size=0.20,
    stratify=full_df["labels"],
    random_state=RANDOM_STATE,
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.125,
    stratify=train_val_df["labels"],
    random_state=RANDOM_STATE,
)

print("\nSau khi chia lại:")
print(f"  Train : {len(train_df):>5}  ({len(train_df) / len(full_df) * 100:.1f}%)")
print(f"  Val   : {len(val_df):>5}  ({len(val_df) / len(full_df) * 100:.1f}%)")
print(f"  Test  : {len(test_df):>5}  ({len(test_df) / len(full_df) * 100:.1f}%)")

# ── 1.4 Tiền xử lý văn bản ───────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    - PhoBERT   : word segmentation bằng underthesea
    - ViDeBERTa : word segmentation bằng PyVi theo repo/paper ViDeBERTa
    - Các model khác: giữ nguyên văn bản thô
    """
    if IS_PHOBERT:
        from underthesea import word_tokenize

        return word_tokenize(str(text), format="text")
    if IS_VIDEBERTA:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(str(text))
    return str(text)


print(f"\n[2] Tiền xử lý văn bản cho model: {MODEL_NAME} ...")
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["text"] = train_df["text"].apply(preprocess_text)
val_df["text"] = val_df["text"].apply(preprocess_text)
test_df["text"] = test_df["text"].apply(preprocess_text)

# ── 1.5 Chuyển sang HuggingFace DatasetDict ──────────────────────────────
def df_to_hf(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "text": df["text"].tolist(),
        "labels": df["labels"].tolist(),
    })


dataset = DatasetDict({
    "train": df_to_hf(train_df),
    "validation": df_to_hf(val_df),
    "test": df_to_hf(test_df),
})

print("\n[3] DatasetDict đã sẵn sàng:")
print(dataset)


# =============================================================================
# BLOCK 2 — TOKENIZATION & DATASET MAPPING
# =============================================================================
print("\n" + "=" * 70)
print("BLOCK 2 — TOKENIZATION & DATASET MAPPING")
print("=" * 70)

print(f"\n[4] Tải tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("[5] Tokenizing dataset ...")
tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=256,
    remove_columns=["text"],
)
tokenized_dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

print("Tokenized dataset:")
print(tokenized_dataset)


# =============================================================================
# BLOCK 3 — TRAINING PIPELINE (ViDeBERTa: Weighted Cross Entropy ổn định)
# =============================================================================
print("\n" + "=" * 70)
print("BLOCK 3 — TRAINING PIPELINE")
print("=" * 70)

import torch.nn as nn

# ── 3.1 compute_metrics ──────────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {
        "accuracy": acc["accuracy"],
        "macro_f1": macro_f1["f1"],
    }


# ── 3.2 Tính class weights cho ViDeBERTa ─────────────────────────────────
def build_class_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts().sort_index().reindex(range(len(LABEL_NAMES)), fill_value=1).astype(float)
    # sqrt inverse frequency: đủ kéo lớp hiếm lên, không quá gắt như inverse frequency thuần.
    weights = np.sqrt(counts.max() / counts.values)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class_weights = build_class_weights(train_df["labels"])
print("\nClass weights:")
for idx, weight in enumerate(class_weights.tolist()):
    print(f"  {id2label[idx]:>9}: {weight:.4f}")

# ── 3.3 Khởi tạo model ───────────────────────────────────────────────────
print(f"\n[6] Tải model: {MODEL_NAME}")

model_kwargs = dict(
    num_labels=len(LABEL_NAMES),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
if IS_VIDEBERTA:
    model_kwargs["attn_implementation"] = "eager"
    model_kwargs["torch_dtype"] = torch.float32

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **model_kwargs)

if IS_VIDEBERTA:
    model = model.float()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("✅ ViDeBERTa: fp32 + gradient checkpointing + eager attention")

# ── 3.4 Custom Trainer cho ViDeBERTa ─────────────────────────────────────
class WeightedLossTrainer(Trainer):
    """
    Trainer ổn định cho ViDeBERTa:
    - CrossEntropyLoss có class weights để mô hình học đủ 7 nhãn.
    - label_smoothing nhẹ để giảm overfit trên VSMEC nhỏ.
    - guard NaN/Inf trong logits/loss.
    """

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.custom_label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits").float()

        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.custom_label_smoothing,
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        if not torch.isfinite(loss):
            loss = torch.zeros(1, requires_grad=True, device=logits.device)

        return (loss, outputs) if return_outputs else loss


TrainerClass = WeightedLossTrainer if IS_VIDEBERTA else Trainer
print(f"Trainer đang sử dụng: {'WeightedLossTrainer' if IS_VIDEBERTA else 'Trainer (standard)'}")

# ── 3.5 TrainingArguments ────────────────────────────────────────────────
eval_strategy_key = (
    "eval_strategy"
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
    else "evaluation_strategy"
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=max(1, BATCH_SIZE * 2),
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    optim="adamw_torch",
    **{eval_strategy_key: "epoch"},
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    seed=RANDOM_STATE,
    data_seed=RANDOM_STATE,
    fp16=use_fp16,
    bf16=use_bf16,
    lr_scheduler_type="cosine" if IS_VIDEBERTA else "linear",
    warmup_ratio=WARMUP_RATIO,
    warmup_steps=WARMUP_STEPS,
    max_grad_norm=1.0,
    disable_tqdm=False,
)

# ── 3.6 Khởi tạo cấu hình tham số Trainer ───────────────────────────────
trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

if IS_VIDEBERTA:
    trainer_kwargs["class_weights"] = class_weights if VIDEBERTA_USE_CLASS_WEIGHTS else None
    trainer_kwargs["label_smoothing"] = VIDEBERTA_LABEL_SMOOTHING

if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = TrainerClass(**trainer_kwargs)

print("\n[7] Bắt đầu huấn luyện mô hình...")
try:
    trainer.train()
except torch.cuda.OutOfMemoryError as exc:
    print("\n❌ CUDA Out Of Memory trên GPU 4GB.")
    print("Cách sửa nhanh: đổi VIDEBERTA_PHYSICAL_BATCH_SIZE = 2 rồi chạy lại.")
    raise exc
print(f"\n✅ Huấn luyện xong! Best model lưu tại: {OUTPUT_DIR}")

trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))


# =============================================================================
# BLOCK 4 — EVALUATION & VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("BLOCK 4 — EVALUATION & VISUALIZATION")
print("=" * 70)

# ── 4.1 Predict trên tập TEST ────────────────────────────────────────────
print("\n[8] Đang predict trên tập TEST ...")
test_output = trainer.predict(tokenized_dataset["test"])
y_pred_logits = test_output.predictions
y_true = test_output.label_ids
y_pred = np.argmax(y_pred_logits, axis=-1)

y_true_names = [id2label[int(i)] for i in y_true]
y_pred_names = [id2label[int(i)] for i in y_pred]

# ── 4.2 Classification Report ────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (tập TEST)")
print("=" * 60)
report_str = classification_report(
    y_true_names,
    y_pred_names,
    labels=LABEL_NAMES,          # BẮT BUỘC: tránh sklearn sort alphabet làm lệch tên nhãn
    target_names=LABEL_NAMES,
    digits=4,
    zero_division=0,
)
print(report_str)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch/device: {BATCH_SIZE}\n")
    f.write(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}\n")
    f.write(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n")
    f.write(f"Loss: {'Weighted Cross Entropy' if IS_VIDEBERTA else 'Cross Entropy'}\n\n")
    f.write(report_str)
print(f"✅ Đã lưu: {report_path}")

# ── 4.3 Confusion Matrix ─────────────────────────────────────────────────
cm = confusion_matrix(y_true_names, y_pred_names, labels=LABEL_NAMES)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    linewidths=0.5,
)
plt.title(f"Confusion Matrix\n{MODEL_NAME}", fontsize=13, fontweight="bold", pad=12)
plt.ylabel("Nhãn thực tế", fontsize=11)
plt.xlabel("Nhãn dự đoán", fontsize=11)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"✅ Đã lưu: {cm_path}")

# ── 4.4 F1-Score Bar Chart ───────────────────────────────────────────────
f1_per_class = f1_score(y_true_names, y_pred_names, labels=LABEL_NAMES, average=None, zero_division=0)
macro_f1 = f1_score(y_true_names, y_pred_names, labels=LABEL_NAMES, average="macro", zero_division=0)

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
ax.set_title(f"F1-Score theo từng nhãn cảm xúc\n{MODEL_NAME}", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
f1_chart_path = os.path.join(OUTPUT_DIR, "f1_per_class.png")
plt.savefig(f1_chart_path, dpi=150)
plt.show()
print(f"✅ Đã lưu: {f1_chart_path}")


# =============================================================================
# BLOCK 5 — ERROR ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("BLOCK 5 — ERROR ANALYSIS")
print("=" * 70)

# ── 5.1 Xây dựng DataFrame kết quả TEST ──────────────────────────────────
test_texts = test_df["text"].tolist()

result_df = pd.DataFrame({
    "Câu văn thực tế": test_texts,
    "Nhãn thực tế": y_true_names,
    "Nhãn dự đoán": y_pred_names,
    "Đúng/Sai": ["✓" if t == p else "✗" for t, p in zip(y_true_names, y_pred_names)],
})

# ── 5.2 Lọc mẫu sai ─────────────────────────────────────────────────────
error_df = result_df[result_df["Đúng/Sai"] == "✗"].drop(columns=["Đúng/Sai"]).reset_index(drop=True)

n_total = len(result_df)
n_correct = (result_df["Đúng/Sai"] == "✓").sum()
n_errors = len(error_df)

print(f"\n[9] Tổng số mẫu TEST  : {n_total}")
print(f"    Số mẫu đúng       : {n_correct}")
print(f"    Số mẫu sai (lỗi)  : {n_errors}  ({n_errors / n_total * 100:.2f}%)")

# ── 5.3 Xuất CSV ─────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "error_analysis.csv")
error_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n✅ Đã lưu {n_errors} mẫu sai vào: {csv_path}")

# ── 5.4 In 10 mẫu sai điển hình nhất ────────────────────────────────────
pair_counts = (
    error_df.groupby(["Nhãn thực tế", "Nhãn dự đoán"])
    .size()
    .reset_index(name="Số lần nhầm")
    .sort_values("Số lần nhầm", ascending=False)
)

print("\n📊 Thống kê cặp nhãn bị nhầm lẫn nhiều nhất:")
if len(pair_counts) > 0:
    print(pair_counts.head(10).to_string(index=False))
else:
    print("Không có mẫu dự đoán sai.")

top_pairs = pair_counts.head(10)[["Nhãn thực tế", "Nhãn dự đoán"]].values if len(pair_counts) > 0 else []
sample_rows = []

for true_label, pred_label in top_pairs:
    subset = error_df[(error_df["Nhãn thực tế"] == true_label) & (error_df["Nhãn dự đoán"] == pred_label)]
    if len(subset) > 0:
        sample_rows.append(subset.iloc[0])

sample_df = pd.DataFrame(sample_rows).reset_index(drop=True)

print("\n📝 Ví dụ câu bị dự đoán sai (1 mẫu mỗi cặp nhãn phổ biến):")
print("-" * 70)
for _, row in sample_df.iterrows():
    print(f"  📌 Câu     : {row['Câu văn thực tế']}")
    print(f"     Thực tế : {row['Nhãn thực tế']}")
    print(f"     Dự đoán : {row['Nhãn dự đoán']} ❌")
    print("-" * 70)


# =============================================================================
# TỔNG KẾT
# =============================================================================
print("\n" + "=" * 70)
print("✅ PIPELINE HOÀN TẤT")
print(f"   Model      : {MODEL_NAME}")
print(f"   Macro-F1   : {macro_f1:.4f}")
print(f"   Output dir : {OUTPUT_DIR}/")
print("     ├── classification_report.txt")
print("     ├── confusion_matrix.png")
print("     ├── f1_per_class.png")
print("     ├── error_analysis.csv")
print("     └── best_model/")
print("=" * 70)
