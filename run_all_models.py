import os
# Đã xóa HF_ENDPOINT để dùng lại server chính thức của HuggingFace
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"        # Tắt hf_transfer (nguyên nhân gây treo 64MB)
os.environ["HF_HUB_DISABLE_XET"] = "1"               # Tắt Xet
os.environ["HF_HUB_ENABLE_EMERGENCY_RETRY"] = "1"    # Tự động thử lại khi rớt mạng
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.serialization
torch.serialization.add_safe_globals([set])

# Sửa lỗi "cannot import name 'VideoReader' from 'torchvision.io'" của thư viện datasets trên Colab mới
import torchvision.io
if not hasattr(torchvision.io, 'VideoReader'):
    class DummyVideoReader: pass
    torchvision.io.VideoReader = DummyVideoReader

if getattr(torch, "_is_patched_load", False) is False:
    _orig_load = torch.serialization.load # Lấy hàm gốc sạch sẽ để không bao giờ bị dính vòng lặp cũ
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
    torch._is_patched_load = True
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# =============================================================================
# CẤU HÌNH CƠ BẢN
# =============================================================================
RANDOM_STATE = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_STATE)

MAX_LENGTH   = 128
BATCH_SIZE   = 16
WEIGHT_DECAY  = 0.01

LOSS_TYPE = "focal"      # "standard", "weighted", "focal"
LABEL_SMOOTHING = 0.1
USE_AUGMENTATION = True

RUN_ALL_MODELS = True    # Bật True để treo máy chạy tất cả, tắt False để chạy thử 1 model
SINGLE_MODEL_NAME = "vinai/phobert-base-v2"  # Mô hình sẽ chạy nếu RUN_ALL_MODELS = False

# Danh sách 6 mô hình cần chạy tự động
MODELS_TO_RUN = [
    "vinai/phobert-base-v2",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "FPTAI/vibert-base-cased",
    "FPTAI/velectra-base-discriminator-cased",
    "Fsoft-AIC/videberta-xsmall"
]

LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}

# =============================================================================
# CHUẨN BỊ DỮ LIỆU (Chỉ chạy 1 lần)
# =============================================================================
print("\n[1] Đang tải dataset tridm/UIT-VSMEC ...")
raw = load_dataset("tridm/UIT-VSMEC")
frames = [split_data.to_pandas() for split_data in raw.values()]
full_df = pd.concat(frames, ignore_index=True)

full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
full_df["labels"] = full_df["labels"].map(label2id)
full_df.dropna(subset=["labels"], inplace=True)
full_df["labels"] = full_df["labels"].astype(int)

train_val_df, test_df = train_test_split(
    full_df, test_size=0.20,
    stratify=full_df["labels"], random_state=RANDOM_STATE
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.125,
    stratify=train_val_df["labels"], random_state=RANDOM_STATE
)

if USE_AUGMENTATION:
    import os
    csv_path = "./TaiLieu/augmented_train_upgrade.csv"
    if os.path.exists(csv_path):
        print(f"\n[Augmentation] Đã tìm thấy file '{csv_path}', tiến hành nạp dữ liệu Tăng cường (PRO)...")
        train_df = pd.read_csv(csv_path)
    else:
        try:
            from augmentation_upgrade import augment_minority_classes_upgrade
            train_df = augment_minority_classes_upgrade(train_df, target_count=1000, save_path=csv_path)
        except ImportError as e:
            print(f"⚠️ CẢNH BÁO: Bỏ qua bước tăng cường dữ liệu do không tìm thấy file augmentation_upgrade.py ({e})")

train_label_counts = np.bincount(train_df["labels"].values, minlength=7)
class_weights_np = len(train_df) / (7 * train_label_counts)

if LOSS_TYPE in ["weighted", "focal"]:
    print(f"\n[{LOSS_TYPE.upper()} LOSS] Trọng số lớp (Alpha - Inverse Class Frequency):")
else:
    print(f"\n[THỐNG KÊ DỮ LIỆU] Tỷ lệ mẫu phân bổ (Không áp dụng làm trọng số Loss):")

for i, name in enumerate(LABEL_NAMES):
    print(f"  {name:>12s}: {class_weights_np[i]:.4f}  (Số mẫu Train: {train_label_counts[i]})")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

def df_to_hf(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "text":   df["text"].tolist(),
        "labels": df["labels"].tolist(),
    })

hf_dataset = DatasetDict({
    "train":      df_to_hf(train_df),
    "validation": df_to_hf(val_df),
    "test":       df_to_hf(test_df),
})

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "eval_macro_f1": macro_f1["f1"]}

class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if LOSS_TYPE == "standard":
            loss = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), label_smoothing=LABEL_SMOOTHING)
        elif LOSS_TYPE == "weighted":
            loss = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
        elif LOSS_TYPE == "focal":
            ce_loss_unweighted = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), reduction='none')
            pt = torch.exp(-ce_loss_unweighted)
            ce_loss_weighted = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), reduction='none', weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
            loss = ((1 - pt) ** 2.0 * ce_loss_weighted).mean()
            
        return (loss, outputs) if return_outputs else loss

class ViDeBERTaWithMeanPooling(DebertaV2PreTrainedModel):
    _tied_weights_keys = []
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=return_dict)
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        mean_pooled = self.layernorm(mean_pooled)
        mean_pooled = self.dropout(mean_pooled)
        logits = self.classifier(mean_pooled)

        loss = None
        if labels is not None:
            if LOSS_TYPE == "standard":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), label_smoothing=LABEL_SMOOTHING)
            elif LOSS_TYPE == "weighted":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
            elif LOSS_TYPE == "focal":
                ce_loss_unweighted = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction='none')
                pt = torch.exp(-ce_loss_unweighted)
                ce_loss_weighted = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction='none', weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
                loss = ((1 - pt) ** 2.0 * ce_loss_weighted).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# =============================================================================
# HÀM CHẠY THỰC NGHIỆM CHO 1 MODEL
# =============================================================================
def run_experiment(model_name):
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN: {model_name}")
    print(f"{'='*60}")
    
    OUTPUT_DIR = f"./results/{model_name.replace('/', '_')}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Bật tách từ cho PhoBERT, ViDeBERTa (FPTAI viBERT và vELECTRA dùng syllable-level tokenizer, không dùng tách từ)
    USE_WORD_SEGMENTATION = "phobert" in model_name.lower() or "videberta" in model_name.lower()
    
    # Tái tạo lại dataset text với word segmentation nếu cần
    current_hf_dataset = hf_dataset
    if USE_WORD_SEGMENTATION:
        from underthesea import word_tokenize
        def word_seg_func(example):
            example["text"] = word_tokenize(example["text"], format="text")
            return example
        print(f"[{model_name}] Đang áp dụng Word Segmentation...")
        current_hf_dataset = current_hf_dataset.map(word_seg_func)
    
    # Ép tải trước model bằng GIT CLONE để lách lỗi treo 64MB của thư viện Python
    local_path = f"/content/local_models/{model_name.replace('/', '_')}"
    # Kiểm tra xem model đã được tải hoàn chỉnh chưa (phải có file config.json)
    if not os.path.exists(os.path.join(local_path, "config.json")):
        print(f"\n[{model_name}] Đang tải model qua GIT CLONE (Cách chống đứt cáp an toàn nhất)...")
        os.makedirs("/content/local_models", exist_ok=True)
        
        # Xóa thư mục rác nếu lần tải trước bị đứt gánh giữa chừng
        import shutil
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
            
        # Lấy token Hugging Face để tránh bị chặn 403 Forbidden
        from huggingface_hub import get_token
        token = get_token()
        repo_url = f"https://oauth2:{token}@huggingface.co/{model_name}" if token else f"https://huggingface.co/{model_name}"
        
        # Git clone trực tiếp (sử dụng LFS)
        os.system(f"GIT_LFS_SKIP_SMUDGE=0 git clone {repo_url} {local_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        
    print(f"[{model_name}] Đang Tokenizing dataset ...")
    tokenized_dataset = current_hf_dataset.map(tokenize_fn, batched=True, batch_size=256, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    
    print("\n" + "="*70)
    print("BLOCK 3 — TRAINING PIPELINE")
    print("="*70)
    
    # 2. Khởi tạo Model
    epochs = 10 if "videberta" in model_name.lower() else 5
    lr = 1e-5 if "videberta" in model_name.lower() else 2e-5
    grad_acc = 4 if "videberta" in model_name.lower() else 1
    
    if "videberta" in model_name.lower():
        model = ViDeBERTaWithMeanPooling.from_pretrained(local_path, num_labels=7, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(local_path, num_labels=7, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
        
    model = model.float().to(device)
    
    # 3. Khởi tạo Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,  # Thêm warmup cho các model học ổn định hơn (nhất là ELECTRA)
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
        fp16=False, bf16=False,
    )
    
    # Fallback an toàn: Dùng Trainer gốc nếu là ViDeBERTa hoặc chạy loss standard
    if "videberta" in model_name.lower() or LOSS_TYPE == "standard":
        TrainerClass = Trainer
    else:
        TrainerClass = CustomLossTrainer
    
    try:
        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
    except TypeError:
        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
    
    # 4. Huấn luyện & Đánh giá
    print("\n[6] Bắt đầu huấn luyện...")
    trainer.train()
    
    print("\n[8] Đang đánh giá trên tập TEST ...")
    test_output = trainer.predict(tokenized_dataset["test"])
    y_pred_logits = test_output.predictions
    if isinstance(y_pred_logits, tuple):
        y_pred_logits = y_pred_logits[0]
        
    y_pred = np.argmax(y_pred_logits, axis=-1)
    y_true = test_output.label_ids
    
    np.save(os.path.join(OUTPUT_DIR, "test_logits.npy"), y_pred_logits)
    np.save(os.path.join(OUTPUT_DIR, "test_y_true.npy"), y_true)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (tập TEST)")
    print("="*60)
    report_str = classification_report(y_true, y_pred, labels=list(range(7)), target_names=LABEL_NAMES, digits=4)
    print(report_str)
    
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(report_str)
    print(f"✅ Đã lưu: {report_path}")
        
    y_true_names = [id2label[i] for i in y_true]
    y_pred_names = [id2label[i] for i in y_pred]

    # 4.3 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, linewidths=0.5)
    plt.title(f"Confusion Matrix\n{model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Nhãn thực tế", fontsize=11)
    plt.xlabel("Nhãn dự đoán", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu: {cm_path}")
    
    # 4.4 F1-Score Bar Chart theo từng nhãn
    f1_per_class = f1_score(y_true, y_pred, labels=list(range(7)), average=None)
    macro_f1     = f1_score(y_true, y_pred, average="macro")
    sorted_idx   = np.argsort(f1_per_class)[::-1]
    sorted_names = [LABEL_NAMES[i] for i in sorted_idx]
    sorted_f1    = f1_per_class[sorted_idx]
    colors       = plt.cm.RdYlGn(sorted_f1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(sorted_names, sorted_f1, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, sorted_f1):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=10)
    ax.axvline(macro_f1, color="navy", linestyle="--", linewidth=1.5, label=f"Macro-F1 = {macro_f1:.4f}")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title(f"F1-Score theo từng nhãn cảm xúc\n{model_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    f1_chart_path = os.path.join(OUTPUT_DIR, "f1_per_class.png")
    plt.savefig(f1_chart_path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu: {f1_chart_path}")

    # 4.5 Error Analysis (Xuất CSV)
    print("\n" + "="*70)
    print("BLOCK 5 — ERROR ANALYSIS")
    print("="*70)
    test_texts = test_df["text"].tolist()
    result_df = pd.DataFrame({
        "Câu văn thực tế": test_texts,
        "Nhãn thực tế":    y_true_names,
        "Nhãn dự đoán":    y_pred_names,
        "Đúng/Sai":        ["✓" if t == p else "✗" for t, p in zip(y_true_names, y_pred_names)],
    })
    error_df = result_df[result_df["Đúng/Sai"] == "✗"].drop(columns=["Đúng/Sai"]).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_DIR, "error_analysis.csv")
    error_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    n_total   = len(result_df)
    n_correct = (result_df["Đúng/Sai"] == "✓").sum()
    n_errors  = len(error_df)
    print(f"\n[9] Tổng số mẫu TEST  : {n_total}")
    print(f"    Số mẫu đúng       : {n_correct}")
    print(f"    Số mẫu sai (lỗi)  : {n_errors}  ({n_errors/n_total*100:.2f}%)")
    print(f"\n✅ Đã lưu {n_errors} mẫu sai vào: {csv_path}")

    # In 10 mẫu sai điển hình nhất
    pair_counts = error_df.groupby(["Nhãn thực tế", "Nhãn dự đoán"]).size().reset_index(name="Số lần nhầm").sort_values("Số lần nhầm", ascending=False)
    print("\n📊 Thống kê cặp nhãn bị nhầm lẫn nhiều nhất:")
    print(pair_counts.head(10).to_string(index=False))
    
    top_pairs = pair_counts.head(10)[["Nhãn thực tế", "Nhãn dự đoán"]].values
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

    print("\n" + "="*70)
    print("✅ PIPELINE HOÀN TẤT")
    print(f"   Model        : {model_name}")
    print(f"   Macro-F1     : {macro_f1:.4f}")
    print(f"   Output dir   : {OUTPUT_DIR}/")
    print("="*70)

    # 5. DỌN DẸP BỘ NHỚ GPU - BƯỚC QUAN TRỌNG NHẤT
    print(f"🧹 Đang dọn dẹp bộ nhớ GPU sau khi chạy {model_name}...")
    del trainer
    del model
    del tokenizer
    del tokenized_dataset
    
    # Bắt buộc Python gọi Garbage Collector
    gc.collect()
    
    # Ép PyTorch xả bộ nhớ GPU (VRAM)
    torch.cuda.empty_cache()
    
    print(f"✅ Hoàn tất {model_name}. RAM và GPU đã được giải phóng.")

# =============================================================================
# VÒNG LẶP CHẠY TỰ ĐỘNG HOẶC CHẠY ĐƠN
# =============================================================================
if __name__ == "__main__":
    if RUN_ALL_MODELS:
        print("BẮT ĐẦU CHUỖI HUẤN LUYỆN TỰ ĐỘNG", len(MODELS_TO_RUN), "MÔ HÌNH...")
        for model_name in MODELS_TO_RUN:
            run_experiment(model_name)
    else:
        print(f"BẮT ĐẦU HUẤN LUYỆN 1 MÔ HÌNH DUY NHẤT: {SINGLE_MODEL_NAME}")
        run_experiment(SINGLE_MODEL_NAME)
        
    print("\n🎉 XONG TOÀN BỘ! BẠN CÓ THỂ KIỂM TRA KẾT QUẢ TRONG THƯ MỤC results/")
