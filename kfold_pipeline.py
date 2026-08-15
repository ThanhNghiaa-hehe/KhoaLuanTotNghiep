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
import torch.nn.functional as F
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput

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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=None, output_attentions=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=return_dict, output_attentions=output_attentions)
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
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# --- CONFIG ---
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
K_FOLDS = 5
LABEL_SMOOTHING = 0.1
WARMUP_RATIO = 0.1
OUTPUT_DIR = f"./results/kfold_{MODEL_NAME.replace('/', '_')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(RANDOM_STATE)

print("="*60)
print(f"BẮT ĐẦU {K_FOLDS}-FOLD CROSS VALIDATION")
print("="*60)

# Load data
from datasets import load_dataset
raw = load_dataset("tridm/UIT-VSMEC")
frames = [split_data.to_pandas() for split_data in raw.values()]
full_df = pd.concat(frames, ignore_index=True)
full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
full_df["labels"] = full_df["labels"].map(label2id)
full_df.dropna(subset=["labels"], inplace=True)
full_df["labels"] = full_df["labels"].astype(int)

USE_WORD_SEGMENTATION = "phobert" in MODEL_NAME.lower() or "videberta" in MODEL_NAME.lower()

# Preprocess
def preprocess_text(text: str) -> str:
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")

if USE_WORD_SEGMENTATION:
    print("Tiền xử lý văn bản (Word Segmentation)...")
    full_df["text"] = full_df["text"].apply(preprocess_text)
else:
    print("Bỏ qua Word Segmentation cho model Multilingual...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    # Trả về tên metric gốc (Trainer sẽ tự thêm tiền tố 'eval_' hoặc 'test_')
    return {"accuracy": acc["accuracy"], "macro_f1": macro_f1["f1"]}

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_results = []

X = full_df["text"].values
y = full_df["labels"].values

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- ĐANG CHẠY FOLD {fold + 1}/{K_FOLDS} ---")
    
    train_texts, test_texts = X[train_idx], X[test_idx]
    train_labels, test_labels = y[train_idx], y[test_idx]
    
    # Chia train_texts thành train và val (để dùng early stopping)
    from sklearn.model_selection import train_test_split
    tr_txt, val_txt, tr_lbl, val_lbl = train_test_split(
        train_texts, train_labels, test_size=0.1, stratify=train_labels, random_state=RANDOM_STATE
    )
    
    ds = DatasetDict({
        "train": Dataset.from_dict({"text": tr_txt, "labels": tr_lbl}),
        "validation": Dataset.from_dict({"text": val_txt, "labels": val_lbl}),
        "test": Dataset.from_dict({"text": test_texts, "labels": test_labels})
    })
    
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        
    tokenized_ds = ds.map(tokenize_fn, batched=True, batch_size=256, remove_columns=["text"])
    tokenized_ds.set_format("torch")
    
    # Model
    if "videberta" in MODEL_NAME.lower():
        model = ViDeBERTaWithMeanPooling.from_pretrained(
            MODEL_NAME, num_labels=7, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        ).float()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=7, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        ).float()
    
    fold_out_dir = os.path.join(OUTPUT_DIR, f"fold_{fold+1}")
    
    training_args = TrainingArguments(
        output_dir=fold_out_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", # Trainer tự map thành eval_macro_f1
        label_smoothing_factor=LABEL_SMOOTHING,
        save_total_limit=1,
        report_to="none",
        fp16=False, bf16=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    
    print(f"\n[Fold {fold+1}] Đánh giá trên tập TEST...")
    test_out = trainer.predict(tokenized_ds["test"])
    macro_f1 = test_out.metrics["test_macro_f1"]
    print(f"Fold {fold+1} Macro-F1: {macro_f1:.4f}")
    
    fold_results.append(macro_f1)

print("\n" + "="*60)
print("KẾT QUẢ K-FOLD CROSS VALIDATION")
print("="*60)
for i, score in enumerate(fold_results):
    print(f"Fold {i+1}: {score:.4f}")
    
mean_f1 = np.mean(fold_results)
std_f1 = np.std(fold_results)
print(f"\n✅ TRUNG BÌNH MACRO-F1: {mean_f1:.4f} ± {std_f1:.4f}")

with open(os.path.join(OUTPUT_DIR, "kfold_summary.txt"), "w") as f:
    f.write(f"K-Fold Results for {MODEL_NAME}\n\n")
    for i, score in enumerate(fold_results):
        f.write(f"Fold {i+1}: {score:.4f}\n")
    f.write(f"\nMEAN MACRO-F1: {mean_f1:.4f} ± {std_f1:.4f}\n")
