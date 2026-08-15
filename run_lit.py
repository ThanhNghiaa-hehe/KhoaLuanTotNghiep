import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DebertaV2Model,
    DebertaV2PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
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

# ==============================================================================
# CONFIG
# ==============================================================================
RESULTS_DIR = "/content/drive/MyDrive/KhoaLuan_Results"

LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}

# ==============================================================================
# LIT WRAPPERS
# ==============================================================================
from lit_nlp.api import model as lit_model
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

class UITDataset(lit_dataset.Dataset):
    def __init__(self, df: pd.DataFrame):
        self._examples = []
        for _, row in df.iterrows():
            self._examples.append({
                "sentence": str(row["text"]),
                "label": id2label[int(row["labels"])]
            })
    def spec(self):
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=LABEL_NAMES)
        }

class PhoBERTLitModel(lit_model.Model):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_name = os.path.basename(os.path.normpath(model_path))
        if "videberta" in model_name.lower():
            self.model = ViDeBERTaWithMeanPooling.from_pretrained(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, inputs):
        texts = [i["sentence"] for i in inputs]
        from underthesea import word_tokenize
        texts = [word_tokenize(text, format="text") for text in texts]
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        return [{"probas": p} for p in probs]

    def input_spec(self):
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=LABEL_NAMES, required=False)
        }

    def output_spec(self):
        return {
            "probas": lit_types.MulticlassPreds(vocab=LABEL_NAMES, parent="label")
        }

# ==============================================================================
# MAIN - Dùng LitWidget cho Colab
# ==============================================================================
def run_lit_colab():
    """Khởi động LIT Dashboard nhúng trực tiếp vào Google Colab."""
    from lit_nlp.notebook import LitWidget

    print("="*60)
    print("🚀 ĐANG KHỞI ĐỘNG GOOGLE LIT (Colab Widget Mode)...")
    print("="*60)

    # 1. Load Data
    print("[1/3] Đang nạp dataset UIT-VSMEC...")
    raw = load_dataset("tridm/UIT-VSMEC")
    frames = [split_data.to_pandas() for split_data in raw.values()]
    full_df = pd.concat(frames, ignore_index=True)
    full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
    full_df["labels"] = full_df["labels"].map(label2id)
    full_df.dropna(subset=["labels"], inplace=True)
    full_df["labels"] = full_df["labels"].astype(int)
    _, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["labels"], random_state=42)

    eval_df = test_df # Chạy toàn bộ dữ liệu
    datasets = {"UIT-VSMEC_Test": UITDataset(eval_df)}

    # 2. Load best model
    print("[2/3] Đang nạp mô hình PhoBERT...")
    model_dirs = glob.glob(os.path.join(RESULTS_DIR, "*/"))
    valid_models = [d for d in model_dirs if os.path.basename(os.path.normpath(d)) not in ["ensemble", "baseline_svm", "baseline_lr", "confusion_analysis", "attention_viz"]]

    models = {}
    for m_dir in valid_models:
        m_name = os.path.basename(os.path.normpath(m_dir))
        checkpoints = glob.glob(os.path.join(m_dir, "checkpoint-*"))
        load_dir = checkpoints[-1] if checkpoints else m_dir
        try:
            print(f"  + Nạp {m_name}...")
            models[m_name] = PhoBERTLitModel(load_dir)
            break  # Chỉ nạp 1 model cho nhẹ RAM Colab
        except Exception as e:
            print(f"  ⚠️ Bỏ qua {m_name}: {e}")

    if not models:
        print("❌ Không có mô hình nào được nạp!")
        return None

    # 3. Tạo LIT Widget (nhúng trực tiếp vào Notebook)
    print("[3/3] Đang tạo LIT Widget...")
    widget = LitWidget(models, datasets, height=800)

    print("\n" + "="*60)
    print("✅ ĐÃ SẴN SÀNG! Gọi lệnh bên dưới để hiển thị:")
    print("widget.render()")
    print("="*60)
    return widget

# Cách dùng trong Colab:
# Copy toàn bộ đoạn code này thả vào 1 ô, sau đó chạy:
# widget = run_lit_colab()
# widget.render()
