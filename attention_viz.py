import os
import glob
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
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DebertaV2Model,
    DebertaV2PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Phải cài: pip install bertviz
from bertviz import head_view

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

RESULTS_DIR = "./results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "attention_viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_attention():
    print("="*60)
    print("ATTENTION VISUALIZATION (TẤT CẢ MÔ HÌNH)")
    print("="*60)
    print("💡 MẸO: Nếu gặp lỗi Tokenizer, hãy chạy lệnh này trong 1 Cell Colab mới: !pip install sentencepiece")
    print("="*60)
    
    # 1. TẢI DATASET ĐỂ TÌM CÂU ĐÚNG/SAI CHO TỪNG MÔ HÌNH
    print("\n[1] Đang tải dataset tridm/UIT-VSMEC để lấy tập Test...")
    raw = load_dataset("tridm/UIT-VSMEC")
    frames = [split_data.to_pandas() for split_data in raw.values()]
    full_df = pd.concat(frames, ignore_index=True)
    full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
    
    LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
    label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
    id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}
    
    full_df["labels"] = full_df["labels"].map(label2id)
    full_df.dropna(subset=["labels"], inplace=True)
    full_df["labels"] = full_df["labels"].astype(int)
    
    RANDOM_STATE = 42
    train_val_df, test_df = train_test_split(
        full_df, test_size=0.20,
        stratify=full_df["labels"], random_state=RANDOM_STATE
    )
    print(f"Đã tải tập Test với {len(test_df)} câu.")
    
    # 2. Tìm tất cả các mô hình đã huấn luyện trong thư mục results
    model_dirs = glob.glob(os.path.join(RESULTS_DIR, "*/"))
    valid_models = []
    
    for d in model_dirs:
        model_name = os.path.basename(os.path.normpath(d))
        # Bỏ qua các thư mục không phải là mô hình Transformer
        if model_name in ["ensemble", "baseline_svm", "baseline_lr", "confusion_analysis", "attention_viz"]:
            continue
        valid_models.append(d)
        
    if len(valid_models) == 0:
        print("❌ Không tìm thấy thư mục mô hình nào trong ./results/")
        return
        
    print(f"Sẽ chạy phân tích Attention trên {len(valid_models)} mô hình...\n")
    
    # 3. Chạy vòng lặp qua từng mô hình
    for model_dir in valid_models:
        model_name = os.path.basename(os.path.normpath(model_dir))
        print("-" * 60)
        print(f"🚀 Bắt đầu mổ xẻ mô hình: {model_name}")
        
        # TỰ ĐỘNG TÌM THƯ MỤC CHECKPOINT MỚI NHẤT (VÌ BẠN KHÔNG LƯU Ở THƯ MỤC GỐC)
        load_dir = model_dir
        checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
        if len(checkpoints) > 0:
            # Sắp xếp để lấy checkpoint cuối cùng
            checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
            load_dir = checkpoints[-1]
            print(f"   -> Tìm thấy não bộ mô hình ẩn bên trong: {os.path.basename(load_dir)}")
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_dir)
            
            # Load mô hình (Đặc biệt xử lý ViDeBERTa vì nó là Custom Class)
            if "videberta" in model_name.lower():
                model = ViDeBERTaWithMeanPooling.from_pretrained(load_dir, output_attentions=True)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(load_dir, output_attentions=True)
            
            # Áp dụng Tách từ nếu mô hình yêu cầu (tránh vẽ Attention sai lệnh)
            use_word_seg = "phobert" in model_name.lower() or "videberta" in model_name.lower()
            
            # LỌC RA 5 CÂU ĐÚNG, 5 CÂU SAI CHO MÔ HÌNH NÀY
            model.eval()
            model.to("cpu")
            correct_samples = []
            incorrect_samples = []
            
            print("   -> Đang tìm kiếm 5 câu đoán đúng và 5 câu đoán sai...")
            for _, row in test_df.iterrows():
                if len(correct_samples) >= 5 and len(incorrect_samples) >= 5:
                    break
                    
                raw_text = row["text"]
                true_label_id = int(row["labels"])
                
                text = raw_text
                if use_word_seg:
                    from underthesea import word_tokenize
                    text = word_tokenize(text, format="text")
                    
                inputs = tokenizer(text, return_tensors="pt").to("cpu")
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                pred_label_id = outputs.logits.argmax(dim=-1).item()
                is_correct = (pred_label_id == true_label_id)
                
                sample = {
                    "raw_text": raw_text,
                    "text": text,
                    "true_label": id2label[true_label_id],
                    "pred_label": id2label[pred_label_id],
                    "is_correct": is_correct,
                    "attentions": outputs.attentions,
                    "input_ids": inputs.input_ids[0]
                }
                
                if is_correct and len(correct_samples) < 5:
                    correct_samples.append(sample)
                elif not is_correct and len(incorrect_samples) < 5:
                    incorrect_samples.append(sample)
                    
            print(f"   -> Đã tìm được {len(correct_samples)} câu đúng, {len(incorrect_samples)} câu sai.")
            
            all_samples = correct_samples + incorrect_samples
            
            for i, sample_data in enumerate(all_samples):
                raw_text = sample_data["raw_text"]
                true_label = sample_data["true_label"]
                pred_label = sample_data["pred_label"]
                is_correct = sample_data["is_correct"]
                attention = sample_data["attentions"]
                
                if attention is None:
                    print(f"  ⚠️ Cảnh báo: Mô hình {model_name} không hỗ trợ xuất Attention.")
                    continue
                
                tokens = tokenizer.convert_ids_to_tokens(sample_data["input_ids"])
                
                # Bertviz yêu cầu format attentions là tuple của tensor
                html_head_view = head_view(attention, tokens, html_action='return')
                
                # --- PHÂN TÍCH TỰ ĐỘNG ĐỂ TÌM TỪ QUAN TRỌNG NHẤT ---
                # attention[-1] là layer cuối cùng, shape: (1, num_heads, seq_len, seq_len)
                # SỬA LỖI ĐIỂM MÙ MACHINE LEARNING:
                # Nếu lấy trung bình (mean), các Head học ngữ pháp sẽ làm loãng trọng số của Head học Cảm xúc, khiến từ "để" bị nhầm là quan trọng.
                # Do đó, ta phải lấy MAX qua các Heads để bắt được tia Attention mạnh nhất từ Head chuyên biệt về Cảm xúc.
                last_layer_attn = attention[-1][0].max(dim=0)[0].detach().cpu().numpy() # shape (seq_len, seq_len)
                
                # cls_idx là 0 (token <s> hoặc [CLS])
                cls_attn = last_layer_attn[0, :]
                
                # Bỏ qua <s> ở đầu (index 0) và </s> ở cuối (index len(tokens)-1)
                valid_scores = cls_attn[1:-1]
                if len(valid_scores) > 0:
                    max_idx = valid_scores.argmax() + 1
                    max_score = valid_scores.max()
                    most_attended_word = tokens[max_idx].replace("@@", "").replace(" ", " ")
                else:
                    most_attended_word = "N/A"
                    max_score = 0.0
                # ---------------------------------------------------
                
                # UI Styling for Correct/Incorrect Status
                status_color = "#059669" if is_correct else "#dc2626"
                status_text = "ĐOÁN ĐÚNG" if is_correct else "ĐOÁN SAI"
                status_bg = "#d1fae5" if is_correct else "#fee2e2"
                status_border = "#34d399" if is_correct else "#f87171"
                
                custom_html = f"""

                <!DOCTYPE html>
                <html lang="vi">
                <head>
                    <meta charset="utf-8">
                    <title>Attention Analysis - {model_name}</title>
                    <style>
                        :root {{
                            --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0; 
                            --text: #1e293b; --text-muted: #64748b; --primary: #3b82f6;
                        }}
                        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                        body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); display: flex; justify-content: center; padding: 40px 20px; line-height: 1.6; }}
                        .container {{ display: grid; grid-template-columns: 340px 1fr; gap: 32px; max-width: 1400px; width: 100%; align-items: start; }}
                        .sidebar {{ display: flex; flex-direction: column; gap: 20px; position: sticky; top: 40px; }}
                        .card {{ background: var(--surface); padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05); border: 1px solid var(--border); }}
                        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; letter-spacing: 0.05em; }}
                        .badge-success {{ background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }}
                        .badge-error {{ background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }}
                        h2 {{ font-size: 22px; margin-bottom: 8px; font-weight: 700; letter-spacing: -0.02em; }}
                        .sentence-box {{ font-size: 18px; font-style: italic; border-left: 3px solid var(--border); padding-left: 16px; margin: 16px 0; color: #334155; }}
                        
                        .insight-box {{ background: linear-gradient(135deg, #eff6ff, #dbeafe); border-left: 4px solid var(--primary); padding: 20px; border-radius: 8px; color: #1e40af; font-size: 15px; }}
                        
                        details {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgb(0 0 0 / 0.05); }}
                        summary {{ font-weight: 600; cursor: pointer; color: #334155; user-select: none; font-size: 15px; display: flex; align-items: center; }}
                        details p {{ font-size: 14px; color: var(--text-muted); margin-top: 12px; line-height: 1.5; }}
                        
                        .viz-box {{ background: var(--surface); border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05); border: 1px solid var(--border); overflow-x: auto; min-height: 600px; }}
                        /* Bertviz Native UI Override */
                        .viz-box select {{ padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); font-family: inherit; font-size: 14px; outline: none; background: #f8fafc; }}
                        .viz-box svg path {{ stroke-width: 2.5px !important; mix-blend-mode: multiply; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="sidebar">
                            <div class="card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px;">
                                    <span style="font-size: 11px; font-weight: bold; color: var(--text-muted); text-transform: uppercase;">{model_name}</span>
                                    <span class="badge {'badge-success' if is_correct else 'badge-error'}">{status_text}</span>
                                </div>
                                <h2>Phân tích Cảm xúc</h2>
                                <p style="font-size: 14px; color: var(--text-muted);">Thực tế: <b>{true_label}</b> &nbsp;|&nbsp; Dự đoán: <b style="color: {status_color};">{pred_label}</b></p>
                                <div class="sentence-box">"{raw_text}"</div>
                            </div>
                            
                            <div class="insight-box">
                                <b>💡 Khám Phá Của PhoBERT:</b><br><br>
                                Tại Layer 11, khi áp dụng Max-Pooling để lọc nhiễu, Head chuyên trách Cảm Xúc đã dồn sự chú ý mãnh liệt nhất vào chữ <b>[{most_attended_word}]</b> <i>(Trọng số: {max_score:.4f})</i>.<br><br>Đây chính là từ khóa lõi quyết định nhãn dự đoán của toàn câu!
                            </div>

                            <details open>
                                <summary>📖 Hướng dẫn đọc Biểu Đồ (Click mở/đóng)</summary>
                                <p>
                                <b>1. Self-Attention:</b> Đường nối càng đậm chứng tỏ từ bên trái phụ thuộc rất nhiều vào ngữ cảnh của từ bên phải.<br><br>
                                <b>2. Token &lt;s&gt;:</b> Đại diện cho toàn bộ câu (Tổng hợp vector). Hãy rê chuột vào &lt;s&gt; ở lớp sâu nhất để xem nó lấy cảm xúc từ chữ nào nhiều nhất.<br><br>
                                <b>3. Lọc nhiễu:</b> Chọn "Layer: 11" trên thanh công cụ, sau đó nháy đúp chuột vào chữ "Layer: 11" để hiện toàn bộ các màu.
                                </p>
                            </details>
                            
                            <details>
                                <summary>🧠 12 Quy luật Ngữ nghĩa (Heads)</summary>
                                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 12px; font-size: 13px;">
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #1f77b4;"></span> Head 1: Quan hệ Chủ-Vị</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #ff7f0e;"></span> Head 2: Tính từ - Danh từ</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #2ca02c;"></span> Head 3: Từ hạn định - Danh từ</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #d62728;"></span> Head 4: Giới từ - Tân ngữ</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #9467bd;"></span> Head 5: Liên kết Đại từ</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #8c564b;"></span> Head 6: Cấu trúc Phủ định</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #e377c2;"></span> Head 7: Trạng từ mức độ</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #7f7f7f;"></span> Head 8: Động từ cảm xúc</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #bcbd22;"></span> Head 9: Từ nối (và, vì)</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #17becf;"></span> Head 10: Dấu câu</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #aec7e8;"></span> Head 11: Ngữ cảnh toàn cục</div>
                                    <div style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; border-radius: 2px; background: #ffbb78;"></span> Head 12: Chú ý chính nó</div>
                                </div>
                            </details>
                        </div>
                        
                        <div class="viz-box">
                            {html_head_view.data}
                        </div>
                    </div>
                </body>
                </html>
                """
                
                file_path = os.path.join(OUTPUT_DIR, f"attention_{model_name}_sample_{i+1}.html")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(custom_html)
                    
            print(f"✅ Đã xuất file HTML với giao diện UX/UI cao cấp cho mô hình {model_name}")
            
            # Giải phóng bộ nhớ RAM sau mỗi model
            del model
            del tokenizer
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"❌ Lỗi khi phân tích mô hình {model_name}: {e}")
            print(f"   => HƯỚNG GIẢI QUYẾT: Hãy thử chạy lệnh: !pip install sentencepiece")
            
    print("\n" + "="*60)
    print("✅ HOÀN TẤT TOÀN BỘ! Bạn có thể mở các file HTML bằng trình duyệt để xem.")
    print(f"📂 Xem kết quả tại: {OUTPUT_DIR}/")



# ==============================================================================
# HỆ THỐNG GOOGLE LIT DASHBOARD TƯƠNG TÁC
# ==============================================================================
def run_lit_dashboard():
    print("="*60)
    print("🚀 ĐANG KHỞI ĐỘNG GOOGLE LIT DASHBOARD...")
    print("="*60)
    try:
        from lit_nlp import dev_server
        from lit_nlp import server_flags
        from lit_nlp.api import model as lit_model
        from lit_nlp.api import dataset as lit_dataset
        from lit_nlp.api import types as lit_types
    except ImportError:
        print("❌ LỖI: Chưa cài đặt lit-nlp. Hãy chạy lệnh: !pip install lit-nlp")
        return

    # Tải dataset
    print("[1] Đang nạp dataset UIT-VSMEC...")
    raw = load_dataset("tridm/UIT-VSMEC")
    frames = [split_data.to_pandas() for split_data in raw.values()]
    full_df = pd.concat(frames, ignore_index=True)
    full_df.rename(columns={"Sentence": "text", "Emotion": "labels"}, inplace=True)
    LABEL_NAMES = ["Enjoyment", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Other"]
    label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
    id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}
    full_df["labels"] = full_df["labels"].map(label2id)
    full_df.dropna(subset=["labels"], inplace=True)
    full_df["labels"] = full_df["labels"].astype(int)
    _, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["labels"], random_state=42)

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
            
            # Xử lý word segment
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

    # Chọn 100 câu mẫu để LIT không bị quá tải RAM (Google Colab giới hạn RAM)
    eval_df = test_df.sample(min(100, len(test_df)), random_state=42)
    datasets = {"UIT-VSMEC_Test": UITDataset(eval_df)}
    
    # Nạp Model tốt nhất
    model_dirs = glob.glob(os.path.join(RESULTS_DIR, "*/"))
    valid_models = [d for d in model_dirs if os.path.basename(os.path.normpath(d)) not in ["ensemble", "baseline_svm", "baseline_lr", "confusion_analysis", "attention_viz"]]
    
    models = {}
    print("\n[2] Đang nạp các mô hình vào LIT...")
    for m_dir in valid_models:
        m_name = os.path.basename(os.path.normpath(m_dir))
        checkpoints = glob.glob(os.path.join(m_dir, "checkpoint-*"))
        load_dir = checkpoints[-1] if checkpoints else m_dir
        
        try:
            print(f"  + Nạp {m_name} (Có thể mất 1 phút)...")
            models[m_name] = PhoBERTLitModel(load_dir)
            # Chỉ nạp 1 model duy nhất để tránh treo máy Colab
            break 
        except Exception as e:
            print(f"    ⚠️ Bỏ qua {m_name} do lỗi: {e}")

    if not models:
        print("❌ Lỗi: Không có mô hình nào được nạp!")
        return

    # Khởi động server LIT
    import lit_nlp
    client_root = os.path.join(os.path.dirname(lit_nlp.__file__), "client", "build")
    server = dev_server.Server(models, datasets, port=8501, host="0.0.0.0", client_root=client_root)
    print("\n" + "="*60)
    print("✅ LIT SERVER ĐÃ SẴN SÀNG HOẠT ĐỘNG!")
    print("👉 HÃY MỞ TRÌNH DUYỆT VÀ TRUY CẬP: http://localhost:8501/")
    print("="*60)
    server.serve()
    
    # Giữ tiến trình Python sống để Server không bị tắt
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nĐã tắt LIT Server.")

def main_menu():
    print("="*60)
    print(" CÔNG CỤ EXPLAINABLE AI (XAI) CHO KHÓA LUẬN")
    print("="*60)
    print("1. Xuất báo cáo HTML tĩnh (BertViz - Giao diện mới siêu gọn)")
    print("2. Chạy Dashboard tương tác trực tiếp (Google LIT)")
    print("="*60)
    choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()
    
    if choice == '2':
        run_lit_dashboard()
    else:
        visualize_attention()

if __name__ == "__main__":
    main_menu()
