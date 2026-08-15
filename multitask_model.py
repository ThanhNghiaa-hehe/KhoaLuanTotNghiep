import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.num_emotion_labels = 7
        self.num_sentiment_labels = 3
        
        # SỬA LỖI CHÍNH XÁC 100%: Phải dùng from_pretrained để load trọng số đã học, 
        # nếu dùng from_config thì mô hình sẽ bị "mất trí nhớ" (random weights).
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.transformer.config
        
        # Two classification heads
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.emotion_classifier = nn.Linear(self.config.hidden_size, self.num_emotion_labels)
        self.sentiment_classifier = nn.Linear(self.config.hidden_size, self.num_sentiment_labels)
        
        # Loss functions
        self.emotion_loss_fct = nn.CrossEntropyLoss()
        self.sentiment_loss_fct = nn.CrossEntropyLoss()
        
        # Trọng số cho sentiment task (thường thấp hơn main task)
        self.sentiment_lambda = 0.3
        
        self.init_weights()

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

        # Trainer của HF dựa vào return_dict để tính toán
        if not return_dict:
            output = (emotion_logits, sentiment_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return {
            "loss": total_loss,
            "logits": emotion_logits, # Trả về main logits để Trainer đánh giá metric
            "sentiment_logits": sentiment_logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
        }
