# filepath: /pengu/pengu/models/scEncoderLoss.py
import torch
import torch.nn as nn

class scEncoderLoss(nn.Module):
    def __init__(self, mlm_loss_weight=1.0, contrastive_loss_weight=1.0):
        super(scEncoderLoss, self).__init__()
        self.mlm_loss_weight = mlm_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.criterion_mlm = nn.CrossEntropyLoss(reduction='mean')
        self.criterion_contrastive = nn.CosineEmbeddingLoss(reduction='mean')

    def masked_language_model_loss(self, predictions, targets, attention_mask):
        # Flatten the predictions and targets while considering the attention mask
        masked_predictions = predictions.view(-1, predictions.size(-1))
        masked_targets = targets.view(-1)
        masked_attention_mask = attention_mask.view(-1)

        # Only compute loss for non-masked positions
        loss = self.criterion_mlm(masked_predictions[masked_attention_mask == 1], masked_targets[masked_attention_mask == 1])
        return loss * self.mlm_loss_weight

    def contrastive_loss(self, anchor, positive, negative):
        # Compute contrastive loss using cosine similarity
        labels = torch.ones(anchor.size(0)).to(anchor.device)  # Positive pairs
        loss = self.criterion_contrastive(anchor, positive, labels) + self.criterion_contrastive(anchor, negative, -labels)  # Negative pairs
        return loss * self.contrastive_loss_weight

    def forward(self, mlm_predictions, mlm_targets, attention_mask, anchor, positive, negative):
        mlm_loss = self.masked_language_model_loss(mlm_predictions, mlm_targets, attention_mask)
        contrastive_loss_value = self.contrastive_loss(anchor, positive, negative)
        total_loss = mlm_loss + contrastive_loss_value
        return total_loss, mlm_loss, contrastive_loss_value