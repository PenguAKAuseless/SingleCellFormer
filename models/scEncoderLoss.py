class scEncoderLoss:
    def __init__(self, mlm_loss_fn, contrastive_loss_fn):
        self.mlm_loss_fn = mlm_loss_fn
        self.contrastive_loss_fn = contrastive_loss_fn

    def calculate_mlm_loss(self, mlm_predictions, mlm_targets):
        """
        Calculate the masked language model (MLM) loss.

        Args:
            mlm_predictions (torch.Tensor): The predictions from the MLM head.
            mlm_targets (torch.Tensor): The ground truth labels for MLM.

        Returns:
            torch.Tensor: The calculated MLM loss.
        """
        return self.mlm_loss_fn(mlm_predictions.view(-1, mlm_predictions.size(-1)), mlm_targets.view(-1))

    def calculate_contrastive_loss(self, contrastive_predictions, contrastive_targets):
        """
        Calculate the contrastive loss.

        Args:
            contrastive_predictions (torch.Tensor): The predictions from the contrastive head.
            contrastive_targets (torch.Tensor): The ground truth labels for contrastive learning.

        Returns:
            torch.Tensor: The calculated contrastive loss.
        """
        return self.contrastive_loss_fn(contrastive_predictions, contrastive_targets)