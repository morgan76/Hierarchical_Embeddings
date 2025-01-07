import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, n_conditions, symmetrical=False):
        """
        Contrastive Loss with temperature scaling and optional symmetry, supporting multiple self-similarity matrices.

        Args:
            temperature (float): Scaling factor for the similarity scores.
            symmetrical (bool): Whether to compute the symmetrical version of the loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.n_conditions = n_conditions
        self.symmetrical = symmetrical


    def decorrelation_loss(self, embedding):
        """
        Computes the decorrelation loss to ensure the n_conditions subspaces of the embeddings are decorrelated.

        Args:
            embedding (torch.Tensor): The embeddings of shape (batch_size, embedding_dim).
            n_conditions (int): Number of subspaces to divide the embedding into.

        Returns:
            torch.Tensor: The decorrelation loss.
        """
        batch_size, embedding_dim = embedding.shape

        # Calculate the subspace size
        subspace_size = embedding_dim // self.n_conditions
        assert embedding_dim % self.n_conditions == 0, "Embedding dimension must be divisible by n_conditions."

        # Split the embedding into n_conditions subspaces
        subspaces = torch.split(embedding, subspace_size, dim=1)  # List of n_conditions tensors

        # Center each subspace
        centered_subspaces = [subspace - subspace.mean(dim=0, keepdim=True) for subspace in subspaces]

        # Compute the decorrelation loss (sum of off-diagonal covariances between all pairs of subspaces)
        covar_loss = 0.0
        for i in range(self.n_conditions):
            for j in range(i + 1, self.n_conditions):  # Only unique pairs
                cov_matrix = (centered_subspaces[i].T @ centered_subspaces[j]) / (batch_size - 1)
                covar_loss += torch.norm(cov_matrix, p='fro')

        return covar_loss

    

    def forward(self, ssms_list, cs, anchors, positives, negatives, embeddings):
        """
        Compute the (optionally symmetrical) contrastive loss for a batch using a list of self-similarity matrices.

        Args:
            ssms_list (list of torch.Tensor): List of self-similarity matrices.
            cs (torch.Tensor): Tensor indicating which ssm in `ssms_list` to use for each anchor.
            anchors (torch.Tensor): Indices of anchor samples, shape (n_anchors,).
            positives (torch.Tensor): Indices of positive samples, shape (n_anchors, n_positives).
            negatives (torch.Tensor): Indices of negative samples, shape (n_anchors, n_negatives).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        total_loss = 0.0
        batch_size = anchors.size(0)

        for i, ssm in enumerate(ssms_list):
            # Mask for the current ssm
            mask = (cs == i)
            if not mask.any():
                continue

            # Extract indices for the current ssm
            anchors_i = anchors[mask]
            positives_i = positives[mask]
            negatives_i = negatives[mask]

            # Scale the similarity matrix with temperature
            scaled_ssm = ssm / self.temperature

            # ** Forward Direction: Anchors -> Positives **
            # Extract positive and negative scores
            pos_scores = scaled_ssm[anchors_i.unsqueeze(1), positives_i]  # Shape: (n_anchors_i, n_positives)
            pos_scores_exp = torch.exp(pos_scores)

            neg_scores = scaled_ssm[anchors_i.unsqueeze(1), negatives_i]  # Shape: (n_anchors_i, n_negatives)
            neg_scores_exp = torch.exp(neg_scores).sum(dim=1, keepdim=True)  # Sum over negatives (n_anchors_i, 1)

            # Normalize positive scores and compute forward loss
            normalized_pos_scores = pos_scores_exp / (pos_scores_exp + neg_scores_exp + 1e-10)
            loss_forward = -torch.log(normalized_pos_scores + 1e-10).mean()

            if not self.symmetrical:
                total_loss += loss_forward
                continue

            # ** Reverse Direction: Positives -> Anchors **
            # Reshape positives and anchors for reverse calculation
            n_anchors_i, n_positives = positives_i.size()
            expanded_positives = positives_i.view(-1)  # Flatten positives
            expanded_negatives = negatives_i.unsqueeze(1).expand(n_anchors_i, n_positives, negatives_i.size(1)).contiguous()
            expanded_negatives = expanded_negatives.view(-1, negatives_i.size(1))

            pos_scores_reverse = scaled_ssm[expanded_positives, anchors_i.repeat_interleave(n_positives)]
            pos_scores_reverse_exp = torch.exp(pos_scores_reverse)

            neg_scores_reverse = scaled_ssm[expanded_positives.unsqueeze(1).expand(-1, negatives_i.size(1)), expanded_negatives]
            neg_scores_reverse_exp = torch.exp(neg_scores_reverse).sum(dim=1)

            # Reshape reverse scores and compute reverse loss
            pos_scores_reverse_exp = pos_scores_reverse_exp.view(n_anchors_i, n_positives)
            neg_scores_reverse_exp = neg_scores_reverse_exp.view(n_anchors_i, n_positives)

            normalized_pos_scores_reverse = pos_scores_reverse_exp / (pos_scores_reverse_exp + neg_scores_reverse_exp + 1e-10)
            loss_reverse = -torch.log(normalized_pos_scores_reverse + 1e-10).mean()

            # Add combined loss for this ssm
            total_loss += (loss_forward + loss_reverse) / 2

        # Normalize the total loss by the number of ssms used
        var_covar = self.decorrelation_loss(embeddings)
        #print(var_covar)
        return total_loss / len(ssms_list) + var_covar
