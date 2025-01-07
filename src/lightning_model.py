import torch
import torch.nn.functional as F
import logging
import numpy as np
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from losses import *
from data_loader import get_dataloader
from predict_async import apply_async_with_callback

class PLModel(pl.LightningModule):
    def __init__(
        self,
        network,
        loss_function,
        data_path,
        val_data_path,
        n_embedding,
        embedding_dim,
        n_conditions,
        n_anchors,
        n_positives,
        n_negatives,
        n_training_samples,
        n_val_samples,
        max_len,
        double_beats,
        hop_length,
        sample_rate,
        learning_rate,
        optimizer,
        num_workers,
        check_val_every_n_epoch,
        random_seed,
    ):
        """
        A PyTorch Lightning model for training and validation.

        Args:
            network (torch.nn.Module): The neural network to train.
            loss_function (Callable): Loss function used for training.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (type): Optimizer class to use.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
            check_val_every_n_epoch (int): Frequency of validation checks.
        """
        super().__init__()
        self.network = network
        self.loss_function = loss_function
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.n_embedding = n_embedding
        self.embedding_dim = embedding_dim
        self.n_conditions = n_conditions
        self.n_anchors = n_anchors
        self.n_positives = n_positives
        self.n_negatives = n_negatives
        self.n_training_samples = n_training_samples
        self.n_val_samples = n_val_samples
        self.max_len=max_len
        self.double_beats = double_beats
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.random_seed = random_seed

        self.save_hyperparameters(ignore=['loss_function', 'network'])
        logging.info('Building PyTorch Lightning model - done')

        self.embeddings_list = []
        self.tracklist = []
        self.val_losses = []
        self.mask = self.build_mask()

    def build_mask(self):
        """
        Constructs a mask for embeddings based on the number of conditions.

        Returns:
            torch.Tensor: Mask tensor of shape (n_conditions, embedding_len).
        """
        mask_len = self.embedding_dim // self.n_conditions
        mask = np.zeros((self.n_conditions, self.embedding_dim))
        
        for i in range(self.n_conditions):
            mask[i, i * mask_len:(i+1) * mask_len] = 1
        return torch.tensor(mask, device='cuda')

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: Training DataLoader.
        """
        return self.get_labeled_dataloader(split='train', 
                                           data_path=self.data_path, 
                                           max_len=self.max_len,
                                           n_embedding=self.n_embedding, 
                                           hop_length=self.hop_length, 
                                           sample_rate=self.sample_rate, 
                                           n_conditions=self.n_conditions, 
                                           n_anchors=self.n_anchors, 
                                           n_positives=self.n_positives, 
                                           n_negatives=self.n_negatives, 
                                           n_samples=self.n_training_samples, 
                                           batch_size=1, 
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return self.get_labeled_dataloader(split='valid', 
                                           data_path=self.val_data_path, 
                                           max_len=self.max_len,
                                           n_embedding=self.n_embedding, 
                                           hop_length=self.hop_length, 
                                           sample_rate=self.sample_rate, 
                                           n_conditions=self.n_conditions, 
                                           n_anchors=self.n_anchors, 
                                           n_positives=self.n_positives, 
                                           n_negatives=self.n_negatives, 
                                           n_samples=self.n_training_samples, 
                                           batch_size=1, 
                                           num_workers=self.num_workers)

    def get_labeled_dataloader(self, split, data_path, max_len, n_embedding, hop_length, sample_rate, n_conditions, n_anchors, n_positives, n_negatives, n_samples, batch_size, num_workers):
        """
        Creates a DataLoader for a specified dataset split.

        Args:
            split (str): Dataset split ('train' or 'valid').
            batch_size (int): Batch size for data loading.

        Returns:
            DataLoader: Labeled DataLoader for the given split.
        """
        return get_dataloader(split, data_path, max_len, n_embedding, hop_length, sample_rate, n_conditions, n_anchors, n_positives, n_negatives, n_samples, batch_size, num_workers)

    def build_ssms(self, embeddings, mask):
        """
        Constructs self-similarity matrices (SSMs) using normalized embeddings and masks.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, embedding_len).
            mask (torch.Tensor): Mask tensor of shape (n_conditions, embedding_len).

        Returns:
            List[torch.Tensor]: List of SSMs for each condition.
        """
        normalized_embeddings = [F.normalize(embeddings * mask[i], p=2, dim=-1) for i in range(self.n_conditions)]
        return [embedding @ embedding.T for embedding in normalized_embeddings]

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (Tuple): A batch containing input features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the training loss.
        """
        track, cs, features, anchors, positives, negatives = batch
        embeddings = self.network(features.squeeze(0)).squeeze()
        ssms_list = self.build_ssms(embeddings, self.mask)

        anchors = anchors.squeeze(0)
        positives = positives.reshape(anchors.size(0), -1)
        negatives = negatives.reshape(anchors.size(0), -1)

        loss = self.loss_function(ssms_list, cs.squeeze(), anchors, positives, negatives, embeddings)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (Tuple): A batch containing input features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the validation loss.
        """
        track, cs, features, anchors, positives, negatives = batch
        embeddings = self.network(features.squeeze(0)).squeeze()
        ssms_list = self.build_ssms(embeddings, self.mask)

        anchors = anchors.squeeze(0)
        positives = positives.reshape(anchors.size(0), -1)
        negatives = negatives.reshape(anchors.size(0), -1)
        self.tracklist.append(track)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        self.embeddings_list.append([track[0], embeddings.cpu().detach().numpy()])
        loss = self.loss_function(ssms_list, cs.squeeze(), anchors, positives, negatives, embeddings)
        self.val_losses.append(loss.item())
        return {'valid_loss': loss.item()}

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        Logs the mean validation loss.
        """
        out = apply_async_with_callback(self.embeddings_list, self.tracklist, True, False, 0, None)
        #print('out shape =', out.shape)
        P1, R1, F1, P3, R3, F3 = np.mean(out[:,0]), np.mean(out[:,1]), np.mean(out[:,2]), np.mean(out[:,3]), np.mean(out[:,4]), np.mean(out[:,5])
        self.log('valid_F3', F3, sync_dist=True)
        self.log('valid_F', (F3+F1)/2, sync_dist=True)
        print('F1 =', F1, 'P3 =', P3, 'R3 =', R3, 'F3 =', F3)
        self.embeddings_list = []
        self.tracklist = []
        val_loss = np.mean(self.val_losses)
        self.log('val_loss', val_loss, sync_dist=True)
        logging.info(f'Validation loss: {val_loss}')
        self.val_losses = []

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            tuple: A tuple containing the optimizer and the scheduler.
        """
        optimizer = self.optimizer(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': self.check_val_every_n_epoch
        }
        return [optimizer], [scheduler]
