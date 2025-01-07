import numpy as np
import torch
from torch.utils import data
import librosa
from utils import *
from tqdm import tqdm
import random


class DatasetSimCLR(data.Dataset):
    """
    Dataset class for SimCLR-style contrastive learning.
    
    Attributes:
        split (str): Data split ('train', 'val', or 'test').
        tracklist (list): List of track identifiers.
        n_embedding (int): Number of embeddings for each frame.
        hop_length (int): Hop length for audio processing.
        sample_rate (int): Sampling rate of audio.
        n_conditions (int): Number of sampling conditions.
        n_anchors (int): Number of anchor samples.
        n_positives (int): Number of positive samples per anchor.
        n_negatives (int): Number of negative samples per anchor.
        n_samples (int): Number of samples to include.
        sampled_frames (list): List of sampled frames with their anchors, positives, and negatives.
    """
    def __init__(
        self, split, data_path, max_len, n_embedding, hop_length, sample_rate, 
        n_conditions, n_anchors, n_positives, n_negatives, n_samples
    ):
        self.split = split
        self.data_path = data_path
        self.tracklist = clean_tracklist_audio(data_path=self.data_path, annotations=False, n_samples=5000)
        print('### len tracklist', len(self.tracklist))
        self.n_embedding = n_embedding
        self.max_len = max_len
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_conditions = n_conditions
        self.n_anchors = n_anchors
        self.n_positives = n_positives
        self.n_negatives = n_negatives
        self.n_samples = n_samples
        self.deltas = self.get_deltas()
        self.sampled_frames = self._sample_frames()
        

    def _sample_frames(self):
        """
        Pre-samples anchors, positives, and negatives for each track.
        
        Returns:
            list: Precomputed frame triplets for each track.
        """
        sampled_frames = []

        for track in tqdm(self.tracklist, desc="Sampling frames"):
            file_struct = FileStruct(track)
            beat_frames, _ = read_beats(file_struct.beat_file)
            beat_frames = librosa.util.fix_frames(beat_frames)[:self.max_len]
            nb_embeddings = len(beat_frames) - 1

            if nb_embeddings > 200:
                # Sample anchor indexes
                anchor_indexes = np.random.choice(
                    np.arange(0, nb_embeddings),
                    size=min(self.n_anchors, nb_embeddings),
                    replace=False
                )
                conditions = list(range(self.n_conditions))
                anchors, positives, negatives, cs = [], [], [], []

                for anchor_index in anchor_indexes:
                    for condition in conditions:
                        pos_idx, neg_idx = self._sampler(
                            anchor_index, nb_embeddings, condition
                        )
                        anchors.append(anchor_index)
                        positives.append(pos_idx)
                        negatives.append(neg_idx)
                        cs.append(condition)
                        #print(track, condition, anchor_index, pos_idx, neg_idx)
                
                sampled_frames.append((track, cs, anchors, positives, negatives))

        return sampled_frames

    def get_deltas(self, delta_min=1, delta_max=96, step=16):
        """
        Generate a dictionary of deltas for each condition index, ensuring continuity between delta_p and delta_n.

        Args:
            delta_min (int): Minimum value of the range.
            delta_max (int): Maximum value of the range.
            n_conditions (int): Number of conditions.

        Returns:
            dict: A dictionary containing deltas for each condition index.
        """

        deltas = {}
        for i in range(self.n_conditions):
            if i == 0:
                # The first element follows the initial fixed pattern
                deltas[i] = {
                    'delta_p_min': delta_min,
                    'delta_p_max': step,
                    'delta_n_min': delta_min + step,
                    'delta_n_max': 2 * step,
                }
            elif i == self.n_conditions - 1:
                # The last element ensures delta_n_max = delta_max
                deltas[i] = {
                    'delta_p_min': deltas[i - 1]['delta_n_min'],
                    'delta_p_max': deltas[i - 1]['delta_n_max'],
                    'delta_n_min': deltas[i - 1]['delta_n_max'],
                    'delta_n_max': delta_max,
                }
            else:
                # Intermediate elements maintain continuity
                deltas[i] = {
                    'delta_p_min': deltas[i - 1]['delta_n_min'],
                    'delta_p_max': deltas[i - 1]['delta_n_max'],
                    'delta_n_min': deltas[i - 1]['delta_n_max'],
                    'delta_n_max': deltas[i - 1]['delta_n_max'] + step,
                }

        return deltas


    def _sampler(self, anchor_index, nb_embeddings, condition):
        """
        Samples positive and negative indexes for a given anchor.
        
        Args:
            anchor_index (int): Index of the anchor frame.
            nb_embeddings (int): Total number of embeddings.
            condition (int): Sampling condition.
            delta_p (int, optional): Range for positive samples. Defaults to 16.
            delta_n_min (int, optional): Min range for negative samples. Defaults to 1.
            delta_n_max (int, optional): Max range for negative samples. Defaults to 96.
        
        Returns:
            tuple: Positive and negative sample indexes.
        """
        random.seed(None)
        L = nb_embeddings - 1

        deltas = self.deltas[condition]
        # Sample positive indexes
        total_positive = (
            list(np.arange(max(anchor_index - deltas['delta_p_max'], 0), max(anchor_index - deltas['delta_p_min'], 0))) +
            list(np.arange(min(anchor_index + deltas['delta_p_min'], L), min(anchor_index + deltas['delta_p_max'], L)))
        )
        positive_indexes = np.random.choice(total_positive, size=self.n_positives, replace=True)

        # Sample negative indexes
        total_negative = (
            list(np.arange(max(anchor_index - deltas['delta_n_max'], 0), max(anchor_index - deltas['delta_n_min'], 0))) +
            list(np.arange(min(anchor_index + deltas['delta_n_min'], L), min(anchor_index + deltas['delta_n_max'], L)))
        )
        negative_indexes = np.random.choice(total_negative, size=self.n_negatives, replace=True)

        return positive_indexes, negative_indexes
    


    def __getitem__(self, index):
        """
        Retrieves a data sample.
        
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: Condition, features, anchors, positives, negatives.
        """
        track, cs, anchors, positives, negatives = self.sampled_frames[index]
        file_struct = FileStruct(track)

        # Read and pad waveform
        waveform = np.load(file_struct.audio_npy_file, mmap_mode='r', allow_pickle=True)
        pad_width = (self.hop_length * self.n_embedding - 2) // 2
        features_padded = np.pad(waveform, pad_width=(pad_width, pad_width), mode='edge')

        # Extract features at beat frames
        beat_frames, _ = read_beats(file_struct.beat_file)
        beat_frames = librosa.util.fix_frames(beat_frames)
        beat_frames = [i*256 for i in beat_frames]
        if self.split == 'train':
            beat_frames = beat_frames[:self.max_len]
        features = np.stack([
            features_padded[i:i + pad_width * 2] for i in beat_frames
        ], axis=0)

        return (
            track, 
            torch.tensor(np.array(cs), dtype=torch.long),
            torch.tensor(np.array(features), dtype=torch.float32),  # Convert list of arrays to a single ndarray first
            torch.tensor(np.array(anchors), dtype=torch.long),
            torch.tensor(np.array(positives).flatten(), dtype=torch.long),  # Flatten before converting
            torch.tensor(np.array(negatives).flatten(), dtype=torch.long)   # Flatten before converting
        )

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.sampled_frames)


def get_dataloader(
    split, data_path, max_len, n_embedding, hop_length, sample_rate,
    n_conditions, n_anchors, n_positives, n_negatives,
    n_samples, batch_size, num_workers
):
    """
    Creates a DataLoader for the SimCLR dataset.

    Args:
        split (str): Data split ('train', 'val', or 'test').
        tracklist (list): List of tracks to process.
        n_embedding (int): Number of embeddings.
        hop_length (int): Hop length for audio processing.
        sample_rate (int): Sampling rate of audio.
        n_conditions (int): Number of sampling conditions.
        n_anchors (int): Number of anchor samples.
        n_positives (int): Number of positive samples per anchor.
        n_negatives (int): Number of negative samples per anchor.
        n_samples (int): Number of samples to process.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of parallel workers.
    
    Returns:
        DataLoader: Configured DataLoader instance.
    """
    dataset = DatasetSimCLR(
        split, data_path, max_len, n_embedding, hop_length, sample_rate, 
        n_conditions, n_anchors, n_positives, n_negatives, n_samples
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
