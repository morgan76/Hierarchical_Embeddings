from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from training_utils import DirManager


def get_callbacks(patience: int, dir_manager: DirManager, monitor: str = 'valid_loss', mode: str = 'min'):
    """return a list of callbacks including early stopping and model checkpoint(s)

    Args:
        patience (int): patience [epoch] for early-stopping
        dir_manager: an instance of DirManager
        monitor (str): value to monitor for early-stopping and checkpoint savers

    Return: A list of callback instances

    Note:
        best_saver: This callback keeps `save_top_k` best checkpoints with epoch number
        simple_name_best_saver: This callback behaves the same as best_saver with simpler filename
        all_saver: This callback saves every checkpoint after each epoch

    """
    callbacks = []

    best_saver = pl_callbacks.model_checkpoint.ModelCheckpoint(
        filename='best_model',  
        dirpath=dir_manager.checkpoint_dir, 
        save_top_k=1,
        monitor=monitor,
        mode=mode,
        verbose=True,
    )

    callbacks.append(pl_callbacks.early_stopping.EarlyStopping(patience=patience, monitor=monitor, mode=mode))
    callbacks.extend([best_saver])
    return callbacks


def get_loggers(tb_save_dir: str, tb_exp_name: str = ''):
    """

    Args:
         tb_save_dir (str): directory to save tensorboard logs
         tb_exp_name (str): experiment name used in tensorboard
    """
    return [pl_loggers.TensorBoardLogger(save_dir=tb_save_dir, name=tb_exp_name)]