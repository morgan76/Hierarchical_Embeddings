import numpy as np
import mir_eval
import librosa
import torch.nn.functional as F

from utils import *
from algorithms.scluster.main2 import do_segmentation as scluster
import warnings
warnings.filterwarnings("ignore")




def eval_segmentation_async(audio_file, embeddings, level):
    
    file_struct = FileStruct(audio_file)
    ref_labels, ref_times, duration = get_ref_labels(file_struct, level)

    beat_frames, duration = read_beats(FileStruct(audio_file).beat_file)
    beat_frames = librosa.util.fix_frames(beat_frames)

    beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)

    
    ref_inter = times_to_intervals(ref_times)

    
    (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=ref_inter.max())
    
    temp_P1, temp_R1, temp_F1 = [], [], []
    temp_P3, temp_R3, temp_F3 = [], [], []
    temp_PFC, temp_NCE = [], []
    
    
    est_inter_list, est_labels_list, Cnorm = scluster(embeddings.T, embeddings.T, True)
    
    

    for est_idxs, est_labels in zip(est_inter_list, est_labels_list):
        est_idxs = [beat_times[int(i)] for i in est_idxs]
        est_idxs, est_labels = remove_empty_segments(est_idxs, est_labels)
        est_inter = times_to_intervals(est_idxs)
        est_inter, est_labels = mir_eval.util.adjust_intervals(est_inter, list(est_labels), t_min=0, t_max=ref_inter.max())
        
        P1, R1, F1 = mir_eval.segment.detection(ref_inter,
                                                est_inter,
                                                window=.5,
                                                trim=True)                                              
        P3, R3, F3 = mir_eval.segment.detection(ref_inter,
                                                est_inter,
                                                window=3,
                                                trim=True) 

        precision, recall, f_PFC = mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels)
        S_over, S_under, S_F = mir_eval.segment.nce(ref_inter, ref_labels, est_inter, est_labels)
        
        temp_P1.append(P1)
        temp_R1.append(R1)
        temp_F1.append(F1)
        temp_P3.append(P3)
        temp_R3.append(R3)
        temp_F3.append(F3)
        temp_PFC.append(f_PFC)
        temp_NCE.append(S_F)


    ind_max_F1 = np.argmax(temp_F1)
    F1 = temp_F1[ind_max_F1]
    R1 = temp_R1[ind_max_F1]
    P1 = temp_P1[ind_max_F1]
    ind_max_F3 = np.argmax(temp_F3)
    F3 = temp_F3[ind_max_F3]
    R3 = temp_R3[ind_max_F3]
    P3 = temp_P3[ind_max_F3]
    ind_max_PFC = np.argmax(temp_PFC)
    ind_max_NCE = np.argmax(temp_NCE)
    PFC = temp_PFC[ind_max_PFC]
    NCE = temp_NCE[ind_max_NCE]
    


    return  P1, R1, F1, P3, R3, F3, PFC, NCE
        