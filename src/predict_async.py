# -*- coding: utf-8 -*-
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
from eval_segmentation import *
import joblib
import warnings
warnings.filterwarnings("ignore")
from models import *



P3s = []
R3s = []
F3s = []


def print_latex_format(out):
    P1_mean = np.round(np.nanmean(out[:,0]), 3)
    P1_std = np.round(np.nanstd(out[:,0]), 2)
    R1_mean = np.round(np.nanmean(out[:,1]), 3)
    R1_std = np.round(np.nanstd(out[:,1]), 2)
    F1_mean = np.round(np.nanmean(out[:,2]), 3)
    F1_std = np.round(np.nanstd(out[:,2]), 2)

    P3_mean = np.round(np.nanmean(out[:,3]), 3)
    P3_std = np.round(np.nanstd(out[:,3]), 2)
    R3_mean = np.round(np.nanmean(out[:,4]), 3)
    R3_std = np.round(np.nanstd(out[:,4]), 2)
    F3_mean = np.round(np.nanmean(out[:,5]), 3)
    F3_std = np.round(np.nanstd(out[:,5]), 2)

    PFC_mean = np.round(np.nanmean(out[:,6]), 3)
    PFC_std = np.round(np.nanstd(out[:,6]), 2)

    NCE_mean = np.round(np.nanmean(out[:,7]), 3)
    NCE_std = np.round(np.nanstd(out[:,7]), 2)

    print('\n')
    print('Latex format: \n')

    print('& $.', int(P1_mean*1000), '{\scriptstyle \pm .',int(P1_std*100), '}$')
    print('& $.', int(R1_mean*1000), '{\scriptstyle \pm .',int(R1_std*100), '}$')
    print('& $.', int(F1_mean*1000), '{\scriptstyle \pm .',int(F1_std*100), '}$')
    print('& $.', int(P3_mean*1000), '{\scriptstyle \pm .',int(P3_std*100), '}$')
    print('& $.', int(R3_mean*1000), '{\scriptstyle \pm .',int(R3_std*100), '}$')
    print('& $.', int(F3_mean*1000), '{\scriptstyle \pm .',int(F3_std*100), '}$')
    print('& $.', int(PFC_mean*1000), '{\scriptstyle \pm .',int(PFC_std*100), '}$')
    print('& $.', int(NCE_mean*1000), '{\scriptstyle \pm .',int(NCE_std*100), '}$ \\')



def apply_async_with_callback(embeddings_list, tracklist, segmentation, clustering, level, masks, embedding_levels=[], annot=0):

    
    jobs = [ joblib.delayed(eval_segmentation_async)(audio_file=i[0], embeddings=i[1], level=0) for i in tqdm(embeddings_list) ] #audio_file, embeddings, level
    out = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
    out = np.array(out)

    out = out.astype(float)
    print('P1 =', np.mean(out[:,0]),'+/-', np.std(out[:,0]))
    print('R1 =', np.mean(out[:,1]),'+/-', np.std(out[:,1]))
    print('F1 =', np.mean(out[:,2]),'+/-', np.std(out[:,2]))
    print('P3 =', np.mean(out[:,3]),'+/-', np.std(out[:,3]))
    print('R3 =', np.mean(out[:,4]),'+/-', np.std(out[:,4]))
    print('F3 =', np.mean(out[:,5]),'+/-', np.std(out[:,5]))
    print('PFC =', np.mean(out[:,6]),'+/-', np.std(out[:,6]))
    print('NCE =', np.mean(out[:,7]),'+/-', np.std(out[:,7]))
    print_latex_format(out)
    return out
