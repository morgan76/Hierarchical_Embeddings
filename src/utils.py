import numpy as np
import os
from pathlib import Path
import librosa
import ujson
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import jams

class FileStruct:
    def __init__(self, audio_file):
        audio_file = Path(audio_file)
        self.track_name = audio_file.stem
        self.audio_file = audio_file
        self.ds_path = audio_file.parents[1]
        self.json_file = self.ds_path.joinpath('features', self.track_name
                                               + '.json')
        self.ref_file = self.ds_path.joinpath('references', self.track_name
                                              + '.jams')
        self.beat_file = self.ds_path.joinpath('features', self.track_name+'_beats_'
                                              + '.json')
        self.predictions_file = self.ds_path.joinpath('predictions', self.track_name
                                              + '.jams')
        self.audio_npy_file = self.ds_path.joinpath('audio_npy', self.track_name
                                              + '.npy')                                  

    def __repr__(self):
        """Prints the file structure."""
        return "FileStruct(\n\tds_path=%s,\n\taudio_file=%s,\n\test_file=%s," \
            "\n\json_file=%s,\n\tref_file=%s\n)" % (
                self.ds_path, self.audio_file, self.est_file,
                self.json_file, self.ref_file)
    
def clean_tracklist_audio(data_path, annotations=False, n_samples=1e6):
    tracklist = librosa.util.find_files(os.path.join(data_path, 'audio'), ext=['wav', 'mp3', 'aiff', 'flac'])[:n_samples]
    tracklist_clean = []
    for song in tqdm(tracklist):
        file_struct = FileStruct(song)
        if os.path.isfile(file_struct.beat_file) and os.path.isfile(file_struct.audio_npy_file):
            if annotations and os.path.isfile(file_struct.ref_file):
                tracklist_clean.append(song)
            elif not annotations:
                tracklist_clean.append(song)
    return tracklist_clean

def read_beats(json_file):
    with open(json_file, 'r') as f:
        out_json = ujson.load(f)
    beat_strings = out_json["est_beats"].split('[')[1].split(']')[0].split(',')
    duration = float(out_json["globals"]["duration"])
    beat_times = [int(i) for i in beat_strings]
    return beat_times, duration

def make_splits(data_path, val_data_path, p=.25, seed=42):
    tracklist = []
    for i in data_path:
        tracklist += clean_tracklist_audio(i, annotations=True)
    if val_data_path != None:
        valid_tracklist = clean_tracklist_audio(val_data_path, annotations=True)
        return tracklist, valid_tracklist
    if p == 0:
        return tracklist, tracklist
    else:
        train_tracklist, valid_tracklist = train_test_split(tracklist, test_size=p, random_state=seed)
        return train_tracklist, valid_tracklist

def get_ref_labels(file_struct, level, annot=0):

    track = str(file_struct.audio_file)
    
    jam = jams.load(str(file_struct.ref_file), validate=False)
    duration = jam.file_metadata.duration
    ref_times, ref_labels = read_references(file_struct.audio_file, False)
    #if 'JSD' not in track:
    #    ref_times, ref_labels = remove_empty_segments(ref_times, ref_labels, th=2)
    return ref_labels, ref_times, duration

def read_references(audio_path, estimates, annotator_id=0, hier=False):
    """Reads the boundary times and the labels.

    Parameters
    ----------
    audio_path : str
        Path to the audio file

    Returns
    -------
    ref_times : list
        List of boundary times
    ref_labels : list
        List of labels

    Raises
    ------
    IOError: if `audio_path` doesn't exist.
    """
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))


    if not estimates:
    # Read references
        try:
            jam_path = os.path.join(ds_path, 'references',
                                    os.path.basename(audio_path)[:-4] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, 'references',
                                    os.path.basename(audio_path)[:-5] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)
    else:
        try:
            jam_path = os.path.join(ds_path, 'references/estimates/',
                                    os.path.basename(audio_path)[:-4] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, 'references/estimates/',
                                    os.path.basename(audio_path)[:-5] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)


    ##################
    low = True # Low parameter for SALAMI
    ##################


    if not hier:
        if low:  
            try:
                ann = jam.search(namespace='segment_salami_lower.*')[0]
            except:
                try:
                    ann = jam.search(namespace='segment_salami_upper.*')[0]
                except:
                    ann = jam.search(namespace='segment_.*')[annotator_id]
        else:
            try:
                ann = jam.search(namespace='segment_salami_upper.*')[0]
            except:
                ann = jam.search(namespace='segment_.*')[annotator_id]
        
        ref_inters, ref_labels = ann.to_interval_values()
        ref_times =  intervals_to_times(ref_inters)
        
        return ref_times, ref_labels

    else:

        list_ref_times, list_ref_labels = [], []
        upper = jam.search(namespace='segment_salami_upper.*')[0]
        ref_inters_upper, ref_labels_upper = upper.to_interval_values()
        
        list_ref_times.append( intervals_to_times(ref_inters_upper))
        list_ref_labels.append(ref_labels_upper)

        annotator = upper['annotation_metadata']['annotator']
        lowers = jam.search(namespace='segment_salami_lower.*')

        for lower in lowers:
            if lower['annotation_metadata']['annotator'] == annotator:
                ref_inters_lower, ref_labels_lower = lower.to_interval_values()
                list_ref_times.append( intervals_to_times(ref_inters_lower))
                list_ref_labels.append(ref_labels_lower)

        return list_ref_times, list_ref_labels
    
def times_to_intervals(times):
    """ Copied from MSAF.
    Given a set of times, convert them into intervals.
    Parameters
    ----------
    times: np.array(N)
        A set of times.
    Returns
    -------
    inters: np.array(N-1, 2)
        A set of intervals.
    """
    return np.asarray(list(zip(times[:-1], times[1:])))


def intervals_to_times(inters):
    """ Copied from MSAF.
    Given a set of intervals, convert them into times.
    Parameters
    ----------
    inters: np.array(N-1, 2)
        A set of intervals.
    Returns
    -------
    times: np.array(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)


def downsample_frames(beat_frames, max_length=600):
    while len(beat_frames)>max_length:
        beat_frames = beat_frames[::2]
    return beat_frames


def remove_empty_segments(times, labels, th=2):
    """Removes empty segments if needed."""
    assert len(times) - 1 == len(labels)
    inters = times_to_intervals(times)
    new_inters = []
    new_labels = []
    j = 0
    for inter, label in zip(inters, labels):
        if inter[0] < inter[1] - th:
            new_inters.append(inter)
            new_labels.append(label)
        elif j == 0:
            if inter[0] != inter[1] :
                new_inters.append(inter)
                new_labels.append(label)
        j += 1
        
    return intervals_to_times(np.asarray(new_inters)), new_labels