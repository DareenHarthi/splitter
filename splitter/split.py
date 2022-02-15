from scipy.ndimage.morphology import binary_dilation
from pathlib import Path
from warnings import warn
import numpy as np
import librosa
import struct
from moviepy.editor import *
import soundfile as sf
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm 
import math
import webrtcvad
import os


def detect_silences(wav, audio_sampling_rate, vad_sampling_rate=16_000):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """


    int16_max = (2 ** 15) - 1


    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out. 
    vad_moving_average_width = 8
    # Maximum number of consecutive silent frames a segment can have.
    vad_max_silence_length = 6

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * vad_sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=vad_sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))

    samples_per_window = (vad_window_length * audio_sampling_rate) // 1000

    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    silences = np.where(audio_mask[:-1] != audio_mask[1:])[0]
    start_idxs = np.where(audio_mask[silences]==False)[0]
    end_idxs = np.where(audio_mask[silences]==True)[0]
    return silences[start_idxs], silences[end_idxs]

def split_on_silence(data, data_path, min_time, start_idxs, end_idxs, out_path, save_type="audio", sr=16_000, trim_silence=False ):
    start = 0 
    end_idxs[-1] = len(data)-1  
    i = 0

    # TODO trim silneces
    
    for end in tqdm(end_idxs, total=len(end_idxs)):
        
        segment = data[start:end]
        duration = len(segment)/sr
        
        if duration> min_time:   

            if save_type=='audio'  or save_type=='all' : 
                sf.write(f'{out_path}/audio/{i}.wav', segment, sr)


            if save_type=='video' or save_type=='all' :    
                start_time = start/sr
                end_time = end/sr
                ffmpeg_extract_subclip(data_path, start_time, end_time, targetname=f'{out_path}/video/{i}.mp4')
                
                
            
            i+=1
            start = end
            
            


def split(path, min_time, save_path="", save_type="audio", trim_silence=False):
    wav, audio_sampling_rate = librosa.load(path)
    #TODO
    # sampling_rates = [8_000, 16_000, 32_000]
    # vad_sampling_rate = min([ ])    

    save_path = save_path if save_path else os.getcwd()

    start_idxs, end_idxs = detect_silences(wav, audio_sampling_rate)

    if save_type=='audio'  or save_type=='all' :
        Path(f'{save_path}/audio').mkdir(parents=True, exist_ok=True)

    if save_type=='video' or save_type=='all' :
        Path(f'{save_path}/video').mkdir(parents=True, exist_ok=True)

    split_on_silence(wav, path, min_time, list(start_idxs), list(end_idxs), save_path, save_type, audio_sampling_rate, trim_silence) 

    print("Done!!!!!")
    

