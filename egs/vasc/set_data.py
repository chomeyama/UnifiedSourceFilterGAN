import os
import shutil
import librosa
import soundfile as sf
import subprocess
import sys

root_dir = 'data/'
scp_dir = root_dir + 'scp/'
wav_dir = root_dir + 'wav/'

spks = ['fujitou_angry', 'fujitou_happy', 'fujitou_normal',
        'tsuchiya_angry', 'tsuchiya_happy', 'tsuchiya_normal',
        'uemura_angry', 'uemura_happy', 'uemura_normal']


# Write scp files
with open(scp_dir + 'vasc_train_22kHz.scp', mode='w') as f:
    for spk in spks:
        for i in range(80):
            f.write(wav_dir + 'vasc_training/' + spk + '/' + spk + f'_{i+1:03}.wav\n')
with open(scp_dir + 'vasc_valid_22kHz.scp', mode='w') as f:
    for spk in spks:
        for i in range(80, 90):
            f.write(wav_dir + 'vasc_training/' + spk + '/' + spk + f'_{i+1:03}.wav\n')
with open(scp_dir + 'vasc_eval_22kHz.scp', mode='w') as f:
    for spk in spks:
        for i in range(90, 100):
            f.write(wav_dir + 'vasc_evaluation/' + spk + '/' + spk + f'_{i+1:03}.wav\n')


# Copy wav files
for spk in spks:
    from_dir = wav_dir + spk
    files = os.listdir(from_dir)
    files.sort()
    
    to_dir = wav_dir + 'vasc_evaluation/' + spk
    for f in files[90:]:
        shutil.copyfile(from_dir + '/' + f, to_dir + '/' + f)

    to_dir = wav_dir + 'vasc_training/' + spk
    for f in files[:90]:
        shutil.copyfile(from_dir + '/' + f, to_dir + '/' + f)


# Dicrease sampling rate and trim silent segment
for spk in spks:
    wav_dir = root_dir + 'wav/vasc_evaluation/' + spk + '/'
    files = os.listdir(wav_dir)
    for f in files:
        file_path = wav_dir + f
        y, sr = librosa.core.load(file_path, sr=22050, mono=True) # 22050Hz
        yt, index = librosa.effects.trim(y) # Trim the beginning and ending silence
        sf.write(file_path, yt, sr, subtype="PCM_16")

    wav_dir = root_dir + 'wav/vasc_training/' + spk + '/'
    files = os.listdir(wav_dir)
    for f in files:
        file_path = wav_dir + f
        y, sr = librosa.core.load(file_path, sr=22050, mono=True) # 22050Hz
        yt, index = librosa.effects.trim(y) # Trim the beginning and ending silence
        sf.write(file_path, yt, sr, subtype="PCM_16")