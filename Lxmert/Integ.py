from IPython.display import Audio
from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io
import librosa
from librosa import display
# %matplotlib inline

import torch
import zipfile
import torchaudio
from glob import glob

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from pydub import AudioSegment

#Recording the voice

freq = 44100
duration = 5

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
print("Speak now :-")
sd.wait()
print("Wait now :-")
wv.write("test_3.wav", recording, freq, sampwidth=2)

#Conversion from .wav to .mp3

# AudioSegment.from_wav("D:\Rahul\Sangam\Speech2Text\Silero\Meta1.wav").export("Meta.mp3", format="mp3")

#Noise Reduction

f_name = "test_3.wav"
data, rate = librosa.load(f_name)

# librosa.display.waveplot(data, rate)

# fig, ax = plt.subplots(figsize=(20,3))
# ax.plot(data)

reduced_noise = nr.reduce_noise(y = data, sr=rate, n_std_thresh_stationary=1.5,stationary=True)

# librosa.display.waveplot(reduced_noise, rate)

# fig, ax = plt.subplots(figsize=(20,3))
# ax.plot(reduced_noise)

sf.write('Meta.wav', reduced_noise, rate)

#Speech to text

test_files = glob('Meta.wav')
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]), device=device)
text = []
output = model(input)
# print(decoder(output.cpu()))
for example in output:
    print(decoder(example.cpu()))
    text.append(decoder(example.cpu())) 

with open('Lxmert\LXMERT\Questions.txt','w') as f:
    f.write(text[0])

print(text)
from subprocess import call
# call(['cd','D:\Rahul\Sangam\Speech2Text\Silero\A'])
print("working")
import os

# os.chdir("Environments\lxmert")
os.system(r'Environments\lxmert\Scripts\activate')
# os.system('chdir')
# os.chdir("Lxmert\LXMERT")
call(['python','Lxmert\LXMERT\LXMERT.py'])
# os.system('chdir')

# os.system('deactivate')
# os.system('cd silero')
# os.system("cd ..")
# os.system("cd LXMERT")
# os.system("cd SANGAM")
# os.system("cd Scripts")
# os.system("activate")
# os.system("cd ..")
# os.system("cd ..")
# os.system("python LXMERT.py")
# os.system('xya')
# os.system('cd..')
# os.system('cd..')

# os.system('chdir')
# os.system("cd D:\Rahul\Sangam\LXMERT\SANGAM\Scripts")

# os.system("activate")
# os.system("cd D:\Rahul\Sangam\LXMERT")
# os.system("python LXMERT.py")