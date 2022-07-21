from subprocess import call
import os
#print("Enter input:")

# os.chdir('')
os.system(r'Environments\silero\Scripts\activate')
# os.chdir('...')
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
print("What would you want me to do:-")
sd.wait()
print("Got it :-")
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
    #print(decoder(example.cpu()))
    text.append(decoder(example.cpu())) 

# with open('D:\Rahul\Sangam\LXMERT\questions.txt','w') as f:
#     f.write(text[0])
print("You said")
print(text)
a =1
CLIP_activate_keywords = ['CLIP','describe','caption','described','description']
lt =[]
lt = text[0].split()
flag_CLIP = 0
for word in lt:
    if word in CLIP_activate_keywords:
        a=1
        break

    a =2
    #run clip


list_images = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg']
# print("######################")
# print(os.path.abspath("Environments\ip_cam\Scripts"))

x =0
while(x < 2):
    # print(os.path.abspath("Environments\ip_cam\Scripts"))
    # os.chdir(os.path.abspath("Environments\ip_cam\Scripts"))  # directory for environment where Camera is used
    os.system(r'Environments\ip_cam\Scripts\activate')
    # os.chdir(os.path.abspath("IP cam"))             # move to directory where script is present for Ip cam
    # call(['python',r'IP cam\ip.py'])                                # Activate Ip cam script it takes an images and saves it in a location
    if a ==1:                                               # specified in there    
        # os.chdir('Environments\clip\Scripts') # move to environment where ClIP is installed
        os.system(r'Environments\clip\Scripts\activate')                               # activate that environment
        # os.chdir('CLIP')
        #os.system('chdir')
        call(['python',r'CLIP\CLIP.py'])
    elif a ==2:
        # os.chdir('Environments\silero\Scripts')
        os.system(r'Environments\silero\Scripts\activate')
        # os.chdir('Lxmert')
        call(['python',r'Lxmert\Integ.py'])

    #print('Would you like to try again ')
    x = x+1
