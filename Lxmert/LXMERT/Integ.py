#imports

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

import torch
import zipfile
import torchaudio
from glob import glob

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from pydub import AudioSegment

from transformers import pipeline

import cv2
from IPython.display import Image, display
import PIL.Image
import io
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils

#Recording the voice

freq = 16000
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
wv.write("Meta1.wav", recording, freq, sampwidth=2)

#Conversion from .wav to .mp3

# AudioSegment.from_wav("/content/Meta1.wav").export("/content/Meta.mp3", format="mp3")?\

#Noise Reduction

f_name = "Meta1.wav"
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

output = model(input)
    
question = []

for example in output:
    print(decoder(example.cpu()))
    question.append(decoder(example.cpu()))

#LXMERT

#URL = "https://raw.githubusercontent.com/RahulraaghavA1308/Sangam/main/tes_images/4.jpeg"
URL = "image.jpeg"
OBJ_URL = "OBJ_URL.txt"
ATTR_URL = "ATTR_URL.txt"
GQA_URL = "GQA_URL.txt"
VQA_URL = "VQA_URL.txt"

objids = utils.get_data(OBJ_URL)
attrids = utils.get_data(ATTR_URL)
gqa_answers = utils.get_data(GQA_URL)
vqa_answers = utils.get_data(VQA_URL)


def showarray(a, fmt="jpeg"):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

image_preprocess = Preprocess(frcnn_cfg)

from transformers import LxmertForQuestionAnswering, LxmertTokenizer

lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")

# image viz
img = cv2.imread(URL, cv2.IMREAD_COLOR)
img1 = torch.from_numpy(img)
frcnn_visualizer = SingleImageViz(img, id2obj=objids, id2attr=attrids)
# run frcnn
images, sizes, scales_yx = image_preprocess(img1)
output_dict = frcnn(
    images,
    sizes,
    scales_yx=scales_yx,
    padding="max_detections",
    max_detections=frcnn_cfg.max_detections,
    return_tensors="pt",
)
# add boxes and labels to the image

frcnn_visualizer.draw_boxes(
    output_dict.get("boxes"),
    output_dict.pop("obj_ids"),
    output_dict.pop("obj_probs"),
    output_dict.pop("attr_ids"),
    output_dict.pop("attr_probs"),
)
 #showarray(frcnn_visualizer._get_buffer())


# test_questions_for_url = [
#     "What is the color of the floor",
#     "Is there a car",
#     "Which direction should I turn?",
#     #"Is there a cord in the scene?",
#     #"What is the colour of the wire?"
# ]

test_questions_for_url = question

import torch.nn.functional as F
# Very important that the boxes are normalized
normalized_boxes = output_dict.get("normalized_boxes")
features = output_dict.get("roi_features")

for test_question in test_questions_for_url:
    # run lxmert
    test_question = [test_question]

    inputs = lxmert_tokenizer(
        test_question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    # run lxmert(s)
    output_gqa = lxmert_gqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    output_vqa = lxmert_vqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    # get prediction
    pred_vqa = output_vqa["question_answering_score"].argmax(-1)
    pred_gqa = output_gqa["question_answering_score"].argmax(-1)
    print("Question:", test_question)
    #print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])
    print("prediction from LXMERT VQA:", vqa_answers[pred_vqa], '\n')

