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

#URL = "https://raw.githubusercontent.com/RahulraaghavA1308/Sangam/main/tes_images/4.jpeg"
URL = "IP cam\Snap.jpg"
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

import pyttsx3 


TTS = pyttsx3.init()

voices = TTS.getProperty('voices')# selecting voice
TTS.setProperty('voice',voices[1].id)# 0 for male and 1 for male
TTS.setProperty('rate',150)


TTS.say('Running Visual Question Answering')
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

# test_questions_for_url = 
with open('Questions.txt','r') as f:
    test_questions_for_url = f.readlines()    
print(test_questions_for_url)
# test_questions_for_url = [
#     #"What is there",
#     "What is the color of the chair",
#     # "Is there any laptop"
#     # "How many laptops are there?",
#     #"Is there a cord in the scene?",
#     #"What is the colour of the wire?"
# ]
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
    # pred_gqa = output_gqa["question_answering_score"].argmax(-1)
    TTS.say(test_question)
    print("Question:", test_question)
    #print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])
    print("prediction from LXMERT VQA:", vqa_answers[pred_vqa], '\n')
    TTS.say(vqa_answers[pred_vqa])
    TTS.runAndWait()

    print("Completed succesfully")

    #pip insall pyTTsx3
