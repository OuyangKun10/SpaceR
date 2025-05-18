import re
from functools import partial
import os
import json
import pandas as pd
import numpy as np
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}
EXAMPLE_MAP={"table":[[0,3],[5,7]],"chair":[[9,3]],"window":[[6,5]]}
COGMAP_TEMPLATE=(
        "Question: {Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process.\n"
        "If generating a cognitive map for the video can help you answer the question, you could follow the below steps to generate a cognitive map in <map> </map> tags\n"
        "[Steps] Identify specific objects within the **video scene**, understand the spatial arrangement of the scene, and estimate the center point of each object, assuming the entire scene is represented by a 10x10 grid. These information should be summarized in <map> </map> tags.\n"
        "[Rule]1. We provide the categories to care about in this scene: {object_list}. Focus ONLY on these categories for the entire video scene.\n2. Estimate the center location of each instance within the provided categories, assuming the entire scene is represented by a 10x10 grid, considering the information from all frames.\n3. If a category contains multiple instances across all frames, include all of them.\n"
        "Present the map using dict format. Here is an example: <map>{map_example}</map>.\n"
        "If you generate a cognitive map, please put it in <map> </map> tags. Provide your detailed reasoning process between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )
OBJECT_LIST=[
    "ceiling light", "trash can", "bed", "heater", "closet", "pillow", "backpack", "chair", "refrigerator",
    "tv", "nightstand", "keyboard", "computer tower", "coat hanger", "table", "trash bin", "whiteboard",
    "monitor", "sofa", "clock", "computer mouse", "radiator", "telephone"
]
PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "Question: {Question}\n",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1)."
    },
    "thinking":
    {   "pre_prompt": COGMAP_TEMPLATE,
        "mca_post_prompt": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "na_post_prompt": " Please provide the numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.",
    }
}
def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.
def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred
def fuzzy_matching_num(pred):

    pred = pred.strip().lower() 


    number_words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'zero': '0', 'a': '1', 'an': '1'  
    }

    
    for word, digit in number_words.items():
        if re.search(r'\b' + word + r'\b', pred):  
            return digit  

    number_match = re.search(r'(\d+(\.\d+)?)', pred)  
    if number_match:
        return number_match.group(1)  

    return "None"  
def vsibench_process_results(doc): 
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc['predicted_answer']=doc['predicted_answer'].replace("Answer:","")
            doc[key] = eval(value)(fuzzy_matching(doc['predicted_answer']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching_num(doc['predicted_answer'])), to_float(doc['ground_truth'])) # Use 'predicted_answer'
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return doc 

def fuzzy_matching(pred):
    match = re.search(r'^[A-D]\.?$', pred.split(' ')[0].strip())
    if match:
        pred=match.group(0).rstrip('.').upper()
        pred=pred.strip()
        return pred
    return pred.strip() 