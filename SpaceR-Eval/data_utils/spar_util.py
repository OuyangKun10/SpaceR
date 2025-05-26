
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import re
import math

MCA_QUESTION_TYPES = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv"
]
NA_QUESTION_TYPES = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",  
]

SPECIAL_QUESTION_TYPES = [
    "view_change_infer",
]

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

Low = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",  
]

Middle = [
    "view_change_infer",
    "position_matching",
    "camera_motion_infer",
]

High = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv"
]


def sparbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") # or "These are frames of a video."
    
    if doc['task'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['task'] in MCA_QUESTION_TYPES:
        post_prompt = ""
        if doc['task'] in ['position_matching', "camera_motion_infer"]:
            post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
        post_prompt2 = "Answer with the option's letter from the given choices directly."
        return pre_prompt + "\n" + question + "\n" + post_prompt + "\n" + post_prompt2
    elif doc['task'] in SPECIAL_QUESTION_TYPES:
        post_prompt1 = ""
        post_prompt2 = ""
        return pre_prompt + "\n" + question + "\n" + post_prompt1 + "\n" + post_prompt2
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")



def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def process_na(pred, task):
    numbers = re.findall(r'(?<!\^)\d+\.\d+|(?<!\^)\d+', pred)

    # Convert the matched numbers to float or int
    extracted_numbers = [float(num) if '.' in num else int(num) for num in numbers]
    if task in ["depth_prediction_oc_mv", 
                "depth_prediction_oo_mv",
                "distance_prediction_oc_mv",
                "distance_prediction_oo_mv",
                ]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    return extracted_numbers[0]


def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def parse_instruction(instruction):
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}

def compute_vci_metric(pred, answer):

    acion_list = ["move_right", "move_left", 
                  "move_forward", "move_backward", 
                  "move_up", "move_down", 
                  "rotate_right", "rotate_left",
                  "rotate_up", "rotate_down"]
    action_order = ["move_right_left",
                    "move_up_down",
                    "move_forward_backward",
                    "rotate_right_left",
                    "rotate_up_down"]

    answer_dict = parse_instruction(pred)
    gt_dict = parse_instruction(answer)

    answer_list = []
    gt_list = []

    for action_pair in action_order:
        if action_pair == "move_right_left":
            answer_list.append(answer_dict.get("move_right", 0) - answer_dict.get("move_left", 0))
            gt_list.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif action_pair == "move_up_down":
            answer_list.append(answer_dict.get("move_up", 0) - answer_dict.get("move_down", 0))
            gt_list.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif action_pair == "move_forward_backward":
            answer_list.append(answer_dict.get("move_forward", 0) - answer_dict.get("move_backward", 0))
            gt_list.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif action_pair == "rotate_right_left":
            answer_list.append(answer_dict.get("rotate_right", 0) - answer_dict.get("rotate_left", 0))
            gt_list.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif action_pair == "rotate_up_down":
            answer_list.append(answer_dict.get("rotate_up", 0) - answer_dict.get("rotate_down", 0))
            gt_list.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))
    
    mra_list = []
    for gt, answer in zip(gt_list, answer_list):
        mra = mean_relative_accuracy(gt, answer, start=.5, end=.95, interval=.05)
        mra_list.append(mra)

    return np.mean(mra_list)

def compute_dic_metric(pred, answer):
    answer = pred
    answer_gt = answer
    if answer == answer_gt:
        return 1
    elif answer_gt in answer:  # TODO: This is a hacky way to handle the case where the answer is a subset of the predicted answer
        return 1
    
    return 0

def parse_cmi(text):
    pattern = r"\([0-9\.]+,[0-9\.]+\)|[0-9\.]+"

    matches = re.findall(pattern, text)

    if len(matches) < 2:
        if len(matches) == 1 and "(" in matches[0]:
            matches.append("0.0")
        elif len(matches) == 1 and "." in matches[0]:
            matches.insert(0, "(0.0,0.0)")

    result = []
    for match in matches:
        if "(" in match and ")" in match:
            num1, num2 = match.strip("()").split(",")
            result.extend([float(num1), float(num2)])
        else:
            result.append(float(match))

    return result

def compute_cmi_metric(pred, answer):

    pred_process = parse_cmi(pred)
    ans_process = parse_cmi(answer)
    dist = math.sqrt(
        (pred_process[0]/1000 - ans_process[0]/1000) ** 2 + 
        (pred_process[1]/1000 - ans_process[1]/1000) ** 2 + 
        (pred_process[2] - ans_process[2]) ** 2
    )
    return dist


def exact_match(pred, target):
    # return 1. if pred.lower() == target.lower() else 0.
    pred = pred.lower()
    target = target.lower()
    if pred.lower() == target.lower():
        return 1.
    elif pred in target:
        return 1.
    elif pred[0] == target:
        return 1.
    else:
        return 0

def abs_dist_norm(pred, target):
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def sparbench_process_results(doc):
    
    if doc['task'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(doc['predicted_answer'], doc['ground_truth'])
        pass
    elif doc['task'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(process_na(doc['predicted_answer'], doc['task'])), to_float(doc['ground_truth']))
            except:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    elif doc['task'] in SPECIAL_QUESTION_TYPES:
        if doc['task'] == "view_change_infer":
            try:
                doc['vci_metric'] = compute_vci_metric(doc['predicted_answer'], doc['ground_truth'])
            except:
                doc['vci_metric'] = 0

    else:
        raise ValueError(f"Unknown question type: {doc['task']}")

    return doc



import pandas as pd
import numpy as np

def sparbench_aggregate_results(results_list_of_dicts):
    """
    Aggregates evaluation results, calculating per-task scores, overall scores,
    and scores by image type. 'overall_accuracy' and category scores (Low, Middle, High)
    are calculated using weighted averages, where weights are the number of samples per task.
    """
    results_df = pd.DataFrame(results_list_of_dicts)

    if results_df.empty:
        eval_logger.warning("Input results_list_of_dicts is empty. Returning empty aggregation.")
        return {"overall": {}, "by_img_type": {}}

    output = {}
    overall_output_metrics = {} 

    metric_suffixes = []
    for m_name in METRICS_FOR_MCA.keys(): metric_suffixes.append(f"_{m_name}")
    for m_name in METRICS_FOR_NA.keys(): metric_suffixes.append(f"_{m_name}")
    if "view_change_infer" in SPECIAL_QUESTION_TYPES:
        metric_suffixes.append("_vci_metric")
    metric_suffixes.sort(key=len, reverse=True)

    def get_task_name_from_key(key_str):
        for suffix in metric_suffixes:
            if key_str.endswith(suffix):
                return key_str[:-len(suffix)]
        eval_logger.warning(f"Could not parse known metric suffix from key: '{key_str}'. Assuming key is base task name.")
        return key_str


    # 1. Populate `overall_output_metrics` with per-task average scores
    for task_name_iter, task_indices in results_df.groupby('task').groups.items():
        per_task_df = results_df.iloc[task_indices] 
        
        if task_name_iter in MCA_QUESTION_TYPES:
            for metric_name in METRICS_FOR_MCA.keys():
                mean_score = per_task_df[metric_name].mean()
                overall_output_metrics[f"{task_name_iter}_{metric_name}"] = mean_score
        elif task_name_iter in NA_QUESTION_TYPES:
            for metric_name in METRICS_FOR_NA.keys():
                mean_score = per_task_df[metric_name].mean()
                overall_output_metrics[f"{task_name_iter}_{metric_name}"] = mean_score
        elif task_name_iter in SPECIAL_QUESTION_TYPES:
            if task_name_iter == "view_change_infer" and "vci_metric" in per_task_df.columns:
                mean_score = per_task_df["vci_metric"].mean()
                overall_output_metrics[f"{task_name_iter}_vci_metric"] = mean_score
            elif task_name_iter == "view_change_infer": 
                 overall_output_metrics[f"{task_name_iter}_vci_metric"] = np.nan
                 eval_logger.warning(f"Metric column 'vci_metric' not found for task '{task_name_iter}'. Setting to NaN.")

    task_counts_overall = results_df.groupby('task').size()

    # 2. Calculate WEIGHTED 'overall_accuracy' for `overall_output_metrics`
    weighted_sum_overall_accuracy = 0.0
    total_weight_for_overall_accuracy = 0.0
    
    for metric_key, avg_task_score in overall_output_metrics.items():
        if pd.isna(avg_task_score): continue
        base_task_name = get_task_name_from_key(metric_key)
        if base_task_name and base_task_name in task_counts_overall:
            task_weight = task_counts_overall[base_task_name]
            weighted_sum_overall_accuracy += avg_task_score * task_weight
            total_weight_for_overall_accuracy += task_weight
        else:
            eval_logger.warning(f"Could not find task_name or count for metric_key '{metric_key}' during overall_accuracy calculation. Base task: '{base_task_name}'")

    if total_weight_for_overall_accuracy > 0:
        overall_output_metrics['overall_accuracy'] = weighted_sum_overall_accuracy / total_weight_for_overall_accuracy
    else:
        overall_output_metrics['overall_accuracy'] = np.nan

    # 3. Calculate WEIGHTED Low, Middle, High category scores for `overall_output`
    output['overall'] = {}
    categories = {"Low": Low, "Middle": Middle, "High": High}

    for cat_name, tasks_in_category in categories.items():
        weighted_sum_category = 0.0
        total_weight_for_category = 0.0
        for metric_key, avg_task_score in overall_output_metrics.items():
            if metric_key == 'overall_accuracy': continue
            if pd.isna(avg_task_score): continue

            base_task_name = get_task_name_from_key(metric_key)
            if base_task_name and base_task_name in task_counts_overall and base_task_name in tasks_in_category:
                task_weight = task_counts_overall[base_task_name]
                weighted_sum_category += avg_task_score * task_weight
                total_weight_for_category += task_weight
        
        if total_weight_for_category > 0:
            output['overall'][cat_name] = weighted_sum_category / total_weight_for_category
        else:
            output['overall'][cat_name] = np.nan
            
    output['overall'] = {**output['overall'], **overall_output_metrics}


    # 4. Process results grouped by 'image_type'
    img_type_output_dict = {}
    if 'image_type' not in results_df.columns:
        eval_logger.warning("'image_type' column not found in results. Skipping 'by_img_type' aggregation.")
        output['by_img_type'] = {}
    else:
        for img_type, img_type_df in results_df.groupby('image_type'):
            current_img_type_scores = {} 
            task_counts_for_img_type = img_type_df.groupby('task').size()

            # 4a. Populate current_img_type_scores with per-task average scores for this img_type
            for task_name_iter, per_task_df_img in img_type_df.groupby('task'):
                if task_name_iter in MCA_QUESTION_TYPES:
                    for metric_name in METRICS_FOR_MCA.keys():
                        current_img_type_scores[f"{task_name_iter}_{metric_name}"] = per_task_df_img[metric_name].mean()
                elif task_name_iter in NA_QUESTION_TYPES:
                    for metric_name in METRICS_FOR_NA.keys():
                        current_img_type_scores[f"{task_name_iter}_{metric_name}"] = per_task_df_img[metric_name].mean()
                elif task_name_iter in SPECIAL_QUESTION_TYPES:
                    if task_name_iter == "view_change_infer" and "vci_metric" in per_task_df_img.columns:
                        current_img_type_scores[f"{task_name_iter}_vci_metric"] = per_task_df_img["vci_metric"].mean()
                    elif task_name_iter == "view_change_infer": # Handle missing column case
                        current_img_type_scores[f"{task_name_iter}_vci_metric"] = np.nan
                        eval_logger.warning(f"Metric column 'vci_metric' not found for task '{task_name_iter}' in image_type '{img_type}'. Setting to NaN.")


            # 4b. Calculate WEIGHTED 'overall_accuracy' for current_img_type_scores
            weighted_sum_img_type_accuracy = 0.0
            total_weight_for_img_type_accuracy = 0.0
            for metric_key, avg_task_score in current_img_type_scores.items():
                if pd.isna(avg_task_score): continue
                base_task_name = get_task_name_from_key(metric_key)
                if base_task_name and base_task_name in task_counts_for_img_type:
                    task_weight = task_counts_for_img_type[base_task_name]
                    weighted_sum_img_type_accuracy += avg_task_score * task_weight
                    total_weight_for_img_type_accuracy += task_weight
                else:
                     eval_logger.warning(f"Could not find task_name or count for metric_key '{metric_key}' during img_type overall_accuracy. Base task: '{base_task_name}'")

            if total_weight_for_img_type_accuracy > 0:
                current_img_type_scores['overall_accuracy'] = weighted_sum_img_type_accuracy / total_weight_for_img_type_accuracy
            else:
                current_img_type_scores['overall_accuracy'] = np.nan

            # 4c. Calculate WEIGHTED Low, Middle, High for current_img_type_scores
            img_type_category_scores = {}
            for cat_name, tasks_in_category in categories.items():
                weighted_sum_img_cat = 0.0
                total_weight_for_img_cat = 0.0
                for metric_key, avg_task_score in current_img_type_scores.items():
                    if metric_key == 'overall_accuracy': continue
                    if pd.isna(avg_task_score): continue
                    base_task_name = get_task_name_from_key(metric_key)
                    if base_task_name and base_task_name in task_counts_for_img_type and base_task_name in tasks_in_category:
                        task_weight = task_counts_for_img_type[base_task_name]
                        weighted_sum_img_cat += avg_task_score * task_weight
                        total_weight_for_img_cat += task_weight
                
                if total_weight_for_img_cat > 0:
                    img_type_category_scores[cat_name] = weighted_sum_img_cat / total_weight_for_img_cat
                else:
                    img_type_category_scores[cat_name] = np.nan
            
            img_type_output_dict[img_type] = {**img_type_category_scores, **current_img_type_scores}
        output['by_img_type'] = img_type_output_dict
    
    def nan_to_null_serializer(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.generic): # Handles numpy float types
             if np.isnan(obj):
                 return None
             return obj.item() 
        return obj

    eval_logger.info(f"Weighted Evaluation results: {json.dumps(output, indent=2, default=nan_to_null_serializer)}")
    return output
