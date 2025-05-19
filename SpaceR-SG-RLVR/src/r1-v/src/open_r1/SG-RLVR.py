import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified, SGRLVRTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from functools import partial
import numpy as np
import random
import copy
import sys
import math
import json
import ast
from tqdm import tqdm
from extract_map import extract_map_data,calculate_prediction_score,read_data
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
def accuracy_reward(completions, solution,path, **kwargs):
    def fuzzy_matching(pred):
        pred = pred.strip().lower() 
        number_words = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
            'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
            'zero': '0' , 'a': '1', 'an': '1' 
        }


        for word, digit in number_words.items():
            if re.search(r'\b' + word + r'\b', pred):
                return digit 

        number_match = re.search(r'\d+(\.\d+)?', pred) 
        if number_match:
            return number_match.group(0) 

        return "None" 
    def to_float(pred):
        try:
            pred = float(pred)
        except BaseException as e:
            pred = None
        return pred
    def exact_match(pred, target):
        return 1. if pred.lower() == target.lower() else 0.

    def abs_dist_norm(pred, target):
        return abs(pred - target) / target

    def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
        num_pts = (end - start) / interval + 2
        conf_intervs = np.linspace(start, end, int(num_pts))
        accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
        return accuracy.mean()

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    METRICS_FOR_NA = {
        "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
    }
    WORST_CASE_FOR_METRICS = {
        "accuracy": 0.,
        "MRA:.5:.95:.05": 0.,
    }
    def compute_na_accuracy(content_answer,sol):
        try:
            score = mean_relative_accuracy(content_answer, sol)
        except Exception as e:
            score = WORST_CASE_FOR_METRICS.get(metric_key, 0.0) 
            print(f"Error calculating NA metric '{metric_key}': {e}")


        return score

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return  d[m][n] / max(1, m)
    def fmt_map(map):
        return fmt_map
    def compute_map_score(content_answer,sol,object_list):
        def extract_map(content_answer):
            pattern = r'<map>\s*(.*?)\s*</map>'
            match = re.search(pattern, content_answer, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        map_response=extract_map(content_answer)
        fmt_map_response=extract_map_data(map_response,object_list)
        overall_score=calculate_prediction_score(fmt_map_response,sol,10)
        return overall_score
    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    for content, sol, pa in zip(contents, solution, path):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
                if reward ==1.0 and '<map>' in content and '</map>' in content:
                    map_solution=MAP_DATA[os.path.splitext(os.path.basename(pa))[0]]
                    cognitive_map=map_solution['cognitive_map']
                    object_list=list(cognitive_map.keys())
                    map_score=compute_map_score(content,cognitive_map,object_list)
                    if map_score > 0:
                        reward=reward+map_score
                    else:
                        reward=0.0
            elif question_type == "numerical":

                gt_number = to_float(gt_ans)
                out_number = to_float(fuzzy_matching(output_ans))
                if gt_number is None or out_number is None:
                    reward = 0.0
                else:
                    reward = compute_na_accuracy(out_number,gt_number)
                    if reward>0.5 and '<map>' in content and '</map>' in content:
                        map_solution=MAP_DATA[os.path.splitext(os.path.basename(pa))[0]]
                        cognitive_map=map_solution['cognitive_map']
                        object_list=list(cognitive_map.keys())
                        map_score=compute_map_score(content,cognitive_map,object_list)
                        if map_score > 0:
                            reward=reward+map_score
                        else:
                            reward=0.0
    
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    format_re=[1.0 if match else 0.0 for match in matches]
    return format_re


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
   
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    
    
    def load_map(path):
        map_data={}
        data=read_data(path)
        for item in tqdm(data,desc='loading map'):
            map_data[item["video_id"]]={"cognitive_map":item["cognitive_map"],"object_list":item["object_list"]}
        return map_data
    
    global MAP_DATA
    MAP_DATA=load_map("annotation/cognitive_map.jsonl")
    EXAMPLE_MAP={"table":[[0,3],[5,7]],"chair":[[9,3]],"window":[[6,5]]}
    QUESTION_TEMPLATE = (
        "Question: {Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )
 
    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }
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
    def make_conversation_image_and_video_map(example):
        
   
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        if example['data_source']=='SR_dataset':
            video_id=os.path.splitext(os.path.basename(example['path']))[0]
            object_list=list(MAP_DATA[video_id]['cognitive_map'].keys())
            prompt=COGMAP_TEMPLATE.format(Question=question,object_list=object_list,map_example=EXAMPLE_MAP) + TYPE_TEMPLATE[example['problem_type']]
        else:
            prompt=QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]

        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                        ]
                }]
            }
        
        return msg
 
 
    
   
    dataset = dataset.map(make_conversation_image_and_video_map)

    
    trainer_cls = SGRLVRTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
