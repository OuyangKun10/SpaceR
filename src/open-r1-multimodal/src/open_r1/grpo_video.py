import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer_v, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from functools import partial
import numpy as np
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["mca_acc","na_acc","format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3211264,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default="",
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        # FIXME
        # This is only for Grounding task
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. "
        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }
        QUESTION_TEMPLATE=  {"mca": "Question:\n{Question}\nFirst output the thinking process in <think> </think> tags and then output an option letter in <answer> </answer> tags.",
        "na": "Question:\n{Question}\nOutput the thinking process in <think> </think> and final answer that is a single word or phrase in <answer> </answer>tags.\nThe output answer format should be as follows: <think>...</think><answer>...</answer> Please strictly follow the format."}
        def make_conversation_video(example,question_type):
            SELECTED_TEMPLATE=QUESTION_TEMPLATE[question_type]
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": SELECTED_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root #also can input video_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
            return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }
        elif 'video' in example:
            video_path= os.path.join(image_root, example['video'])
            while not os.path.exists(video_path):
                print(f"Warning: Video {video_path} not found, randomly selecting another video")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                video_path = os.path.join(image_root, example['video'])
            question_type=example.get("question_type")
            return {
            'video': video_path, # only feeds the path
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_video(example,question_type)['prompt'] if 'video' in example else make_conversation(example)['prompt'],
        }
        else:
            image = None
            return {
                'image': image,
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
            }

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''
def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards
def mca_reward(completions, solution, **kwargs):
    def find_option_in_string(content_answer):
        """
        在字符串中查找并返回第一个找到的选项 (A, B, C, 或 D)。

        Args:
            content_answer: 需要检查的字符串。

        Returns:
            找到的选项字符 ('A', 'B', 'C', 'D')，如果都未找到则返回 None。
        """
        options = ['A', 'B', 'C', 'D']
        for option in options:
            if option in content_answer:
                return option  
        return 'None' 
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
            else:
                content_answer="None"
            content_answer=find_option_in_string(content_answer)
            if content_answer==sol or content_answer.lower()==sol.lower():
                reward=1.0
        except Exception:
            pass  # Continue to next verification method if this fails
        # print("pre_response:",content)
        print("mca_reward\n")
        print("response:",content_answer)
        print("gt:",sol)
        print("reward:",reward)
        print("\n")
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards
def na_reward(completions, solution, **kwargs):
    METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}
    WORST_CASE_FOR_METRICS = {
        "accuracy": 0.,
        "MRA:.5:.95:.05": 0.,
    }
    def fuzzy_matching(pred):
        """
        更加鲁棒的答案提取函数，从整段话中识别数字单词和数字形式并转换为数字。

        Args:
            pred: 预测字符串。

        Returns:
            提取出的数值字符串或原始字符串 (如果无法提取数值)。
        """
        pred = pred.strip().lower() # 去除首尾空白并转换为小写，方便统一处理

        # 数字单词映射 (小写)
        number_words = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
            'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
            'zero': '0' , 'a': '1', 'an': '1' # 添加 'a' 和 'an' 映射到 '1'
        }

        # 先查找数字单词
        for word, digit in number_words.items():
            if re.search(r'\b' + word + r'\b', pred): # 使用 \b 确保匹配完整单词
                return digit # 如果找到数字单词，立即返回对应的数字字符串 (优先数字单词)

        # 如果没有找到数字单词，再查找数字形式 (整数或小数)
        number_match = re.search(r'\d+(\.\d+)?', pred) # 查找数字模式
        if number_match:
            return number_match.group(0) # 返回找到的第一个数字字符串

        return "None" # 如果两种方式都找不到数字，则返回 "None"
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

    def mean_relative_accuracy(pred, target, start, end, interval):
        num_pts = (end - start) / interval + 2
        conf_intervs = np.linspace(start, end, int(num_pts))
        accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
        print("accuracy:",accuracy)
        return accuracy.mean()
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        # try:
            
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
        else:
            content_answer="None"
        for key, value in METRICS_FOR_NA.items():
            try:
                score = eval(value)(to_float(fuzzy_matching(content_answer)), to_float(sol)) 
            except TypeError:
                score = WORST_CASE_FOR_METRICS[key]
        reward=score
        print("na_reward\n")
        print("score:",score)
        print("content_answer:",content_answer)
        print("sol:",sol)
        print("\n")
            # print("pre_response:",content)
            # print("response:",to_float(fuzzy_matching(content_answer)))
            # print("gt:",to_float(sol))
            # print("reward:",reward)
            # print("\n")
        # except Exception:
        #     print("na_reward_error")
        #     pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>\s*<answer>.+?</answer>"
    # pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "mca_acc": mca_reward,
    "na_acc": na_reward,
    # "iou_v_acc": iou_v_reward
    "format": format_reward,
}
MCA_QUESTION_TYPES = [
    
    "object_rel_direction",
    # "object_rel_direction_hard",
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

def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    data_file ="/mnt/moonfs/ouyangkun-m2/dataset/ScanNet/annotations/fmt_train_counting_data_v2.jsonl"
    
    # Load the dataset
    # dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    # Load the JSONL datasets
    import json
    from datasets import Dataset
    all_data = []
    accu_reward_method_mca = ["mca_acc","format"] 
    accu_reward_method_na = ["na_acc","format"]
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            solution_value=item['solution']
            # if isinstance(solution_value, str):
            #     item['solution'] = solution_value
            # else:
            #     item['solution'] = str(solution_value)
            
            if item["question_type"] in MCA_QUESTION_TYPES:
                # print(item["question_type"])
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method_mca) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
            elif item["question_type"] in NA_QUESTION_TYPES:
                # print("use na_reward")
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method_na) 
            else:
                print("unknown_question_type")
                print(item["question_type"])
                exit()
            all_data.append(item)
    import random
    # all_data=random.sample(all_data, 100)
    # print(all_data)
    # random.shuffle(all_data)
    # print(all_data)
    # all_data=all_data[:9900]
    # print(type(all_data))
    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                'image_path': example['image_path'],  # Store path instead of loaded image
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem'] +  ' Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'}
                    ]
                }]
            }
        elif 'video' in example:
            QUESTION_TEMPLATE=  {"mca": "Question:\n{Question}\nOutput the thinking process in <think> </think> and final answer that is an option letter in <answer> </answer>tags.\nThe output answer format should be as follows: <think>...</think><answer>...</answer>. Please strictly follow the format.",
        "na": "Question:\n{Question}\nOutput the thinking process in <think> </think> and final answer that is a single word or phrase in <answer> </answer>tags.\nThe output answer format should be as follows: <think>...</think><answer>...</answer>. Please strictly follow the format."}
            question_type=example['question_type']
            SELECTED_TEMPLATE=QUESTION_TEMPLATE["mca" if question_type in MCA_QUESTION_TYPES else "na"]
            return {
                'video': example['video'],  # Store path instead of loaded image
                'problem': example['problem'],
                'solution': example['solution'],
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'video', 'text': None},
                        {'type': 'text', 'text': SELECTED_TEMPLATE.format(Question=example["problem"])  }
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': example['problem'] + ' Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'}
                    ]
                }]
            }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)
    # print(dataset)
    # print(dataset[0])
    # Split dataset for validation if requested
    splits = {'train': dataset}
    # if script_args.val_split_ratio > 0:
    #     train_val_split = dataset.train_test_split(
    #         test_size=script_args.val_split_ratio
    #     )
    #     splits['train'] = train_val_split['train']
    #     splits['validation'] = train_val_split['test']
    trainer_cls = Qwen2VLGRPOTrainer_v
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
