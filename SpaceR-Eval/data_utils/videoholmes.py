from util import *
import time
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
from loguru import logger as eval_logger
from .internvl_video_utils import load_video_internvl 
QUESTION_TEMPLATE = (
        "Question: {Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )
PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "Question: {Question}\n",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1)."
    },
    "thinking":
    {
        "pre_prompt": QUESTION_TEMPLATE,
        "mca_post_prompt": "Please provide only the single option letter (e.g., A, B, C, D, E, F, etc.) within the <answer> </answer> tags.",#"Please output the thinking process in <think>...</think> and final answer that is an option letter in <answer>...</answer>.\nThe output answer format should be as follows: <think>...</think><answer>...</answer>. You must output an option letter as the final answer in <answer>...</answer>.",
        "na_post_prompt": "Please provide the numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.",#"Please output the thinking process in <think>...</think> and final answer that is a single word or phrase in <answer>...</answer>.\nThe output answer format should be as follows: <think>...</think><answer>...</answer>. You must output a single word or phrase as the final answer in <answer>...</answer>."
        "special_post_prompt": "First output the thinking process in <think> </think> tags and then output the answer in <answer> </answer> tags.",
    },
    "gemini_api": {
        "pre_prompt": "",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Do not response anything other than a single number!"
    },
    "gpt4v": {
        "pre_prompt": "",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Do not response anything other than a single number!"
    }
}



def evaluate_videoholmes(rank, world_size, parquet_file, video_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, fps=1, target_resolution=(256, 256), debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="default", use_cognitive_map=True, offload_cogmap=False, cogmap_file_path=None, cogmap_id_key="id", cogmap_cog_key="cog_map", cogmap_data_format="list_dict"):
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} 使用设备: {device}")

    df = pd.read_json(parquet_file)
    if debug:
        df = df.sample(n=debug_size)
        logger.info(f"进程 {rank} Debug 模式开启，随机处理 {debug_size} 条数据。")

    if world_size > 1:
        df_shard = np.array_split(df, world_size)[rank]
    else:
        df_shard = df
    logger.info(f"Rank {rank} Shard size: {len(df_shard)}")

    if 'Qwen2.5' in model_name:
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        processor.tokenizer.padding_side = 'left'
    elif 'Kimi-VL' in model_name:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    elif 'VideoLLaMA3' in model_name:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    elif 'InternVL' in model_name:
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    elif 'MiniCPM-V' in model_name:
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if world_size == 1 and len(gpu_ids.split(',')) > 1:
        if 'Qwen2.5' in model_name:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True,
            )
        elif 'VideoLLaMA3' in model_name:
            processor = AutoProcessor.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True,)
        elif 'Kimi-VL' in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True,
            )
        elif 'InternVL' in model_name:
            model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_flash_attn=True,
                    trust_remote_code=True)
        elif 'MiniCPM-V' in model_name:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto",
                attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
        model = accelerator.prepare(model)
        model.eval()
    else:
        if 'Qwen2.5' in model_name:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval().to(device)
        elif 'Kimi-VL' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",trust_remote_code=True).eval().to(device)
        elif 'VideoLLaMA3' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",trust_remote_code=True).eval().to(device)
        elif 'InternVL' in model_name:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True,trust_remote_code=True).eval().to(device)
        elif 'MiniCPM-V' in model_name:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",trust_remote_code=True).eval().to(device)
        model = accelerator.prepare(model)

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has empty shard, skipping processing.")
        return os.path.join(output_dir, f"Video-Holmes_results_rank_{rank}.jsonl"), 0

    prompt_template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    
    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"进程 {rank}", total= (total_samples + batch_size - 1) // batch_size):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]
        batch_messages_list = []
        batch_row_infos = []
        prompt_list = []
        predicted_answers_batch=[]
        if prompt_type=="default":
            max_new_token=128
        else:
            max_new_token=1024
        for _, row in batch_df.iterrows():
            question = row.get('Question')
            options = row.get('Options')
            options = ', '.join([f"{key}: {value}" for key, value in options.items()])
            question = question+ "\nOptions:\n" + options
            video_id = row.get('video ID')
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue
            prompt_text = prompt_template["pre_prompt"].format(Question=question)

            prompt_text += "\n" + prompt_template["mca_post_prompt"]
            frames, timestamps, duration = load_video_frames(video_path, num_frames, fps, target_resolution) 
            row['duration']=duration
            if frames is None:
                continue
            if 'InternVL' in model_name:
                pixel_values, num_patches_list = load_video_internvl(video_path, num_segments=num_frames)
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                video_prefix = ''.join([f'Frame{i+1}: <image>\\n' for i in range(len(num_patches_list))])
                prompt_text = video_prefix + prompt_text
                response, _ = model.chat(
                    processor, pixel_values, prompt_text,
                    generation_config={"max_new_tokens": max_new_token, "do_sample": False},
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
                predicted_answers_batch.append(response)
            elif 'MiniCPM-V' in model_name:
                msgs = [
                            {'role': 'user', 'content': frames + [prompt_text]}, 
                        ]
                params={}
                params["use_image_id"] = False
                params["max_slice_nums"] = 2
                params["max_new_tokens"]=max_new_token
                params["temperature"]=0.01
                response=model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=processor,
                    **params
                )
                predicted_answers_batch.append(response)
            elif 'VideoLLaMA3' in model_name:
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": num_frames}},
                            {"type": "text", "text": prompt_text},
                        ]
                    },
                ]
                inputs = processor(conversation=conversation, return_tensors="pt")
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                output_ids = model.generate(**inputs, max_new_tokens=max_new_token)
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                predicted_answers_batch.append(response)
            prompt_list.append(prompt_text)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            batch_messages_list.append(messages)
            batch_row_infos.append(row)

        if 'Qwen2.5' in model_name or 'Kimi-VL' in model_name:
            if not batch_messages_list:
                continue

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages_list
            ]
            image_inputs_batch, video_inputs_batch = process_vision_info(batch_messages_list)
            inputs_batch = processor(
                text=texts,
                images=image_inputs_batch,
                videos=video_inputs_batch,
                padding=True,
                return_tensors="pt",
            ).to(device)
            try:
                generated_ids_batch = model.generate(**inputs_batch, use_cache=True, max_new_tokens=max_new_token, temperature=1.0,top_p=1.0)
                generated_ids_trimmed_batch = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
                ]
                predicted_answers_batch = processor.batch_decode(
                    generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            except Exception as e:
                logger.error(f"进程 {rank} batch 推理失败: {e}")
                predicted_answers_batch = [""] * len(batch_messages_list)

        for i, predicted_answer in enumerate(predicted_answers_batch):
            row = batch_row_infos[i]
            duration=row['duration']
            prompt_text = prompt_list[i]
            results.append({
                'id': row.get('Question ID'),
                'question': row.get('Question'),
                'ground_truth': row.get('Answer'),
                'predicted_answer': predicted_answer,
                'task': row.get('Question Type'),
                'promt': prompt_text,
                'duration':duration
            })

    process_output_file = os.path.join(output_dir, f"Video-Holmes_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} 结果已保存到: {process_output_file}, 进程运行时间: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process
def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEF]", s):
        return ""
    matches = re.search(r'[ABCDEF]', s)
    if matches is None:
        return ""
    return matches[0]

def videoholmes_aggregate_results(results):
    results_df = pd.DataFrame(results) 

    output = {}


    for task, task_group in results_df.groupby('task'):

        task_accuracy = task_group['accuracy'].mean()
        output[f"{task}_accuracy"] = task_accuracy

    output['overall_accuracy'] = results_df['accuracy'].mean()



    eval_logger.info(f"Evaluation results: {output}")

    return output

def videoholmes_eval(jsonl_file_path,mode="thinking"):
    results = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if mode=="thinking" and "<answer>" in doc["predicted_answer"]:
                doc["predicted_answer"]=extract_answer_text(doc["predicted_answer"])
            doc["predicted_answer"]=extract_characters_regex(doc["predicted_answer"])
            if doc["predicted_answer"] is None:
                doc["predicted_answer"] = "ERROR"
            if doc["predicted_answer"]== doc["ground_truth"]:
                doc['accuracy']=1.0
            else:
                doc['accuracy']=0.0
            results.append(doc)
    aggregated_results = videoholmes_aggregate_results(results) 
    return aggregated_results 


