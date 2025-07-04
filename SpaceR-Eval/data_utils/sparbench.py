from util import *
import time
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import base64
from .spar_util import sparbench_process_results,sparbench_aggregate_results
from .internvl_video_utils import load_video_internvl2_5, load_image 

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
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1).",
        "special_post_prompt": "Please output the answer directly.",
    },
    "thinking":
    {
        "pre_prompt": QUESTION_TEMPLATE,
        "mca_post_prompt": "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "na_post_prompt": "Please provide the numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.",
        "special_post_prompt": "First output the thinking process in <think> </think> tags and then output the answer in <answer> </answer> tags.",
    },
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

def evaluate_sparbench(rank, world_size, parquet_file, video_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, fps=1, target_resolution=(256, 256), debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="default"):
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    data_frames = []
    for file_path in parquet_file:
        file_path=os.path.join("SPAR-Bench/data",file_path)
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            data_frames.append(df)
        else:
            print(f"File {file_path} does not exist, skipping.")

    df = pd.concat(data_frames, ignore_index=True)
    if debug:
        df = df.sample(n=debug_size)
        logger.info(f"Process {rank} Debug mode enabled, randomly processing {debug_size} samples.")

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
    elif 'InternVL2_5' in model_name:
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
        elif 'Kimi-VL' in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True,
            )
        elif 'InternVL2_5' in model_name:
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
        elif 'InternVL2_5' in model_name:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True,trust_remote_code=True).eval().to(device)
        elif 'MiniCPM-V' in model_name:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",trust_remote_code=True).eval().to(device)
        model = accelerator.prepare(model)

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has empty shard, skipping processing.")
        return os.path.join(output_dir, f"SPAR-Bench_results_rank_{rank}.jsonl"), 0

    prompt_template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Process {rank}", total= (total_samples + batch_size - 1) // batch_size):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]

        batch_messages_list = []
        batch_row_infos = []
        prompt_list = []
        predicted_answers_batch=[]
        if prompt_type == "default":
            max_new_token = 128
        else:
            max_new_token = 1024
        for _, row in batch_df.iterrows():
            
            question =row["question"]

            prompt_text = prompt_template["pre_prompt"].format(Question=question)
            if row['task'] in NA_QUESTION_TYPES:
                prompt_text = prompt_text+'\n'+prompt_template['na_post_prompt']
            elif row['task'] in MCA_QUESTION_TYPES:
                post_prompt = ""
                if row['task']in ['position_matching', "camera_motion_infer"]:
                    post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
                post_prompt2 = prompt_template['mca_post_prompt']
                prompt_text=prompt_text + "\n" + post_prompt + "\n" + post_prompt2
            elif row['task'] in SPECIAL_QUESTION_TYPES:
                post_prompt1 = ""
                post_prompt2 = ""
                prompt_text=prompt_text+'\n'+prompt_template['special_post_prompt']
            else:
                raise ValueError(f"Unknown question type: {row['task']}")
            frames=[]
            for i in range(len(row['image'])):
                frames.append(row['image'][i]['bytes'])
            if 'InternVL2_5' in model_name:
                num_patches_list=[]
                for i in range(len(row['image'])):
                    cur_pixel=load_image(row['image'][i]['bytes'], max_num=12).to(torch.bfloat16).cuda()
                    if i>=1:
                        pixel_values = torch.cat((pre_pixel, cur_pixel), dim=0)
                    else:
                        pixel_values=cur_pixel
                    pre_pixel=cur_pixel
                    num_patches_list.append(cur_pixel.size(0))
                video_prefix = ''.join([f'<image>\n' for i in range(len(num_patches_list))])
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
            prompt_list.append(prompt_text)
            
            content=[]
            for frame in frames:
                encoded_frame=base64.b64encode(frame).decode("utf-8")
                content.append({
                            "type": "image",
                            "image": f"data:image;base64,{encoded_frame}",
                        })
            content.append({"type": "text", "text": prompt_text})
            messages = [
                {
                    "role": "user",
                    "content": content,
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
                generated_ids_batch = model.generate(**inputs_batch, use_cache=True, max_new_tokens=max_new_token, temperature=0.01)
                generated_ids_trimmed_batch = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
                ]
                predicted_answers_batch = processor.batch_decode(
                    generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            except Exception as e:
                logger.error(f"Process {rank} batch inference failed: {e}")
                predicted_answers_batch = [""] * len(batch_messages_list)

        for i, predicted_answer in enumerate(predicted_answers_batch):
            row = batch_row_infos[i]
            ground_truth = row['answer']
            question_type = row['task']
            prompt_text = prompt_list[i]
            image_type=row['img_type']
            results.append({
                'id': row['id'],
                'question': row['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'task': question_type,
                'image_type':image_type,
                'promt': prompt_text
            })

    process_output_file = os.path.join(output_dir, f"SPAR-Bench_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} results saved to: {process_output_file}, process time: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process


def sparbench_eval(jsonl_file_path,mode="thinking"):
    results = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if mode=="thinking" and "<answer>" in doc["predicted_answer"]:
                doc["predicted_answer"]=extract_answer_text(doc["predicted_answer"])
            if doc["predicted_answer"] is None:
                doc["predicted_answer"] = "None"
            doc=sparbench_process_results(doc)
            results.append(doc)
    aggregated_results = sparbench_aggregate_results(results) 
    return aggregated_results 
