from util import *
from loguru import logger as eval_logger
import time
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import copy
import random
from .vsi_util import *
from .internvl_video_utils import load_video_internvl 

def vsibench_aggregate_results(results):
    results_df = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results_df.groupby('question_type').groups.items():
        per_question_type = results_df.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    try:
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy'),
            output.pop('object_rel_direction_medium_accuracy'),
            output.pop('object_rel_direction_hard_accuracy'),
        ]) / 3.
    except:
        output['object_rel_direction_accuracy'] =0
    output['overall_accuracy'] = (results_df["MRA:.5:.95:.05"].sum()+results_df["accuracy"].sum())/len(results_df)
    eval_logger.info(f"Evaluation results: {output}")
    return output 

def vsibench_eval(jsonl_file_path,mode="thinking"):
    results = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if (mode=="thinking" or mode=='thinking_map') and "<answer>" in doc["predicted_answer"] :
                doc["predicted_answer"]=extract_answer_text(doc["predicted_answer"])
            processed_doc = vsibench_process_results(doc)  # Process each doc to add metrics
            results.append(processed_doc)

    aggregated_results = vsibench_aggregate_results(results)  # Aggregate results after processing
    return aggregated_results

def evaluate_vsibench(rank, world_size, parquet_file, video_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, fps=1, target_resolution=(256, 256), debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="default"):
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    df = pd.read_parquet(parquet_file)
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
        return os.path.join(output_dir, f"VSI-Bench_results_rank_{rank}.jsonl"), 0

    prompt_template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Process {rank}", total=(total_samples + batch_size - 1) // batch_size):
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
            video_path = os.path.join(video_dir, row['dataset'], f"{row['scene_name']}.mp4")
            if not os.path.exists(video_path):
                print("Warning: video not found at: ", video_path)
                continue
            
            frames, timestamps, duration = load_video_frames(video_path, num_frames, fps, target_resolution)  # Get frames, timestamps, duration
            if frames is None:
                print("Warning: failed to extract frames for: ", video_path)
                continue
            
            question = row['question']
            options = row.get('options')
            if options is not None and len(options) > 0:
                options = options.tolist()                    
                question += "\nOptions:\n" + "\n".join(options)
            if prompt_type == 'thinking':
                prompt_text = prompt_template["pre_prompt"].format(Question=question, object_list=OBJECT_LIST, map_example=EXAMPLE_MAP)
                if row['question_type'] in MCA_QUESTION_TYPES:
                    prompt_text += "\n" + prompt_template["mca_post_prompt"]
                elif row['question_type'] in NA_QUESTION_TYPES:
                    prompt_text += "\n" + prompt_template["na_post_prompt"]
            else:
                prompt_text = prompt_template["pre_prompt"].format(Question=question)
                if row['question_type'] in MCA_QUESTION_TYPES:
                    prompt_text += "\n" + prompt_template["mca_post_prompt"]
                elif row['question_type'] in NA_QUESTION_TYPES:
                    prompt_text += "\n" + prompt_template["na_post_prompt"]
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
            # Batch inference 
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
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
                ]
                predicted_answers_batch = processor.batch_decode(
                    generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            except Exception as e:
                logger.error(f"Process {rank} batch inference failure: {e}")
                predicted_answers_batch = [""] * len(batch_messages_list)
        
        # Save results
        for i, predicted_answer in enumerate(predicted_answers_batch):
            row = batch_row_infos[i]
            ground_truth = row['ground_truth']
            question_type = row['question_type']
            prompt_text = prompt_list[i]
   
            results.append({
                'id': row['id'],
                'dataset': row['dataset'],
                'scene_name': row['scene_name'],
                'question': row['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'question_type': question_type,
                'promt': prompt_text,
                'frame_timestamps': timestamps, 
                'video_duration': duration
            })

    # Write results file
    process_output_file = os.path.join(output_dir, f"VSI-Bench_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} results saved to: {process_output_file}, time usage: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process
