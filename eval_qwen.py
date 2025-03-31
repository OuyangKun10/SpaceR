import os
import json
import torch
import pandas as pd
import numpy as np
import logging
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from multiprocessing import set_start_method
import time
from datetime import datetime, timedelta
import random
from util import extract_object_coordinates,load_cog_map,read_data,cog_img
# 导入评测函数
from vsibench_acc import evaluate_jsonl_results, MCA_QUESTION_TYPES, NA_QUESTION_TYPES
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = r"/data/ouyangkun/.cache/huggingface"
# os.environ["no_proxy"]="hf-mirror.com"
PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a single word or phrase."
    },
    "thinking":
    {
        "pre_prompt": "",
        "mca_post_prompt": "First output the thinking process in <think> </think> tags and then output an option letter in <answer> </answer> tags.",#"Please output the thinking process in <think>...</think> and final answer that is an option letter in <answer>...</answer>.\nThe output answer format should be as follows: <think>...</think><answer>...</answer>. You must output an option letter as the final answer in <answer>...</answer>.",
        "na_post_prompt": "First output the thinking process in <think> </think> tags and then output a number in <answer> </answer> tags."#"Please output the thinking process in <think>...</think> and final answer that is a single word or phrase in <answer>...</answer>.\nThe output answer format should be as follows: <think>...</think><answer>...</answer>. You must output a single word or phrase as the final answer in <answer>...</answer>."
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
PROMPT_COGMAP_TEMPLATE = """[Task]
These frames represent an indoor scene from a video. Your objective
is to identify specific objects within the **video scene**, un-
derstand the spatial arrangement of the scene, and
estimate the center point of each object, assuming
the entire scene is represented by a 10x10 grid. Base your analysis on the provided frames.
[Rule]
1. We provide the categories to care about in this
scene: {[
    "ceiling light", "trash can", "bed", "heater", "closet", "pillow", "backpack", "chair", "refrigerator",
    "tv", "nightstand", "keyboard", "computer tower", "coat hanger", "table", "trash bin", "whiteboard",
    "monitor", "sofa", "clock", "computer mouse", "radiator", "telephone"
]}. Focus ONLY on
these categories for the entire video scene.
2. Estimate the center location of each instance
within the provided categories, assuming the entire
scene is represented by a 10x10 grid, considering the information from all frames.
3. If a category contains multiple instances across all frames, include
all of them.
4. Each object’s estimated location should ac-
curately reflect its real position in the scene,
preserving the relative spatial relationships among
all objects in the video scene.
[Output]
Present the estimated center locations for each ob-
ject as a list within a dictionary. STRICTLY follow
this JSON format: {"category name": [(x_1, y_1),
...], ...} for the **entire video scene**. Notably, x,y should be in the range of [1, 10].
"""
def format_time(elapsed_seconds):
    """将秒数格式化为 XXhXXmXXs 格式"""
    time_delta = timedelta(seconds=int(elapsed_seconds))
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    return f"{hours:02}h{minutes:02}m{seconds:02}s"

def setup_logger(rank, log_file, params_dict):
    """为不同进程创建日志文件, 包含参数和时间戳"""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_with_timestamp = log_file.replace(".log", f"_{timestamp_str}_rank_{rank}.log")
    logging.basicConfig(
        filename=log_file_with_timestamp,
        level=logging.INFO,
        format=f'%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting process with rank {rank}")
    logger.info("Running parameters:")
    for key, value in params_dict.items():
        logger.info(f"  {key}: {value}")
    return logger

def allocate_gpu(rank, gpu_ids, world_size):
    """为每个进程分配 GPU"""
    if isinstance(gpu_ids, str):
        gpu_ids_list = gpu_ids.split(',')
    else:
        gpu_ids_list = [str(gpu_id) for gpu_id in gpu_ids]
    num_gpus_available = len(gpu_ids_list)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_list)
    if world_size == 1 and num_gpus_available > 1:
        # 单进程多卡
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_list)
        selected_gpu = ",".join(gpu_ids_list)
    elif world_size > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_list)
        if rank < num_gpus_available:
            # 多进程多卡，每个进程一张卡
            selected_gpu = gpu_ids_list[rank]
            torch.cuda.set_device(int(selected_gpu))
        else:
            # GPU 不够时，循环使用可用的 GPU
            selected_gpu = gpu_ids_list[rank % num_gpus_available]
            torch.cuda.set_device(int(selected_gpu))
            logger = logging.getLogger(__name__)
            logger.warning(f"Rank {rank}: GPU资源不足，重用GPU: {selected_gpu}，请减少进程数或增加GPU")
    else: # world_size == 1 and num_gpus_available <= 1
        selected_gpu = gpu_ids_list[rank % num_gpus_available] if gpu_ids_list else "0" # 默认 GPU 0

    logger = logging.getLogger(__name__)
    logger.info(f"Rank {rank}: 选择了GPU: {selected_gpu}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    return selected_gpu

def load_video_frames(video_path, num_frames=4, fps=1, target_resolution=(256, 256)):
    """使用 decord 读取视频帧, 并返回帧的时间戳"""
    # print(f"读取{video_path}")
    def resize_image(image, max_size=448):
        """调整图片大小，保持宽高比，最长边不超过 max_size"""
        h, w = image.size
        if max(h, w) <= max_size:
            return image
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        # resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # return resized_image
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        video_duration = total_frames / vr.get_avg_fps() if vr.get_avg_fps() > 0 else total_frames / 30 # 获取视频时长
        video_duration=int(video_duration)
        if fps > 0:
            target_frames = min(num_frames, int(video_duration * fps))
            target_frames = max(1, target_frames) # 至少采样一帧
        else:
            target_frames = num_frames

        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames_np = vr.get_batch(frame_indices).asnumpy()
        t1,t2=target_resolution
        frames_pil = [resize_image(Image.fromarray(f),max(t1,t2)) for f in frames_np]
        timestamps = [int(idx / vr.get_avg_fps()) for idx in frame_indices] if vr.get_avg_fps() > 0 else [int(idx / 30) for idx in frame_indices] # 获取整数时间戳
        return frames_pil, timestamps, video_duration # 返回帧、时间戳和视频时长
    except Exception as e:
        return None, None, None # 错误时返回 None for frames, timestamps and duration

def generate_cognitive_map(frames_pil, processor, model, device):
    """生成认知地图"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames_pil,
                },
                {"type": "text", "text": PROMPT_COGMAP_TEMPLATE},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    try:
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)# 增加 max_new_tokens
        generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
        cognitive_map_str = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

        return cognitive_map_str
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"认知地图生成失败: {e}")
        return None


def evaluate_vsibench(rank, world_size, parquet_file, video_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, fps=1, target_resolution=(256, 256), debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="default", use_cognitive_map=True, offload_cogmap=False, cogmap_file_path=None, cogmap_id_key="id", cogmap_cog_key="cog_map", cogmap_data_format="list_dict"):
    """评测 VSIBench 数据集"""
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} 使用设备: {device}")

    # 读取数据
    df = pd.read_parquet(parquet_file)
    if debug:
        df = df.sample(n=debug_size)
        logger.info(f"进程 {rank} Debug 模式开启，随机处理 {debug_size} 条数据。")

    #fixme

    # df=df[df['question_type'].isin(MCA_QUESTION_TYPES)]
    # allowed_types = ['object_size_estimation', 'room_size_estimation']
    # df = df[df['question_type'].isin(allowed_types)]
    # df = df[df['question_type'] == 'object_size_estimation']
    # 数据分片
    if world_size > 1:
        df_shard = np.array_split(df, world_size)[rank]
    else:
        df_shard = df
    logger.info(f"Rank {rank} Shard size: {len(df_shard)}")

    # processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    # 加载模型
    # model_grpo="/mnt/moonfs/ouyangkun-m2/sptial/video-rft/log_dvd/checkpoint-200"
     # default processer
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = 'left'

    
    if world_size == 1 and len(gpu_ids.split(',')) > 1:
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto",
        #     attn_implementation="flash_attention_2",device_map="auto")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
       
        # model = Qwen2VLForConditionalGeneration.from_pretrained(model_grpo, torch_dtype="auto",
        #     attn_implementation="flash_attention_2",device_map="auto")
        model = accelerator.prepare(model)
        model.eval()
    else:
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", attn_implementation="flash_attention_2").eval().to(device)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval().to(device)
        model = accelerator.prepare(model)

    results = []
    total_samples = len(df_shard)
    # print("total_samples:",total_samples)
    if total_samples == 0:
        logger.info(f"Rank {rank} has empty shard, skipping processing.")
        return os.path.join(output_dir, f"vsibench_results_rank_{rank}.jsonl"), 0

    prompt_template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    cog_maps_dict_from_file = {} # 用于存储从文件加载的 cog_maps
    if use_cognitive_map and offload_cogmap: # 在循环外加载一次即可，假设 cogmap 文件包含所有视频的 cogmap
        if cogmap_file_path:
            logger.info(f"Rank {rank}: Attempting to load cogmaps from file: {cogmap_file_path}")
            loaded_cog_map_data = read_data(cogmap_file_path)
            if loaded_cog_map_data is not None:
                cog_maps_dict_from_file = load_cog_map(loaded_cog_map_data, cogmap_id_key, cogmap_cog_key)
                if cog_maps_dict_from_file:
                    logger.info(f"Rank {rank}: Successfully loaded {len(cog_maps_dict_from_file)} cogmaps from file.")
                else:
                    logger.warning(f"Rank {rank}: load_cog_map returned empty or None. Cogmap offloading failed. Falling back to generation.")
                    offload_cogmap = False # 加载失败，回退到生成
            else:
                logger.error(f"Rank {rank}: Failed to read cogmap data from {cogmap_file_path}. Falling back to cogmap generation.")
                offload_cogmap = False # 读取文件失败，回退到生成
        else:
            logger.warning(f"Rank {rank}: offload_cogmap is True but cogmap_file_path is not provided. Falling back to cogmap generation.")
            offload_cogmap = False # 未提供文件路径，回退到生成

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"进程 {rank}", total= (total_samples + batch_size - 1) // batch_size):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]

        batch_messages_list = []
        batch_row_infos = []
        prompt_list = []
        for _, row in batch_df.iterrows():
            video_path = os.path.join(video_dir, row['dataset'], f"{row['scene_name']}.mp4")
            # print("video_path:",video_path)
            if not os.path.exists(video_path):
                continue

            frames, timestamps, duration = load_video_frames(video_path, num_frames, fps, target_resolution) # 获取 frames, timestamps, duration
            # print("frames:",frames)
            if frames is None:
                continue

            cognitive_map = None # 初始化 cognitive_map
            cogmap_source = "generated" # 标记 cogmap 来源

            if use_cognitive_map:
                if offload_cogmap and row['scene_name'] in cog_maps_dict_from_file: # 尝试从文件加载
                    cognitive_map = cog_maps_dict_from_file.get(row['scene_name']) # 假设 scene_name 可以作为 cogmap 的 id
                    cogmap_source = "offloaded"
                    logger.info(f"Rank {rank}: Loaded cogmap from file for scene: {row['scene_name']}")
                else: # 生成 cognitive_map
                    # print("生成 cognitive_map")
                    cognitive_map_str_list = generate_cognitive_map(frames, processor, model, device) # generate_cognitive_map 返回的是列表
                    cognitive_map_str = cognitive_map_str_list[0] if cognitive_map_str_list else None # 取第一个元素
                    if cognitive_map_str:
                        cognitive_map = extract_object_coordinates(cognitive_map_str) # 解析 JSON
                    if cognitive_map:
                         cogmap_source = "generated"
                         logger.info(f"Rank {rank}: Generated cogmap for scene: {row['scene_name']}")
                    else:
                        logger.warning(f"Rank {rank}: Failed to generate cogmap for scene: {row['scene_name']}")

            # print(cognitive_map)
            cognitive_map_text = f"\nCognitive Map:\n{cognitive_map}" if cognitive_map else "\nCognitive Map generation failed or not offloaded."
            if use_cognitive_map and offload_cogmap and cognitive_map:
                cognitive_map_text = f"\nCognitive Map:\n{cognitive_map}" if cognitive_map else "\nCognitive Map offloading failed."
            elif use_cognitive_map and not offload_cogmap and cognitive_map:
                 cognitive_map_text = f"\nCognitive Map:\n {cognitive_map}" if cognitive_map else "\nCognitive Map generation failed."
            elif use_cognitive_map and not cognitive_map:
                cognitive_map_text = "\nCognitive Map Unavailable (Generation/Offloading Failed)."
            else:
                cognitive_map_text = "" # use_cognitive_map=False 时不显示

            # 构建 Prompt
            question = row['question']
            options = row.get('options')
            prompt_text = prompt_template["pre_prompt"]
            if use_cognitive_map: # 只有当 use_cognitive_map 为 True 时才加入 cogmap
                prompt_text += "\nHere is a cognitive_map for the video that may help you understand the space in the video:\n" +cognitive_map_text+"\n"
            prompt_text += "Question:\n"+question# 将认知地图加入 prompt

            if options is not None and len(options) > 0:
                options = options.tolist()
                # print("options:",options)
                
                prompt_text += "\nOptions:\n" + "\n".join(options)
                # print("prompt_text:",prompt_text)
                
            if row['question_type'] in MCA_QUESTION_TYPES:
                prompt_text += "\n" + prompt_template["mca_post_prompt"]
            elif row['question_type'] in NA_QUESTION_TYPES:
                prompt_text += "\n" + prompt_template["na_post_prompt"]
            # prompt_text += "\nThe answer is:"
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

        if not batch_messages_list:
            continue

        # 批量推理
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
        # print("开始推理")
        try:
            generated_ids_batch = model.generate(**inputs_batch, use_cache=True, max_new_tokens=512, do_sample=False)
            generated_ids_trimmed_batch = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
            ]
            predicted_answers_batch = processor.batch_decode(
                generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception as e:
            logger.error(f"进程 {rank} batch 推理失败: {e}")
            predicted_answers_batch = [""] * len(batch_messages_list)

        # 保存结果
        for i, predicted_answer in enumerate(predicted_answers_batch):
            # print("prompt:",prompt_text)
            print("predicted_answer:",predicted_answer)
            print("ground truth:",row['ground_truth'])
            row = batch_row_infos[i]
            ground_truth = row['ground_truth']
            question_type = row['question_type']
            prompt_text = prompt_list[i]
            cognitive_map_to_save = cognitive_map if cognitive_map else "Cognitive Map Generation/Offloading Failed or not used" # 保存 cognitive_map

            results.append({
                'id': row['id'],
                'dataset': row['dataset'],
                'scene_name': row['scene_name'],
                'question': row['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'question_type': question_type,
                'promt': prompt_text,
                'frame_timestamps': timestamps, # 保存时间戳
                'video_duration': duration, # 保存视频时长
                'cognitive_map': cognitive_map_to_save, # 保存 cognitive_map
                'cogmap_source': cogmap_source # 保存 cogmap 来源
            })

    # 写入结果文件
    process_output_file = os.path.join(output_dir, f"vsibench_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} 结果已保存到: {process_output_file}, 进程运行时间: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process

def merge_results(world_size, output_file):
    """合并多个进程的结果文件"""
    with open(output_file, 'w') as outfile:
        for rank in range(world_size):
            process_file = os.path.join(os.path.dirname(output_file), f"vsibench_results_rank_{rank}.jsonl")
            if os.path.exists(process_file):
                with open(process_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(process_file)

if __name__ == "__main__":
    set_start_method('spawn')
    main_start_time = time.time()

    parquet_file = "/mnt/moonfs/ouyangkun-m2/dataset/VSI_bench/test-00000-of-00001.parquet"
    video_dir = "/mnt/moonfs/ouyangkun-m2/dataset/VSI_bench"
    output_dir_base = "/home/ouyangkun/code/sptial/res/Qwen2.5-VL-7B-GRPO-SPA-Size3k"
    model_name = "/mnt/moonfs/ouyangkun-m2/code/spatial_reasoning/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-SPA-Size3k/checkpoint-1500"
    # model_name ="/mnt/moonfs/ouyangkun-m2/model/Qwen2.5-VL-7B-Instruct"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_base, model_name.split("/")[-1] ,timestamp_str)
    os.makedirs(output_dir, exist_ok=True)

    output_jsonl_file = os.path.join(output_dir, "vsibench_results_GRPO.jsonl")
    log_output_file = os.path.join(output_dir, "vsibench_eval_GRPO.log")

    SELECTED_PROMPT_TYPE = "thinking" # 可修改为 "gemini_api" 或 "gpt4v" “thinking” default
    gpu_ids = "0,1,2,3"
    num_processes = 4
    num_frames = 32
    fps=1
    target_resolution=(448, 448)
    debug_mode = False
    use_cognitive_map=False
    offload_cogmap=False #  是否从文件加载 cogmap
    cogmap_file_path = "/data2/ouyangkun/code/spatial_reasoning/cog_maps/vsibench_results_with_cogmap.jsonl" 
    cogmap_id_key = "id" # 新增参数：cogmap 文件中 id 的键名，根据你的文件格式修改
    cogmap_cog_key = "cognitive_map" # 新增参数：cogmap 文件中 cogmap 数据的键名，根据你的文件格式修改
    cogmap_data_format = "jsonl" # 新增参数: cogmap 文件的数据格式，根据你的文件格式修改： "json", "jsonl", "csv", "pkl", "parquet"
    batch_size = 1
    debug_size = 64

    params_to_log = {
        "model_name": model_name,
        "num_frames": num_frames,
        "fps": fps,
        "target_resolution": target_resolution,
        "debug_mode": debug_mode,
        "batch_size": batch_size,
        "debug_size": debug_size,
        "gpu_ids": gpu_ids,
        "num_processes": num_processes,
        "prompt_type": SELECTED_PROMPT_TYPE,
        "use_cognitive_map": use_cognitive_map,
        "offload_cogmap": offload_cogmap, 
        "cogmap_file_path": cogmap_file_path, 
        "cogmap_id_key": cogmap_id_key, 
        "cogmap_cog_key": cogmap_cog_key, 
        "cogmap_data_format": cogmap_data_format 
    }

    main_logger = setup_logger(0, log_output_file, params_to_log) #  主进程的logger

    process_runtimes = []
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(evaluate_vsibench, [
                (rank, num_processes, parquet_file, video_dir, model_name, output_dir, log_output_file, gpu_ids, num_frames, fps, target_resolution, debug_mode, batch_size, debug_size, params_to_log, SELECTED_PROMPT_TYPE, use_cognitive_map, offload_cogmap, cogmap_file_path, cogmap_id_key, cogmap_cog_key, cogmap_data_format)
                for rank in range(num_processes)
            ])
            process_runtimes = [res[1] for res in results]
            process_files = [res[0] for res in results]
        merge_results(num_processes, output_jsonl_file)
    else:
        # 单进程
        process_output_file, elapsed_time_process = evaluate_vsibench(0, num_processes, parquet_file, video_dir, model_name, output_dir, log_output_file, gpu_ids, num_frames, fps, target_resolution, debug_mode, batch_size, debug_size, params_to_log, SELECTED_PROMPT_TYPE, use_cognitive_map, offload_cogmap, cogmap_file_path, cogmap_id_key, cogmap_cog_key, cogmap_data_format)
        process_runtimes = [elapsed_time_process]
        os.rename(process_output_file, output_jsonl_file)

    main_end_time = time.time()
    main_elapsed_time = main_end_time - main_start_time
    max_process_runtime = max(process_runtimes) if process_runtimes else 0

    main_logger.info(f"Maximum process runtime: {format_time(max_process_runtime)}")

    print(f"所有评测完成，最终结果保存在: {output_jsonl_file}")
    print(f"程序总运行时间: {format_time(main_elapsed_time)}")

    # 评测结果并输出准确率
    evaluation_results = evaluate_jsonl_results(output_jsonl_file,SELECTED_PROMPT_TYPE)
    main_logger.info(f"Evaluation Metrics: {evaluation_results}") # 将评测结果记录到主程序日志
    print(evaluation_results)
    # print(f"总体准确率: {evaluation_results.get('overall_accuracy', 0.0) * 100.:.2f}%")
    log_str = f"VSIBench 评测完成，结果文件: {output_jsonl_file}\n"
    log_str += f"总体准确率: {evaluation_results.get('overall_accuracy', 0.0) * 100.:.2f}%\n"
    # 打印不同 question type 的准确率
    for question_type in MCA_QUESTION_TYPES + NA_QUESTION_TYPES:
        accuracy_key = f"{question_type}_accuracy"
        if accuracy_key in evaluation_results:
            # print(f"{question_type} 准确率: {evaluation_results[accuracy_key] * 100.:.2f}%")
            log_str += f"{question_type} 准确率: {evaluation_results[accuracy_key] * 100.:.2f}%\n"
    if 'object_rel_direction_accuracy' in evaluation_results:
        # print(f"object_rel_direction 总体准确率: {evaluation_results['object_rel_direction_accuracy'] * 100.:.2f}%")
        log_str += f"object_rel_direction 总体准确率: {evaluation_results['object_rel_direction_accuracy'] * 100.:.2f}%\n"
    main_logger.info(log_str) # 完整评测结果信息记录到日志文件
    print(log_str) # 打印结果
  