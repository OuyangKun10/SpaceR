import json
import re
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
import numpy as np
import textwrap
from PIL import Image
from datetime import datetime, timedelta
import logging
import os
import torch
from decord import VideoReader, cpu
import pandas as pd

def extract_answer_text(text_with_tags):
    match = re.search(r"<answer>(.*?)</answer>", text_with_tags, re.DOTALL)
    if match:
        return match.group(1).strip()  
    else:
        return "None"

def format_time(elapsed_seconds):
    time_delta = timedelta(seconds=int(elapsed_seconds))
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    return f"{hours:02}h{minutes:02}m{seconds:02}s"

def setup_logger(rank, log_file, params_dict):
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
    if isinstance(gpu_ids, str):
        gpu_ids_list = gpu_ids.split(',')
    else:
        gpu_ids_list = [str(gpu_id) for gpu_id in gpu_ids]
    num_gpus_available = len(gpu_ids_list)
    if world_size == 1 and num_gpus_available > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_list)
        selected_gpu = ",".join(gpu_ids_list)
    elif world_size > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_list)
        if rank < num_gpus_available:
            selected_gpu = gpu_ids_list[rank]
            torch.cuda.set_device(int(selected_gpu))
        else:
            selected_gpu = gpu_ids_list[rank % num_gpus_available]
            torch.cuda.set_device(int(selected_gpu))
            logger = logging.getLogger(__name__)
            logger.warning(f"Rank {rank}: Not enough GPUs, reusing GPU: {selected_gpu}. Reduce number of processes or increase GPUs.")
    else: 
        selected_gpu = gpu_ids_list[rank % num_gpus_available] if gpu_ids_list else "0"  # Default to GPU 0

    logger = logging.getLogger(__name__)
    logger.info(f"Rank {rank}: Selected GPU: {selected_gpu}, CUDA_VISIBLE_DEVICES: {gpu_ids}")
    return selected_gpu

def read_data(file_path):
    file_extension = file_path.lower().split('.')[-1]

    try:
        if file_extension == 'csv':
            return pd.read_csv(file_path)
        elif file_extension == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_extension == 'jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        elif file_extension == 'pkl' or file_extension == 'pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_extension == 'parquet':
            return pd.read_parquet(file_path)
        else:
            print(f"Error: Unsupported file format: {file_extension}")
            return None

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding error in file: {file_path}. {e}")
        return None
    except pickle.UnpicklingError as e:
        print(f"Error: Pickle unpickling error in file: {file_path}. {e}")
        return None
    except ImportError as e: 
        print(f"Error: Import error while reading {file_extension} file. Please ensure necessary libraries are installed. {e}")
        return None
    except Exception as e: 
        print(f"Error: An unexpected error occurred while reading file: {file_path}. {e}")
        return None

def load_cog_map(data, id_key, cog_key):
    cog_maps = {}

    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        if not id_key or not cog_key:
            print("ERROR: id_key and cog_key must be provided for list of dictionary data.")
            return None

        for item in data:
            item_id = item.get(id_key) 
            cog_map_str = item.get(cog_key)

            if item_id is None or cog_map_str is None:
                print(f"Warning: Missing '{id_key}' or '{cog_key}' in dictionary item: {item}. Skipping.")
                continue  

            try:
                item_id = int(item_id)  
            except ValueError:
                print(f"Warning: Invalid ID format '{item_id}' for item: {item}. Skipping.")
                continue
            print(cog_map_str)
            print(type(print(cog_map_str)))
            if not isinstance(cog_map_str, str):
                if isinstance(cog_map_str, list):
                    cog_map_str = cog_map_str[0]
                else:
                    cog_map_str=str(cog_map_str)
            cog_map = extract_json_from_string(cog_map_str)
            if cog_map is not None:
                cog_maps[item_id] = cog_map
            else:
                print(f"Warning: Failed to extract cog_map from '{cog_map_str}' for ID: {item_id}. Skipping.")

    elif isinstance(data, dict): 
        if cog_key in data.keys() and id_key in data.keys(): 
            print("Warning: Assuming input data is a dictionary of dictionaries based on keys check. Please verify data format.")  # Inform user of potential misunderstanding
            for item_key in data: 
                item = data[item_key] 
                if isinstance(item, dict):  
                    item_cog_map_str = item.get(cog_key)  
                    item_id = item.get(id_key)  
                    if item_cog_map_str and item_id:  
                        if isinstance(cog_map_str, list):
                            cog_map_str = cog_map_str[0]
                        else:
                            cog_map_str=str(cog_map_str)
                        cog_map = extract_json_from_string(item_cog_map_str)
                        if cog_map is not None:
                            cog_maps[item_id] = cog_map
                        else:
                            print(f"Warning: Failed to extract cog_map from '{item_cog_map_str}' for ID: {item_id}, item '{item_key}'. Skipping.")
                    else:
                        print(f"Warning: Missing '{cog_key}' or '{id_key}' in dictionary item '{item_key}'. Skipping.")
                else:
                    print(f"Warning: Expected dictionary for item '{item_key}' in dictionary data, but got {type(item)}. Skipping.")

        else:
            print("ERROR: cognitive_map or id_key not found in dictionary data (or data is not in expected dictionary format).")
            return None

    else:
        print("ERROR: Unsupported data format. Input data must be a list of dictionaries or a dictionary of dictionaries.")
        return None

    if not cog_maps:  
        print("Warning: No cognitive maps loaded. Please check your input data and format.")
        return None  

    return cog_maps

def load_video_frames(video_path, num_frames=4, fps=1, target_resolution=(256, 256)):
    """Use decord to read video frames and return timestamps of those frames."""
    def resize_image(image, max_size=448):
        """Resize image maintaining aspect ratio. Max dimension does not exceed max_size."""
        h, w = image.size
        if max(h, w) <= max_size:
            return image
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        video_duration = total_frames / vr.get_avg_fps() if vr.get_avg_fps() > 0 else total_frames / 30  # Estimate duration
        video_duration = int(video_duration)
        if fps > 0:
            target_frames = min(num_frames, int(video_duration * fps))
            target_frames = max(1, target_frames) 
        else:
            target_frames = num_frames

        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames_np = vr.get_batch(frame_indices).asnumpy()
        t1, t2 = target_resolution
        frames_pil = [resize_image(Image.fromarray(f), max(t1, t2)) for f in frames_np]
        timestamps = [int(idx / vr.get_avg_fps()) for idx in frame_indices] if vr.get_avg_fps() > 0 else [int(idx / 30) for idx in frame_indices]  # Get integer timestamps
        return frames_pil, timestamps, video_duration
    except Exception as e:
        return None, None, None 
