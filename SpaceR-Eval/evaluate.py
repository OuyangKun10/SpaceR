import os
import json
import torch
import pandas as pd
import numpy as np
import logging
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import set_start_method
import time
from datetime import datetime, timedelta
import random
from util import setup_logger, format_time
from data_utils.vsibench import evaluate_vsibench,vsibench_eval
from data_utils.stibench import evaluate_stibench,stibench_eval
from data_utils.sparbench import evaluate_sparbench,sparbench_eval
from data_utils.videomme import evaluate_videomme,videomme_eval
from data_utils.longvideobench import evaluate_longvideobench,longvideobench_eval
from data_utils.tempcompass import evaluate_tempcompass,tempcompass_eval


def merge_results(world_size: int, output_file: str, task: str):
    """
    Merges result files generated by multiple processes into a single output file.

    Args:
        world_size (int): The total number of processes that generated result files.
        output_file (str): The path to the final merged output file.
        task (str): The name of the evaluation task, used to determine the naming
                    pattern of individual process result files.
    """
    with open(output_file, 'w') as outfile:
        for rank in range(world_size):
            # Construct the path to the result file for each process
            process_file = os.path.join(os.path.dirname(output_file), f"{task}_results_rank_{rank}.jsonl")
            if os.path.exists(process_file):
                with open(process_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
            else:
                print(f"Warning: Process file {process_file} not found for merging.") # Optional: Add a warning if a file is missing

def prepare_data(task: str) -> tuple[str | list[str], str]:
    """
    Prepares and returns the data file path(s) and video directory for a given task.

    Args:
        task (str): The name of the evaluation task.

    Returns:
        tuple[str | list[str], str]: A tuple containing:
            - The path to the data file (or a list of paths for SPAR-Bench).
            - The path to the directory containing video files.
    """
    if task =='VSI-Bench':
        return "VSI_bench/test-00000-of-00001.parquet", "VSI_bench"
    elif task =="STI-Bench":
        return "STI-Bench/qa.parquet","STI-Bench/video"
    elif task =='SPAR-Bench':
         return ['SPAR-Bench/data/test-00000-of-00004.parquet','SPAR-Bench/data/test-00001-of-00004.parquet','SPAR-Bench/data/test-00002-of-00004.parquet','SPAR-Bench/data/test-00003-of-00004.parquet'],'SPAR-7M/spar/structured3d/images'
    elif task=='Video-MME':
        return 'Video-MME/videomme/test-00000-of-00001.parquet','Video-MME/data'
    elif task=='LongVideoBench':
        return 'LongVideoBench/lvb_val.json', 'LongVideoBench/videos'
    elif task=='TempCompass':
        return "TempCompass/eval_tempcompass.json", "TempCompass/videos"
    else:
        raise ValueError(f"Task {task} not recognized for data preparation.")

# List of supported evaluation benchmark tasks
SUPPORTED_TASK=['VSI-Bench',"STI-Bench",'SPAR-Bench','Video-MME','LongVideoBench','TempCompass']

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'.
    set_start_method('spawn', force=True)

    main_start_time = time.time() # Record the start time of the main script

    # --- Configuration ---
    EVAL_TASK='Video-MME' # Specify the evaluation task to run
    if EVAL_TASK not in SUPPORTED_TASK:
        print(f"Error: Task '{EVAL_TASK}' is not supported. Supported tasks are: {SUPPORTED_TASK}")
        exit()

    data_file, video_dir = prepare_data(EVAL_TASK)

    # Base directory for storing evaluation results
    output_dir_base = "/res/Qwen2.5-VL-7B-Instruct"
    # Path of the model being evaluated
    model_name="Qwen/Qwen2.5-VL-7B-Instruct"
   
    # Create a timestamped directory for this specific run's outputs
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_base, EVAL_TASK, model_name.split("/")[-1], timestamp_str)
    os.makedirs(output_dir, exist_ok=True)

    # Path for the final merged JSONL results file
    output_jsonl_file = os.path.join(output_dir, f"{EVAL_TASK}_results.jsonl")
    # Path for the main evaluation log file
    log_output_file = os.path.join(output_dir, f"{EVAL_TASK}_eval.log")

    # Type of prompt to use for the model (e.g., "thinking", "default")
    SELECTED_PROMPT_TYPE = "thinking"

    # GPU and multiprocessing configuration
    gpu_ids = "0,1,2,3" # Comma-separated string of GPU IDs to use
    num_processes = 4   # Number of parallel processes to run

    # Video processing parameters
    num_frames = 32             # Number of frames to sample from each video
    fps = 1                     # Frames per second for sampling
    target_resolution = (448, 448) # Target resolution for video frames (width, height)

    # Debugging parameters
    debug_mode = True   # If True, run in debug mode (e.g., process a subset of data)
    debug_size = 4      # Number of samples to process if debug_mode is True
    batch_size = 1      # Batch size for model inference

    # Collect parameters to log
    params_to_log = {
        "model_name": model_name,
        "eval_task": EVAL_TASK,
        "data_file": data_file,
        "video_dir": video_dir,
        "num_frames": num_frames,
        "fps": fps,
        "target_resolution": target_resolution,
        "debug_mode": debug_mode,
        "batch_size": batch_size,
        "debug_size": debug_size if debug_mode else "N/A",
        "gpu_ids": gpu_ids,
        "num_processes": num_processes,
        "prompt_type": SELECTED_PROMPT_TYPE,
        "output_dir": output_dir,
    }

    # Setup main logger for the script
    main_logger = setup_logger(0, log_output_file, params_to_log)
    main_logger.info("Main script started. Configuration logged.")

    process_runtimes = [] # To store runtime of each process

    # --- Run Evaluation ---
    # Dispatch to task-specific evaluation functions, either in parallel or sequentially
    if num_processes > 1:
        main_logger.info(f"Starting evaluation with {num_processes} processes.")
        with mp.Pool(processes=num_processes) as pool:
            args_list = [
                (rank, num_processes, data_file, video_dir, model_name, output_dir, log_output_file,
                 gpu_ids, num_frames, fps, target_resolution, debug_mode, batch_size,
                 debug_size, params_to_log, SELECTED_PROMPT_TYPE)
                for rank in range(num_processes)
            ]
            if EVAL_TASK == 'VSI-Bench':
                results = pool.starmap(evaluate_vsibench, args_list)
            elif EVAL_TASK == 'STI-Bench':
                results = pool.starmap(evaluate_stibench, args_list)
            elif EVAL_TASK == 'SPAR-Bench':
                results = pool.starmap(evaluate_sparbench, args_list)
            elif EVAL_TASK == 'Video-MME':
                results = pool.starmap(evaluate_videomme, args_list)
            elif EVAL_TASK == 'LongVideoBench':
                results = pool.starmap(evaluate_longvideobench, args_list)
            elif EVAL_TASK == 'TempCompass':
                results = pool.starmap(evaluate_tempcompass, args_list)
            else:
                main_logger.error(f"Task '{EVAL_TASK}' not recognized for multiprocessing dispatch.")
                exit()

          
            process_runtimes = [res[1] for res in results if isinstance(res, tuple) and len(res) == 2]
        # Merge the .jsonl files produced by each process
        merge_results(num_processes, output_jsonl_file, EVAL_TASK)
        main_logger.info(f"Results from {num_processes} processes merged into {output_jsonl_file}")
    else:
        # Single process execution
        main_logger.info("Starting evaluation with a single process.")
        common_args = (0, 1, data_file, video_dir, model_name, output_dir, log_output_file,
                       gpu_ids, num_frames, fps, target_resolution, debug_mode, batch_size,
                       debug_size, params_to_log, SELECTED_PROMPT_TYPE)
        process_output_file, elapsed_time_process = None, 0

        if EVAL_TASK == 'VSI-Bench':
            process_output_file, elapsed_time_process = evaluate_vsibench(*common_args)
        elif EVAL_TASK == 'STI-Bench':
            process_output_file, elapsed_time_process = evaluate_stibench(*common_args)
        elif EVAL_TASK == 'SPAR-Bench':
            process_output_file, elapsed_time_process = evaluate_sparbench(*common_args)
        elif EVAL_TASK == 'Video-MME':
            process_output_file, elapsed_time_process = evaluate_videomme(*common_args)
        elif EVAL_TASK == 'LongVideoBench':
            process_output_file, elapsed_time_process = evaluate_longvideobench(*common_args)
        elif EVAL_TASK == 'TempCompass':
            process_output_file, elapsed_time_process = evaluate_tempcompass(*common_args)
        else:
            main_logger.error(f"Task '{EVAL_TASK}' not recognized for single process dispatch.")
            exit()

        process_runtimes = [elapsed_time_process]
        if process_output_file and os.path.exists(process_output_file):
            os.rename(process_output_file, output_jsonl_file)
            main_logger.info(f"Single process result saved to {output_jsonl_file}")
        else:
            main_logger.error(f"Single process output file {process_output_file} not found or not generated.")


    main_end_time = time.time()
    main_elapsed_time = main_end_time - main_start_time
    max_process_runtime = max(process_runtimes) if process_runtimes else 0

    main_logger.info(f"All evaluation tasks completed. Final results are in: {output_jsonl_file}")
    main_logger.info(f"Maximum individual process runtime: {format_time(max_process_runtime)}")
    main_logger.info(f"Total script runtime: {format_time(main_elapsed_time)}")

    print(f"All evaluation tasks completed. Final results are in: {output_jsonl_file}")
    print(f"Total script runtime: {format_time(main_elapsed_time)}")

    # --- Perform Final Evaluation Scoring ---
    # This section calculates metrics based on the generated results file.
    evaluation_results = {}
    log_str = "" # Initialize log string for evaluation results

    main_logger.info(f"Starting final scoring for {EVAL_TASK} using results from {output_jsonl_file}")

    if EVAL_TASK == 'VSI-Bench': 
        try:
            from util import MCA_QUESTION_TYPES, NA_QUESTION_TYPES
        except ImportError:
            MCA_QUESTION_TYPES = []
            NA_QUESTION_TYPES = []
            main_logger.warning("MCA_QUESTION_TYPES and NA_QUESTION_TYPES not found, VSI-Bench detailed metrics might be incomplete.")

        evaluation_results = vsibench_eval(output_jsonl_file, SELECTED_PROMPT_TYPE)
        print("VSI-Bench Evaluation Results:", evaluation_results)
        log_str = f"VSI-Bench Evaluation Complete. Results file: {output_jsonl_file}\n"
        log_str += f"Overall Accuracy: {evaluation_results.get('overall_accuracy', 0.0) * 100.:.2f}%\n"
        for question_type in MCA_QUESTION_TYPES + NA_QUESTION_TYPES:
            accuracy_key = f"{question_type}_accuracy"
            if accuracy_key in evaluation_results:
                log_str += f"{question_type} Accuracy: {evaluation_results[accuracy_key] * 100.:.2f}%\n"
        if 'object_rel_direction_accuracy' in evaluation_results: 
            log_str += f"Object Relative Direction Accuracy: {evaluation_results['object_rel_direction_accuracy'] * 100.:.2f}%\n"

    elif EVAL_TASK == 'STI-Bench':
        evaluation_results = stibench_eval(output_jsonl_file, SELECTED_PROMPT_TYPE)
        print("STI-Bench Evaluation Results:", evaluation_results)
        log_str = f"STI-Bench Evaluation Complete. Results file: {output_jsonl_file}\n"
        log_str += f"Overall Accuracy: {evaluation_results.get('overall_accuracy', 0.0) * 100.:.2f}%\n"

    elif EVAL_TASK == 'SPAR-Bench':
        evaluation_results = sparbench_eval(output_jsonl_file, SELECTED_PROMPT_TYPE)
        print("SPAR-Bench Evaluation Results:", evaluation_results)
        log_str = f"SPAR-Bench Evaluation Complete. Results file: {output_jsonl_file}\n"
        overall_metrics = evaluation_results.get('overall', {})
        log_str += f"Overall Accuracy: {overall_metrics.get('overall_accuracy', 0.0) * 100.:.2f}%\n"

    elif EVAL_TASK == 'Video-MME':
        evaluation_results = videomme_eval(output_jsonl_file, SELECTED_PROMPT_TYPE)
        print("Video-MME Evaluation Results:", evaluation_results)
        log_str = f"Video-MME Evaluation Complete. Results file: {output_jsonl_file}\n"
        all_duration_metrics = evaluation_results.get('all_duration_tasks', {})
        log_str += f"Overall Accuracy (all duration tasks): {all_duration_metrics.get('overall_accuracy', 0.0) * 100.:.2f}%\n"

    elif EVAL_TASK == 'LongVideoBench':
        evaluation_results = longvideobench_eval(output_jsonl_file, SELECTED_PROMPT_TYPE)
        print("LongVideoBench Evaluation Results:", evaluation_results)
        log_str = f"LongVideoBench Evaluation Complete. Results file: {output_jsonl_file}\n"
        log_str += f"Overall Accuracy: {evaluation_results.get('overall_accuracy', 0.0) * 100.:.2f}%\n"

    elif EVAL_TASK == 'TempCompass':
        evaluation_results = tempcompass_eval(output_jsonl_file, SELECTED_PROMPT_TYPE)
        print("TempCompass Evaluation Results:", evaluation_results)
        log_str = f"TempCompass Evaluation Complete. Results file: {output_jsonl_file}\n"
        log_str += f"Overall Accuracy: {evaluation_results.get('overall_accuracy', 0.0) * 100.:.2f}%\n"
    else:
        log_str = f"No specific final scoring implemented for task: {EVAL_TASK} in this script."
        main_logger.warning(log_str)

    if evaluation_results: # If any evaluation was performed
        print(log_str)
        main_logger.info("--- Final Evaluation Metrics ---")
        main_logger.info(log_str)
        main_logger.info(f"Full Evaluation Metrics Dictionary: {json.dumps(evaluation_results, indent=2)}")
    else:
        main_logger.info(f"No final evaluation metrics calculated for {EVAL_TASK} or evaluation failed.")