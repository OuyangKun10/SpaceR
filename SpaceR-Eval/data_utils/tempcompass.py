from util import *
import time
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration, AutoProcessor
import numpy as np
from tqdm import tqdm
from loguru import logger as eval_logger

system_message = "You are a helpful assistant"
QUESTION_TEMPLATE = (
    "Question: {Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "default":
    {"pre-prompt":"Question: {Question}\n",
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.).",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14).",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer.",
    "free-form": " Please provide your text answer.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14)."},
   "thinking": 
   {"pre-prompt":QUESTION_TEMPLATE,
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."}
}


ANSWER_PROMPT = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""  
}

def evaluate_tempcompass(rank, world_size, parquet_file, video_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, fps=1, target_resolution=(256, 256), debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="default"):
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")
    df=pd.read_json(parquet_file)

    if debug:
        df = df.sample(n=debug_size)
        logger.info(f"Process {rank} Debug mode enabled, randomly processing {debug_size} samples.")
    if world_size > 1:
        df_shard = np.array_split(df, world_size)[rank]
    else:
        df_shard = df
    logger.info(f"Rank {rank} Shard size: {len(df_shard)}")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = 'left'

    
    if world_size == 1 and len(gpu_ids.split(',')) > 1:

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
       
        model = accelerator.prepare(model)
        model.eval()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval().to(device)
        model = accelerator.prepare(model)

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has empty shard, skipping processing.")
        return os.path.join(output_dir, f"TempCompass_results_rank_{rank}.jsonl"), 0

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Process {rank}", total= (total_samples + batch_size - 1) // batch_size):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]

        batch_messages_list = []
        batch_row_infos = []
        prompt_list = []
        for _, row in batch_df.iterrows():
            video_path = os.path.join(video_dir, f"{row['video_id']}.mp4")
            if not os.path.exists(video_path):
                print("Warning: video not found at: ", video_path)
                continue

            frames, timestamps, duration = load_video_frames(video_path, num_frames, fps, target_resolution) 
            if frames is None:
                print("Warning: failed to extract frames for: ", video_path)
                continue

            question = "Question:\n"+row["question"]

            if row["problem_type"] == 'multiple choice':
                question =row['problem'] + "Options:\n"
                for op in row["options"]:
                    question += op + "\n"
            else:
                question = row['problem']
            prompt_text=TYPE_TEMPLATE[prompt_type]['pre-prompt'].format(Question=question) +TYPE_TEMPLATE[prompt_type][row['problem_type']]
            prompt_list.append(prompt_text)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
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
            if prompt_type=="default":
                max_new_token=128
            else:
                max_new_token=1024
            generated_ids_batch = model.generate(**inputs_batch, use_cache=True, max_new_tokens=max_new_token, temperature=0.01)
            generated_ids_trimmed_batch = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
            ]
            predicted_answers_batch = processor.batch_decode(
                generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception as e:
            logger.error(f"Process {rank} batch inference failure: {e}")
            predicted_answers_batch = [""] * len(batch_messages_list)

        for i, predicted_answer in enumerate(predicted_answers_batch):
            row = batch_row_infos[i]
            ground_truth = row['solution']
            prompt_text = prompt_list[i]
            results.append({
                'question': question,
                'answer': ground_truth,
                'prediction': predicted_answer,
                'promt': prompt_text
            })

    process_output_file = os.path.join(output_dir, f"TempCompass_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} results saved to: {process_output_file}, time usage: {elapsed_time_process_formatted}")
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

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]
def eval_rule(video_llm_output, question, answer):
    option_strs = question.split("\n")[1:]  
    option_sents = [opt.split(': ')[1] for opt in option_strs]  
    option_inds = [opt.split(': ')[0] for opt in option_strs] + [opt.split(': ')[0].replace('Sentence ', '').replace('Option ', '').replace('Caption ', '') for opt in option_strs]   
    video_llm_pred = None
    for option_str in option_strs:
        if option_str==video_llm_output:
            video_llm_pred = option_str
    for option_sent in option_sents:
        if option_sent==video_llm_output or (') ' in video_llm_output and option_sent==video_llm_output.split(') ')[1]):
            video_llm_pred = option_sent
    for option_ind in option_inds:
        if option_ind==video_llm_output or option_ind==video_llm_output.replace('.', ''):
            video_llm_pred = option_ind
    if video_llm_pred is None:
        return "fail"
    else:
        return 1 if video_llm_pred==answer or video_llm_pred==answer.split(":")[0] or video_llm_pred==answer.split(": ")[1] or video_llm_pred==answer.split(": ")[0].split()[1] else 0
def eval_mc(pred,ans):
    if pred==ans:
        rating = 1
    elif pred in ["A", "B", "C", "D"]:
        rating = 1 if pred==ans[0] else 0
    elif any(pred.startswith(prefix) for prefix in ['A.', 'B.', 'C.', 'D.']):
        rating = 1 if pred.split('.')[0]==ans[0] else 0
    elif any(pred.startswith(prefix) for prefix in ['A)', 'B)', 'C)', 'D)']):
        rating = 1 if pred.split(')')[0]==ans[0] else 0
    else:  
        rating = 0               
    return rating
def extract_pred(video_llm_output):
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "yes"
    elif video_llm_output.startswith("no"):
        return "no"
    else:
        return False

def tempcompass_aggregate_results(results):
    results_df = pd.DataFrame(results)

    output = {}

    output['overall_accuracy'] = results_df['accuracy'].mean()


    eval_logger.info(f"Evaluation results: {output}")

    return output

def tempcompass_eval(jsonl_file_path,mode="thinking"):
    results = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
    
            doc["answer"]=extract_answer_text(doc["answer"])
            if mode=="thinking" and "<answer>" in doc["prediction"]:
                doc["prediction"]=extract_answer_text(doc["prediction"])
            doc['accuracy']=eval_mc(doc["prediction"],doc["answer"])

            results.append(doc)
    aggregated_results = tempcompass_aggregate_results(results)
    return aggregated_results 


