# Spatial-R1
The first MLLM trained using GRPO for spatial reasoning in videos

data sample

{"problem_id": 1, "problem": "Question:\nHow many monitor(s) are in this room?", "data_type": "video", "problem_type": ["object_counting"], "solution": "<answer>7</answer>", "video_filename": "scannet/scene0653_00.mp4"}

prediction example:

predicted_answer: <think>There are three chairs in the room. Two are placed around the dining table, and one is placed near the window.</think> <answer>3</answer>

ground truth: 3

We curate a dataset tailored for video spatial reasoning based on ScanNet and train Qwen2-VL-7B using grpo method. It achieves promising gains in VSI-Bench.

| Model                      | obj_appearance_order | object_abs_distance | object_counting | object_rel_distance | object_size_estimation | room_size_estimation | route_planning | object_rel_direction | Overall_acc |
| :------------------------- | :------------------- | :------------------ | :-------------- | :------------------ | :--------------------- | :------------------- | :------------- | :------------------- | :---------- |
| Qwen2.5-VL-7B(zero-shot) | 32.69                | 17.48               | 33.96           | 35.77               | 51.85                  | 36.60                | 29.38          | 37.73                | 34.43       |
| Qwen2.5-VL-7B(CoT)       | 30.42                | 12.10               | 15.84           | 31.83               | 19.12                  | 24.24                | 34.54          | 34.68                | 25.35       |
| Qwen2.5-VL-7B+GRPO       | 36.76                | 32.99               | 62.94           | 38.15               | 58.12                  | 31.04                | 28.87          | 32.72                | 41.81       |
