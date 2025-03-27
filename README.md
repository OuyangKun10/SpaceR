# Spatial-R1
The first MLLM trained using GRPO for spatial reasoning in videos

data sample

{"problem_id": 1, "problem": "Question:\nHow many monitor(s) are in this room?", "data_type": "video", "problem_type": ["object_counting"], "solution": "<answer>7</answer>", "video_filename": "scannet/scene0653_00.mp4"}

prediction example:

predicted_answer: <think>There are three chairs in the room. Two are placed around the dining table, and one is placed near the window.</think> <answer>3</answer>

ground truth: 3

We curate a dataset tailored for video spatial reasoning and train Qwen2-VL-7B using grpo method. It achieves promising gains in VSI-Bench.

| Model             | object_rel_distance | route_planning | object_rel_direction | object_rel_distance | object_counting |
|-------------------|----------------------|-----------------|-----------------------|----------------------|-----------------|
| Qwen2.5-VL-7B     | 35.77                | 29.38           | 32.69                 | 37.73                | 33.96           |
| Qwen2.5-VL-7B+GRPO| 36.76                | 32.99           | 34.63                 | 38.15                | 48.32           |
