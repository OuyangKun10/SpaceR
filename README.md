# Spatial-R1
The first MLLM trained using GRPO for spatial reasoning in videos

data sample

{"problem_id": 1, "problem": "Question:\nHow many monitor(s) are in this room?", "data_type": "video", "problem_type": ["object_counting"], "solution": "<answer>7</answer>", "video_filename": "scannet/scene0653_00.mp4"}

prediction example:

predicted_answer: <think>There are three chairs in the room. Two are placed around the dining table, and one is placed near the window.</think><answer>3</answer>

ground truth: 3
