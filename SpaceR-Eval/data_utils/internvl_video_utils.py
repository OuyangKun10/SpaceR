import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import io
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # candidate ratios
    target_ratios = sorted(
        [(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num],
        key=lambda x: x[0] * x[1]
    )

    # find closest
    best_diff = float("inf")
    best_ratio = (1, 1)
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_diff:
            best_diff = ratio_diff
            best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        col = i % (target_width // image_size)
        row = i // (target_width // image_size)
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) > 1:
        thumbnail = image.resize((image_size, image_size))
        processed_images.append(thumbnail)

    return processed_images

def get_frame_indices(bound, fps, max_frame, num_segments=8, first_idx=0):
    if bound:
        start, end = bound
    else:
        start, end = -1e6, 1e6

    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)

    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + seg_size * idx + seg_size / 2)
        for idx in range(num_segments)
    ])
    return frame_indices
def load_image(image_byte, input_size=448, max_num=12):
    if isinstance(image_byte, bytes):
        image = Image.open(io.BytesIO(image_byte)).convert('RGB')  # ← 关键：从bytes转成PIL.Image
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
def load_video_internvl(video_path, bound=None, input_size=448, max_num=1, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    frame_indices = get_frame_indices(bound, fps, max_frame, num_segments=num_segments)
    transform = build_transform(input_size)
    
    pixel_values_list, num_patches_list = [], []

    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        tile_tensors = [transform(tile) for tile in tiles]
        stacked = torch.stack(tile_tensors)
        pixel_values_list.append(stacked)
        num_patches_list.append(stacked.shape[0])

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list
