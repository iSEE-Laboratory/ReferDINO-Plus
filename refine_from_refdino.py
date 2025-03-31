import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm

from utils.track_utils import sample_points_from_masks
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_bounding_box(mask):
    mask = np.array(mask)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0), 0

    row_min = np.where(rows)[0][0]
    row_max = np.where(rows)[0][-1]
    col_min = np.where(cols)[0][0]
    col_max = np.where(cols)[0][-1]

    return (row_min, col_min, row_max, col_max), (row_max - row_min) * (col_max - col_min)

def process_video(args, video_file, device_id):
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model, device=device)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    threshold = 0.275
    num_points = 20
    use_point = False
    use_box = False
    use_mask = True

    predata_path = "Annotations_intermediate"
    data = np.load(os.path.join(predata_path, video_file), allow_pickle=True)
    video_path = str(data['video_path'])
    video_name = video_path.split('/')[-1]

    frame_names = sorted(os.listdir(video_path))
    H, W = data['size']
    video_len = len(frame_names)
    exp_ids = [k for k in data.keys() if k not in ('video_path', 'size')]

    for exp_id in exp_ids:
        meta = data[exp_id].item()
        scores = torch.from_numpy(meta['scores']).to(device)
        boxes = torch.from_numpy(meta['boxes']).to(device)
        masks = torch.from_numpy(meta['masks']).to(device)
        valid_frame = (scores > threshold).float()
        avg_scores = (scores * valid_frame).sum(0) / (valid_frame.sum(0) + 1e-6)

        idx = avg_scores > threshold
        if idx.any():
            scores = scores[:, idx]
            boxes = boxes[:, idx]
            masks = masks[:, idx]

            all_boxes = rescale_bboxes(boxes, H, W)
            all_masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False) > 0.

            frame_ids = scores.argmax(0)
            N = len(frame_ids)
            N_ind = torch.arange(N, device=device)
            boxes = all_boxes[frame_ids, N_ind]
            masks = all_masks[frame_ids, N_ind]
            frame_ids = frame_ids.cpu().numpy().tolist()

            masks = masks.cpu().numpy()
            boxes = boxes.cpu().numpy()
            all_masks = all_masks.cpu().numpy()

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = video_predictor.init_state(video_path=video_path)

                if use_box:
                    for object_id, (box, fid) in enumerate(zip(boxes, frame_ids)):
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=fid,
                            obj_id=object_id,
                            box=box,
                        )

                if use_point:
                    all_sample_points = sample_points_from_masks(masks=masks, num_points=num_points)

                    labels = np.ones((num_points), dtype=np.int32)
                    for object_id, (points, fid) in enumerate(zip(all_sample_points, frame_ids)):
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=fid,
                            obj_id=object_id,
                            points=points,
                            labels=labels
                        )

                if use_mask:
                    for object_id, (mask, fid) in enumerate(zip(masks, frame_ids)):
                        labels = np.ones((1), dtype=np.int32)
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=fid,
                            obj_id=object_id,
                            mask=mask
                        )

                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(state):
                    video_segments[out_frame_idx] = (out_mask_logits > 0).squeeze(1).cpu().numpy()
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(state,
                                                                                                      reverse=True):
                    video_segments[out_frame_idx] = (out_mask_logits > 0).squeeze(1).cpu().numpy()
        else:
            all_masks = np.zeros((video_len, 1, H, W), dtype=bool)
            video_segments = {t: all_masks[t] for t in range(video_len)}

        new_save_path = os.path.join("refine_output", video_name, exp_id)
        os.makedirs(new_save_path, exist_ok=True)

        cnt = valid_frames =0

        for t in range(video_len):
            mask_ori = all_masks[t].sum(0).clip(0, 1)
            mask_sam2 = video_segments[t].sum(0).clip(0, 1)
            box_ori, area_ori = get_bounding_box(mask_ori)
            box_sam2, area_sam2 = get_bounding_box(mask_sam2)
            if area_ori > 0 and area_sam2 > 0:
                valid_frames += 1
                if area_sam2 * 3 < area_ori * 2:
                    cnt += 1

        multi_obj = (cnt * 2 > valid_frames)

        for t in range(video_len):
            frame_name = frame_names[t].split(".")[0]
            if multi_obj:
                mask = (video_segments[t] + all_masks[t]).sum(0).clip(0, 1)
            else:
                mask = video_segments[t].sum(0).clip(0, 1)
            mask = mask.astype(np.float32)
            mask = Image.fromarray(mask * 255).convert('L')
            save_file = os.path.join(new_save_path, frame_name + ".png")
            mask.save(save_file)


def rescale_bboxes(bboxes, h, w):
    cx = bboxes[..., 0] * w
    cy = bboxes[..., 1] * h
    bw = bboxes[..., 2] * w
    bh = bboxes[..., 3] * h

    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh

    return torch.stack([x1, y1, x2, y2], dim=-1)


def process_video_worker(args, video_list, device_id):
    with tqdm(total=len(video_list), desc=f"GPU {device_id}", position=device_id) as pbar:
        for video_file in video_list:
            process_video(args, video_file, device_id)
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='Video processing with multiple GPUs')
    parser.add_argument('--gids', nargs='+', type=int, required=True, help='GPU IDs to use')
    args = parser.parse_args()

    predata_path = "Annotations_intermediate"
    video_list = os.listdir(predata_path)

    mp.set_start_method('spawn')

    num_gpus = len(args.gids)
    videos_per_gpu = (len(video_list) + num_gpus - 1) // num_gpus

    processes = []
    for i, gpu_id in enumerate(args.gids):
        start_idx = i * videos_per_gpu
        end_idx = min((i + 1) * videos_per_gpu, len(video_list))
        gpu_video_list = video_list[start_idx:end_idx]

        if gpu_video_list:
            p = mp.Process(target=process_video_worker, args=(args, gpu_video_list, gpu_id))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()