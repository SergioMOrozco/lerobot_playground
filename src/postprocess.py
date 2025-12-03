# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import sys

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from typing import Union
from pathlib import Path
import argparse
import os
import subprocess
import numpy as np
import glob
import cv2
import torch
import shutil
from PIL import Image, ImageDraw
import supervision as sv
import open3d as o3d
import time
import imageio
#from modules_teleop.perception import PerceptionModule

# cotracker
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

# sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

def mp4_to_pil_list(path):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frame = Image.fromarray(np.array(frame))  # ensure RGB PIL image
        frames.append(frame)
    reader.close()
    return frames

def load_camera(episode_data_dir):
    intr = np.load(episode_data_dir / 'calibration' / 'intrinsics.npy').astype(np.float32)
    rvec = np.load(episode_data_dir / 'calibration' / 'rvecs.npy')
    tvec = np.load(episode_data_dir / 'calibration' / 'tvecs.npy')
    R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
    T = [tvec[i, :, 0] for i in range(tvec.shape[0])]
    extrs = np.zeros((len(R), 4, 4)).astype(np.float32)
    for i in range(len(R)):
        extrs[i, :3, :3] = R[i]
        extrs[i, :3, 3] = T[i]
        extrs[i, 3, 3] = 1
    return intr, extrs

class PostProcessor:

    def __init__(self, 
            text_prompts='cloth.',
        ):

        self.H, self.W = 480, 640

        #self.bbox = get_bounding_box()  # 3D bounding box of the scene

        self.text_prompts = text_prompts

    def run_sam2(self):
        checkpoint = "models/weights/sam2/sam2.1_hiera_large.pt"

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 

        image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        #perception_module = PerceptionModule(vis_path=self.data_dir / "perception_vis", device='cuda', load_model=False)

        recordings = ["recording/video_044322073544.mp4", "recording/video_244622072067.mp4"]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for r in recordings:


                #episode_data_dir_cam = self.data_dir / f"episode_{episode_id:04d}" / f"camera_{cam}" 
                
                rgb_frames = mp4_to_pil_list(r)
                #rgb_paths = sorted(glob.glob(str(episode_data_dir_cam / 'rgb' / '*.jpg')))
                #depth_paths = sorted(glob.glob(str(episode_data_dir_cam / 'depth' / '*.png')))
                
                #n_frames = min(len(rgb_paths), self.max_frames)
                #n_frames = min(len(rgb_frames), self.max_frames)
                n_frames = len(rgb_frames)
                #seq_len = 720
                seq_len = 100

                for pivot_frame in range(0, n_frames, seq_len):
                    #print(f"[run_sam2] Processing episode {episode_id} camera {cam} pivot frame {pivot_frame}")

                    #if os.path.exists(save_dir / "mask" / f'episode_{episode_id:04d}_camera_{cam}' / f'pivot_frame_{pivot_frame:06d}'):
                    #    continue

                    masks = np.zeros((1, 1))
                    ann_frame = pivot_frame + 1
                    ann_frame_idx = 0
                    objects = [None, None]
                    no_objs = False
                    multi_objs = False
                    
                    while masks.sum() == 0:
                        if ann_frame == 0:
                            import ipdb; ipdb.set_trace()
                        ann_frame -= 1
                        print(f"[run_sam2] Finding a frame with mask for frame {ann_frame}")

                        #save_dir_video_cam = save_dir / "video_frames" / f"episode_{episode_id:04d}_camera_{cam}"
                        #save_dir_video_pivot = save_dir / "video_frames" / f"episode_{episode_id:04d}_camera_{cam}" / f"pivot_frame_{pivot_frame:06d}"
                        #os.makedirs(save_dir_video_pivot, exist_ok=True)
                        #for frame_id in range(ann_frame, min(n_frames, pivot_frame + seq_len)):  # save video
                        #    subprocess.run(f'cp {rgb_paths[frame_id]} {save_dir_video_pivot / f"{frame_id:06d}.jpg"}', shell=True)
                        #rgb_paths_segment = rgb_paths[ann_frame:min(n_frames, pivot_frame + seq_len)]

                        #rgb_path = rgb_paths[ann_frame]
                        #image = Image.open(rgb_path)

                        image =  rgb_frames[ann_frame]

                        # ground
                        inputs = processor(images=image, text=self.text_prompts, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = grounding_model(**inputs)
                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            #box_threshold=0.25,
                            threshold=0.25,
                            #text_threshold=0.3,
                            target_sizes=[image.size[::-1]]
                        )

                        input_boxes = results[0]["boxes"].cpu().numpy()
                        objects = results[0]["labels"]

                        # Convert PIL image to draw-able image
                        draw = ImageDraw.Draw(image)

                        for box in input_boxes:
                            x1, y1, x2, y2 = box.tolist()
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                        image.show()   # or save

                        if len(objects) == 0:
                            no_objs = True
                            break

                        #TODO: Worry about this later
                        #TODO: Rather than do this stuff, just query use which bounding box to choose.
                        #if len(objects) > 1:
                        #    objects_masked = []
                        #    input_boxes_masked = []
                        #    depth = cv2.imread(depth_paths[ann_frame], cv2.IMREAD_UNCHANGED) / 1000.0
                        #    mask = perception_module.get_mask_raw(depth, intrs[cam], extrs[cam])
                        #    for i, obj in enumerate(objects):
                        #        if obj == '':
                        #            continue
                        #        box = input_boxes[i].astype(int)
                        #        mask_box = mask[box[1]:box[3], box[0]:box[2]]
                        #        if mask_box.sum() > 0: # and not (mask_box.shape[0] > 200 and mask_box.shape[1] > 300):
                        #            objects_masked.append(obj)
                        #            input_boxes_masked.append(box)
                        #    objects = objects_masked
                        #    input_boxes = input_boxes_masked
                        #    if len(objects) == 0:
                        #        no_objs = True
                        #        break
                        #    elif len(objects) > 1:
                        #        multi_objs = True

                        #image_predictor.set_image(np.array(image.convert("RGB")))
                        image_predictor.set_image(np.array(image))
                        masks, scores, logits = image_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )
                        if masks.ndim == 3:
                            masks = masks[None]
                            scores = scores[None]
                            logits = logits[None]
                        elif masks.ndim == 4:
                            assert multi_objs
                    
                    if no_objs:
                        assert "No objects found".
                        #print(episode_id, cam, 'no_objs')
                        if os.path.exists(save_dir.parent / "sam2_select_images_mask" / f'{episode_id}_{cam}_{pivot_frame}.png'):
                            img = cv2.imread(
                                str(save_dir.parent / "sam2_select_images_mask" / f'{episode_id}_{cam}_{pivot_frame}.png'),
                                cv2.IMREAD_UNCHANGED
                            )
                            alpha = img[:, :, 3] / 255.0
                            masks = alpha > 0

                            h_max = np.where(masks.sum(1) > 0)[0].max() + 3
                            h_min = np.where(masks.sum(1) > 0)[0].min() - 3
                            w_max = np.where(masks.sum(0) > 0)[0].max() + 3
                            w_min = np.where(masks.sum(0) > 0)[0].min() - 3

                            objects = ['paper']
                            input_boxes = np.array([[w_min, h_min, w_max, h_max]]).astype(np.float32)

                            masks = masks[None, None]
                            scores = np.ones((1, 1))
                            logits = np.ones((1, 1))

                        else:
                            os.makedirs(episode_data_dir_cam / "mask", exist_ok=True)
                            for frame_id in range(len(rgb_paths_segment)):
                                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                                cv2.imwrite(episode_data_dir_cam / "mask" / f"{(frame_id + ann_frame):06d}.png", mask)
                            continue

                    inference_state = video_predictor.init_state(video_path=str(save_dir_video_pivot))

                    # Using box prompt
                    for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            box=box,
                        )

                    video_segments = {}  # video_segments contains the per-frame segmentation results
                    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    
                    del inference_state
                    torch.cuda.empty_cache()
                    
                    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(objects, start=1)}

                    # dir to save intermediate results
                    save_dir_mask_cam = save_dir / "mask" / f"episode_{episode_id:04d}_camera_{cam}"
                    save_dir_mask_pivot = save_dir_mask_cam / f"pivot_frame_{pivot_frame:06d}"
                    os.makedirs(save_dir_mask_pivot, exist_ok=True)

                    # dir to save final results
                    os.makedirs(episode_data_dir_cam / "mask", exist_ok=True)

                    for idx, (frame_idx, segments) in enumerate(video_segments.items()):
                        if idx != frame_idx:
                            import ipdb; ipdb.set_trace()
                        try:
                            img = cv2.imread(os.path.join(save_dir_video_pivot, rgb_paths_segment[frame_idx]))
                        except:
                            import ipdb; ipdb.set_trace()
                        
                        object_ids = list(segments.keys())
                        masks = list(segments.values())
                        masks = np.concatenate(masks, axis=0)

                        if masks.shape[0] > 1:
                            assert multi_objs
                            masks_save = np.logical_or.reduce(masks, axis=0, keepdims=True)
                        else:
                            masks_save = masks
                        cv2.imwrite(episode_data_dir_cam / "mask" / f"{(frame_idx + ann_frame):06d}.png", masks_save[0] * 255)

                        vis = True
                        if vis:
                            detections = sv.Detections(
                                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                                mask=masks, # (n, h, w)
                                class_id=np.array(object_ids, dtype=np.int32),
                            )
                            box_annotator = sv.BoxAnnotator()
                            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                            label_annotator = sv.LabelAnnotator()
                            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
                            mask_annotator = sv.MaskAnnotator()
                            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                            cv2.imwrite(save_dir_mask_pivot / f"annotated_frame_{(frame_idx + ann_frame):06d}.jpg", annotated_frame)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_prompts', type=str, default='')
    args = parser.parse_args()

    pp = PostProcessor(args.text_prompts)
    pp.run_sam2()
    #pp.get_tracking()
    #pp.get_pcd()
    #pp.get_sub_episodes()
