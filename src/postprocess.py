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
import json
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw
import supervision as sv
import open3d as o3d
import time
import imageio
#from modules_teleop.perception import PerceptionModule

# cotracker
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

# sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_bounding_box():
    return np.array([[-0.01, 0.8], [-0.01, 0.8], [-0.01, 0.8]])  # the world frame robot workspace

def depth2pcd(depth):
    with open("extrinsic_calibration.json", "r") as f:
        e = json.load(f)

        extrinsics = {}

        for serial, data in e.items():
            extrinsics[serial] = {
                "X_WC": np.array(data["X_WC"]),
            }

    fl_x = 607.9873657226562
    fl_y = 608.1102905273438
    cx = 317.81854248046875 
    cy = 230.1835479736328 
    w = 640 
    h = 480

    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

    depth = np.ascontiguousarray(depth.astype(np.float32))
    depth_image = o3d.geometry.Image(depth)

    pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        intrinsics,
        depth_trunc=1e9,  # no truncation
        stride=1,         # do not subsample
        project_valid_depth_only=False,
        depth_scale=1.0,
    )

    pointcloud.transform(extrinsics['044322073544']['X_WC'])

    return np.array(pointcloud.points)

def mp4_to_pil_list(path):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frame = Image.fromarray(np.array(frame))  # ensure RGB PIL image
        frames.append(frame)
    reader.close()
    return frames

def mp4_to_numpy_list(path):
    reader = imageio.get_reader(path)
    frames = [np.array(f) for f in reader]   # already RGB
    reader.close()
    return frames

#def chunk_mp4(path, chunk_size):
#    reader = imageio.get_reader(path)
#    frames = [np.array(f) for f in reader]   # already RGB
#    reader.close()
#
#    i = 0
#    chunk_paths = []
#    for chunk_idx in range(0, len(frames), chunk_size):
#
#        chunk_frames = np.array(frames[chunk_idx, chunk_idx + chunk_size])
#
#        chunk_path = path + f"chunk_{i}"
#        chunk_paths.append(chunk_path)
#
#        imageio.mimsave(
#            chunk_path,
#            np.array(chunk_frames).astype(int) * 255,
#            fps=30,
#            codec="libx264"
#        )
#
#    return chunk_paths

class PostProcessor:

    def __init__(self, 
            text_prompts='cloth.',
        ):

        self.H, self.W = 480, 640

        self.text_prompts = text_prompts

        self.bbox = get_bounding_box()

    def run_sam2(self):
        checkpoint = "models/weights/sam2/sam2.1_hiera_large.pt"

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 

        image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        #recordings = ["recording/video_044322073544.mp4", "recording/video_244622072067.mp4"]
        recordings = ["recording/video_044322073544_rgb.mp4"]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for r in recordings:

                rgb_frames_full_pil = mp4_to_pil_list(r)
                rgb_frames_full_np = mp4_to_numpy_list(r)

                n_frames = len(rgb_frames_full_pil)
                seq_len = 50

                mask_video = []

                for pivot_frame in range(0, n_frames, seq_len):

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

                        image =  rgb_frames_full_pil[ann_frame]

                        # ground
                        inputs = processor(images=image, text=self.text_prompts, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = grounding_model(**inputs)
                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            threshold=0.25,
                            target_sizes=[image.size[::-1]]
                        )

                        input_boxes = results[0]["boxes"].cpu().numpy()
                        objects = results[0]["labels"]

                        ## Convert PIL image to draw-able image
                        #draw = ImageDraw.Draw(image)

                        #for box in input_boxes:
                        #    x1, y1, x2, y2 = box.tolist()
                        #    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                        #image.show()   # or save

                        if len(objects) == 0:
                            print("No objects found. Looking back..")
                            continue
                            #assert "No objects found."

                        if len(objects) > 1:
                            assert "Too many objects found."

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
                    
                    # save this as a tmp video file to propagate over a chunk using SAM2
                    rgb_frames_chunk_np = rgb_frames_full_np[ann_frame: pivot_frame + seq_len ]

                    imageio.mimsave(
                        f"recording/tmp.mp4",
                        rgb_frames_chunk_np,
                        fps=30,
                        codec="libx264"
                    )

                    inference_state = video_predictor.init_state(video_path="recording/tmp.mp4")

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

                    # in order not to duplicate frames, only take the last seq_len frames
                    for idx, (frame_idx, segments) in enumerate(list(video_segments.items())[-seq_len:]):
                        
                        object_ids = list(segments.keys())
                        masks = list(segments.values())
                        masks = np.concatenate(masks, axis=0)

                        if masks.shape[0] > 1:
                            assert multi_objs
                            masks_save = np.logical_or.reduce(masks, axis=0, keepdims=True)
                        else:
                            masks_save = masks

                        mask_video.append(masks_save[0])

                        #vis = True
                        #if vis:
                        #    img = rgb_frames_chunk_np[idx]
                        #    detections = sv.Detections(
                        #        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                        #        mask=masks, # (n, h, w)
                        #        class_id=np.array(object_ids, dtype=np.int32),
                        #    )
                        #    box_annotator = sv.BoxAnnotator()
                        #    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                        #    label_annotator = sv.LabelAnnotator()
                        #    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
                        #    mask_annotator = sv.MaskAnnotator()
                        #    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                        #    cv2.imwrite("debug.png", annotated_frame)
                        #    cv2.waitKey(100)

                imageio.mimsave(
                    f"recording/test.mp4",
                    np.array(mask_video).astype(int) * 255,
                    fps=30,
                    codec="libx264"
                )

    def get_tracking(self):
            cotracker_predictor = CoTrackerPredictor(checkpoint="models/weights/cotracker/scaled_offline.pth", v2=False, offline=True, window_len=60).to('cuda')
            visualizer = Visualizer(save_dir="results_tracking", pad_value=120, linewidth=3)

            recordings = ["recording/video_044322073544_rgb.mp4"]
            mask_recordings = ['recording/test.mp4']
            depth_recordings = ["recording/044322073544_depth.npz"]

            for r, m_r, d_r in zip(recordings, mask_recordings, depth_recordings):

                rgb_frames_full_np = mp4_to_numpy_list(r)
                mask_frames_full_np = (np.array(mp4_to_numpy_list(m_r))[:, :, :,0] / 255).astype(np.uint8)

                with np.load(d_r) as data:
                    depth_frames_full_np = data['depth']

                start_frame = 0

                n_frames = len(rgb_frames_full_np) 
                pivot_skip = 5
                #seq_len = 100
                seq_len = 15

                velocities = []

                # iterate 0, 5, 10, etc
                for pivot_frame in range(start_frame, n_frames, pivot_skip):  # determine the speed for frame (pivot_frame, pivot_frame + pivot_skip)

                    mask_pivot = mask_frames_full_np[pivot_frame]

                    img = rgb_frames_full_np[pivot_frame]

                    img_masked = img.copy() * (mask_pivot[:, :, None] > 0)

                    frames = []

                    # find the bounding box of the mask
                    no_mask = False
                    mask_min_h_all, mask_min_w_all, mask_max_h_all, mask_max_w_all = self.H, self.W, 0, 0

                    # pivot_frame, pivot_frame + 1, pivot_frame + 2, etc. up to pivot_frame + seq_len
                    for frame_id in range(pivot_frame, min(pivot_frame + seq_len, n_frames)):

                        mask = mask_frames_full_np[frame_id]

                        if mask.sum() == 0:
                            no_mask = True
                            break

                        # get bounding box of each frame
                        ys, xs = np.where(mask > 0)

                        pad = 5
                        mask_min_h = max(0, ys.min() - pad)
                        mask_max_h = min(mask.shape[0], ys.max() + pad)
                        mask_min_w = max(0, xs.min() - pad)
                        mask_max_w = min(mask.shape[1], xs.max() + pad)

                        mask_min_h_all = min(mask_min_h_all, mask_min_h)
                        mask_min_w_all = min(mask_min_w_all, mask_min_w)
                        mask_max_h_all = max(mask_max_h_all, mask_max_h)
                        mask_max_w_all = max(mask_max_w_all, mask_max_w)

                    if no_mask:
                        mask_min_h_all = 0
                        mask_max_h_all = 200
                        mask_min_w_all = 0
                        mask_max_w_all = 200
                    else:
                        # Make bounding box a square for some reason? I gues cotracker likes squares
                        center = ((mask_max_h_all + mask_min_h_all) // 2, (mask_max_w_all + mask_min_w_all) // 2)
                        max_w_h = max(mask_max_h_all - mask_min_h_all, mask_max_w_all - mask_min_w_all)
                        mask_min_h_all = max(0, center[0] - max_w_h // 2)
                        mask_max_h_all = min(self.H, center[0] + max_w_h // 2)
                        mask_min_w_all = max(0, center[1] - max_w_h // 2)
                        mask_max_w_all = min(self.W, center[1] + max_w_h // 2)
                    
                    # crop image and masks according to bounding box, upscale them and put them all into a list
                    for frame_id in range(pivot_frame, min(pivot_frame + seq_len, n_frames)):
                        
                        mask = mask_frames_full_np[frame_id]
                        img = rgb_frames_full_np[frame_id]
                        
                        # crop image and mask according to bounding box, then upscale them
                        img = img[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]
                        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_LINEAR)

                        mask = mask[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]
                        mask = cv2.resize(mask, (mask.shape[1] * 4, mask.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

                        img_masked = img.copy() * (mask[:, :, None] > 0)
                        frames.append(torch.tensor(img_masked))
                    
                    video = torch.stack(frames).permute(0, 3, 1, 2)[None].float().to('cuda')  # B T C H W

                    # crop mask according to bounding box
                    mask_pivot = mask_pivot[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]

                    # shrink segmentation mask slightly to reduce 1 pixel edges, and producing a tighter and cleaner mask so cotracker only tracks points inside stable part of object
                    mask_pivot = cv2.erode(mask_pivot, np.ones((3, 3), np.uint8), iterations=1)

                    pred_tracks, pred_visibility = cotracker_predictor(
                        video, 
                        segm_mask=torch.from_numpy(mask_pivot).float().to('cuda')[None, None],
                        grid_size=80,
                    ) # B T N 2,  B T N
                    
                    # smooth the tracks multiple time (up to 3 times)
                    for _ in range(3):
                        for i in range(1, pred_tracks.shape[1] - 1):
                            #Weighted moving average so there is less jitter. More weight to current track (weight =2)
                            pred_tracks[:, i] = (2 * pred_tracks[:, i] + pred_tracks[:, i-1] + pred_tracks[:, i+1]) // 4
                    
                    vis = True
                    if vis:
                        visualizer.visualize(video, pred_tracks, pred_visibility, filename=f"tracks_{pivot_frame}")

                    # transform pred tracks and pred visibility to original image size
                    pred_tracks = pred_tracks.squeeze(0).cpu().numpy()  # T N 2
                    pred_visibility = pred_visibility.squeeze(0).cpu().numpy()  # T N 1
                    pred_tracks[:, :, 0] = pred_tracks[:, :, 0] / 4 + mask_min_w_all
                    pred_tracks[:, :, 1] = pred_tracks[:, :, 1] / 4 + mask_min_h_all
                    pred_tracks = pred_tracks[:, :, ::-1].copy()
                    
                    # calculate point speed in 3D
                    gap = 5
                    # Target frames starts at pivot frame
                    for target_frame in range(pivot_frame, min(pivot_frame + pivot_skip, n_frames - gap)):
                        depth_now = depth_frames_full_np[target_frame] / 1000.0
                        try:
                            depth_future = depth_frames_full_np[target_frame + gap] / 1000.0
                        except:
                            import ipdb; ipdb.set_trace()
                        
                        mask_now = mask_frames_full_np[target_frame]

                        # Gets all pixel coordinates the belong to object
                        mask_now_xy = np.where(mask_now > 0)
                        mask_now_xy = np.stack(mask_now_xy, axis=1)  # (n, 2)

                        # Gets all predicted tracker points that belong to object
                        track_now = pred_tracks[target_frame - pivot_frame]
                        vis_now = pred_visibility[target_frame - pivot_frame]

                        if len(track_now) == 0:
                            indices = np.zeros((0, 4)).astype(int)
                        else:
                            # each masked pixel is associated with the nearest track points
                            # indices.shape = (N_pixels, 4). Each pixel has 4 nearest tracker points
                            knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(track_now)
                            _, indices = knn.kneighbors(mask_now_xy)

                        mask_future = mask_frames_full_np[target_frame + gap]

                        # Gets all pixel coordinates that belong to object
                        mask_future_xy = np.where(mask_future > 0)
                        mask_future_xy = np.stack(mask_future_xy, axis=1)  # (n, 2)

                        # Gets all predicted tracker points that belong to object
                        track_future = pred_tracks[target_frame - pivot_frame + gap]
                        vis_future = pred_visibility[target_frame - pivot_frame + gap]

                        # for each pixel, get the x,y coordinate of each tracker in its knn
                        tracks_indices = track_now[indices]  # (n, k, 2)
                        tracks_future_indices = track_future[indices]  # (n, k, 2)

                        # for each pixel, get the visibility of each tracker in its knn
                        vis_indices = vis_now[indices]  # (n, k, 1)
                        vis_indices = np.all(vis_indices, axis=1).reshape(-1) # all neighbors must be 1, otherwise pixel gets a single 0 for its visibility
                        vis_future_indices = vis_future[indices]  # (n, k, 1)
                        vis_future_indices = np.all(vis_future_indices, axis=1).reshape(-1) # all neighbors must be 1, otherwise pixel gets a single 0 for its visibility

                        # for each pixel, get average position of track  from knn (belief state of where pixel is)
                        pred_mask_now_xy = np.round(tracks_indices.mean(axis=1)).astype(int)  # (n, 2)
                        pred_mask_now_xy[:, 0] = np.clip(pred_mask_now_xy[:, 0], 0, self.H - 1)
                        pred_mask_now_xy[:, 1] = np.clip(pred_mask_now_xy[:, 1], 0, self.W - 1)

                        # for each pixel, get average position of track  from knn (beliefe state of where pixel went)
                        pred_mask_future_xy = np.round(tracks_future_indices.mean(axis=1)).astype(int)  # (n, 2)  # actually we dont need to round here
                        pred_mask_future_xy[:, 0] = np.clip(pred_mask_future_xy[:, 0], 0, self.H - 1)
                        pred_mask_future_xy[:, 1] = np.clip(pred_mask_future_xy[:, 1], 0, self.W - 1)

                        ## extract depth and project to world coordinates
                        points_now = depth2pcd(depth_now)
                        points_future = depth2pcd(depth_future)

                        # make it into shape of image
                        points_now = points_now.reshape(depth_now.shape[0], depth_now.shape[1], 3)
                        points_future = points_future.reshape(depth_future.shape[0], depth_future.shape[1], 3)

                        # mask points
                        depth_threshold = [0.0, 2.0]

                        # build mask of valid pixels
                        depth_mask_now = np.logical_and((depth_now > depth_threshold[0]), (depth_now < depth_threshold[1]))  # (H, W)

                        # keep only points inside a 3d bounding box
                        depth_mask_now_bbox = np.logical_and(
                            np.logical_and(points_now[:, :, 0] > self.bbox[0][0], points_now[:, :, 0] < self.bbox[0][1]),
                            np.logical_and(points_now[:, :, 1] > self.bbox[1][0], points_now[:, :, 1] < self.bbox[1][1])
                        )  # does not include z axis

                        depth_mask_now_bbox = depth_mask_now_bbox.reshape(depth_now.shape[0], depth_now.shape[1])

                        # combine depth validity and workspace validity
                        depth_mask_now = np.logical_and(depth_mask_now, depth_mask_now_bbox)

                        #depth_mask_now_xy (N,) is just the N pixels with either a 1 or 0 for validity
                        depth_mask_now_xy = depth_mask_now[pred_mask_now_xy[:, 0], pred_mask_now_xy[:, 1]].reshape(-1)

                        # filter out any points that are not visible now or in the future
                        depth_mask_now_xy = np.logical_and(depth_mask_now_xy > 0, np.logical_and(vis_indices, vis_future_indices))  # filter out invalid points

                        # valid pixel points that satisfy:
                        #   - below depth threshold
                        #   - inside bounding box 
                        #   - visible in the present and in the future 
                        valid_idx = np.where(depth_mask_now_xy > 0)[0]
                        
                        # only keep valid pixels
                        mask_now_xy = mask_now_xy[valid_idx]
                        pred_mask_now_xy = pred_mask_now_xy[valid_idx]
                        pred_mask_future_xy = pred_mask_future_xy[valid_idx]

                        # build mask of valid pixels
                        depth_mask_future = np.logical_and((depth_future > depth_threshold[0]), (depth_future < depth_threshold[1]))  # (H, W)

                        # keep only points inside a 3d bounding box
                        depth_mask_future_bbox = np.logical_and(
                            np.logical_and(points_future[:, :, 0] > self.bbox[0][0], points_future[:, :, 0] < self.bbox[0][1]),
                            np.logical_and(points_future[:, :, 1] > self.bbox[1][0], points_future[:, :, 1] < self.bbox[1][1])
                        )  # does not include z axis

                        depth_mask_future_bbox = depth_mask_future_bbox.reshape(depth_future.shape[0], depth_future.shape[1])

                        # combine depth validity and workspace validity
                        depth_mask_future = np.logical_and(depth_mask_future, depth_mask_future_bbox)

                        #depth_mask_future_xy (N,) is just the N pixels with either a 1 or 0 for validity
                        depth_mask_future_xy = depth_mask_future[mask_future_xy[:, 0], mask_future_xy[:, 1]].reshape(-1)

                        # valid pixel points that satisfy:
                        #   - below depth threshold
                        #   - inside bounding box 
                        valid_idx_future = np.where(depth_mask_future_xy > 0)[0]

                        # only keep valid pixels
                        mask_future_xy_valid = mask_future_xy[valid_idx_future]

                        if valid_idx_future.shape[0] < 4 or valid_idx.shape[0] < 4:
                            print(f"Warning: not enough valid points for frame {target_frame}")
                            # if too few points, zero velocity
                            speed_now = np.zeros((self.H, self.W, 3))
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)

                        else:
                            # proj tracks
                            k = 4

                            # for each future predicted track location, we get the 4 nearest pixels in the future mask (get the 4 nearest pixels where we think the tracked partilce went)
                            knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(mask_future_xy_valid)
                            _, indices = knn.kneighbors(pred_mask_future_xy)

                            # produce list of pixel coordinates where each tracked point might have moved in the future
                            proj_mask_future_xy = mask_future_xy_valid[indices]  # (n, k, 2)
                            proj_mask_future_xy = proj_mask_future_xy.reshape(-1, 2)
                            proj_mask_future_xy = np.round(proj_mask_future_xy).astype(int)
                            proj_mask_future_xy[:, 0] = np.clip(proj_mask_future_xy[:, 0], 0, self.H - 1)
                            proj_mask_future_xy[:, 1] = np.clip(proj_mask_future_xy[:, 1], 0, self.W - 1)
                            
                            # pred_mask_now_xy: belief state where pixel is
                            # points_now: current points in 3d space from depth
                            points_now = points_now[pred_mask_now_xy[:, 0], pred_mask_now_xy[:, 1]].reshape(-1, 3)

                            # proj_mask_future_xy: belief state of where tracked point moved to (not just tracked particle, but average position of 4 nearest particles)
                            # points_future: future points in 3d space from depth
                            points_future = points_future[proj_mask_future_xy[:, 0], proj_mask_future_xy[:, 1]]

                            #TODO: get average position from above knn pixel locations
                            points_future = points_future.reshape(-1, k, 3).mean(axis=1)  # average the k points

                            speed = points_future - points_now  # actually velocity
                            speed /= (1. / 30. * gap)  # divide by the time interval
                            
                            #mask_now_xy: all pixels that belong to object
                            speed_now = np.zeros((self.H, self.W, 3))
                            speed_now[mask_now_xy[:, 0], mask_now_xy[:, 1]] = speed
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)
                            
                            # if velocity magnitude is way larger than typical values, it is an outlier
                            outlier_mask = speed_now_norm > min(1, speed_now_norm.mean() + 50 * speed_now_norm.std())
                            outlier_xy = np.stack(np.where(outlier_mask), axis=1)  # (n, 2)

                            # replace outlier with average of its 8 neighbors
                            for xy in outlier_xy:
                                speed_now[xy[0], xy[1]] = (speed_now[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2].sum(axis=(0, 1)) - speed_now[xy[0], xy[1]]) / 8
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)

                        # visualize the speed
                        #viz = True
                        #if viz:
                        #    speed_running_max = 0.5
                        #    _ = cv2.applyColorMap((speed_now_norm / speed_running_max * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        #    cv2.imwrite(save_dir_speed_cam / f"{target_frame:06d}.jpg", speed_now_norm / speed_running_max * 255)

                        # save the 3d vel
                        #np.savez_compressed(episode_data_dir_cam / "vel" / f"{target_frame:06d}.npz", vel=speed_now.astype(np.float16))

                        # we calculated the speed for the current timestep.
                        velocities.append(speed_now)

                        # vis depth mask
                        #depth_mask_vis = np.logical_and(depth_mask_now, mask_now)
                        #cv2.imwrite(episode_data_dir_cam / "depth_mask" / f"{target_frame:06d}.png", depth_mask_vis * 255)

                np.savez_compressed("recording/velocities.npz", velocities=np.array(velocities))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_prompts', type=str, default='')
    args = parser.parse_args()

    pp = PostProcessor(args.text_prompts)
    #pp.run_sam2()
    pp.get_tracking()
    #pp.get_pcd()
    #pp.get_sub_episodes()
