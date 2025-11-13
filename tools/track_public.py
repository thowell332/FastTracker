#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2
import argparse
import glob
import json
from typing import Dict, List
from loguru import logger
from tqdm import tqdm

from yolox.tracker.fasttracker import Fasttracker


def load_mot_det(det_path: str) -> Dict[int, list]:
    """Loads an MOT-style detection file and returns a detection dictionary.
    
    MOT format: frame, -1, x, y, w, h, conf, -1, -1, -1
    Returns: dict mapping frame_id -> list of [x1, y1, x2, y2, score]
    
    :param det_path: path to the detection txt file
    :return: dictionary mapping frames to lists of detections
    """
    per_frame = {}
    if not os.path.exists(det_path):
        logger.warning(f"Detection file not found: {det_path}")
        return per_frame
    
    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            try:
                frame = int(parts[0])
                # parts[1] is -1 (track id, not used for detections)
                x = float(parts[2])  # left
                y = float(parts[3])  # top
                w = float(parts[4])  # width
                h = float(parts[5])  # height
                score = float(parts[6])  # confidence
                # Convert to x1, y1, x2, y2 format
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                per_frame.setdefault(frame, []).append([x1, y1, x2, y2, score])
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed line: {line.strip()}")
                continue
    return per_frame


def write_results(filename, results):
    """Writes tracking results to MOT/TrackEval format file.
    
    Results format: (frame_id, tlwhs, track_ids, scores)
    Output format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    
    This format is compatible with:
    - MOTChallenge evaluation format
    - TrackEval evaluation toolkit
    - Standard MOT tracking format
    """
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id, id=track_id,
                    x1=round(x1, 1), y1=round(y1, 1),
                    w=round(w, 1), h=round(h, 1),
                    s=round(score, 2)
                )
                f.write(line)
    logger.info(f'Saved results to {filename}')


def parse_seqinfo(seqinfo_path: str) -> dict:
    """Parse seqinfo.ini file to get sequence information."""
    info = {}
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ['imWidth', 'imHeight', 'seqLength', 'frameRate']:
                        info[key] = int(value)
                    elif key in ['imExt', 'imDir']:
                        info[key] = value
    return info


def load_config(config_path: str) -> dict:
    """Load tracking configuration from JSON file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    else:
        # Default FastTracker config
        default_config = {
            "track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "min_box_area": 100,
            "reset_velocity_offset_occ": 5,
            "reset_pos_offset_occ": 3,
            "enlarge_bbox_occ": 1.2,
            "dampen_motion_occ": 0.85,
            "active_occ_to_lost_thresh": 15,
            "init_iou_suppress": 0.8,
            "ROIs": {},
            "roi_repair_max_gap": 15,
            "dir_window_N": 10,
            "dir_margin_deg": 2.0,
        }
        logger.info("Using default FastTracker config")
        return default_config


def process_sequence(seq_path: str, det_path: str, result_path: str, args, config: dict):
    """Process a single sequence with public detections.
    
    :param seq_path: path to sequence folder (e.g., MOT17-02-DPM)
    :param det_path: path to detection file
    :param result_path: path to save tracking results
    :param args: arguments including tracker config
    :param config: tracking configuration dictionary
    """
    # Load detections
    dets = load_mot_det(det_path)
    if not dets:
        logger.warning(f"No detections found for {seq_path}")
        return
    
    # Get sequence info
    seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
    seq_info = parse_seqinfo(seqinfo_path)
    img_dir_name = seq_info.get('imDir', 'img1')
    img_ext = seq_info.get('imExt', '.jpg')
    img_dir = os.path.join(seq_path, img_dir_name)
    frame_rate = seq_info.get('frameRate', 30)
    
    if not os.path.exists(img_dir):
        logger.error(f"Image directory not found: {img_dir}")
        return
    
    # Create args object for FastTracker
    class DummyArgs:
        def __init__(self):
            self.mot20 = args.mot20
    
    # Adjust tracker parameters per sequence (similar to track.py)
    video_name = os.path.basename(seq_path)
    seq_config = config.copy()
    
    # Sequence-specific adjustments (from mot_evaluator.py logic)
    if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
        seq_config["track_buffer"] = 14
    elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
        seq_config["track_buffer"] = 25
    
    if video_name == 'MOT17-01-FRCNN':
        seq_config["track_thresh"] = 0.65
    elif video_name == 'MOT17-06-FRCNN':
        seq_config["track_thresh"] = 0.65
    elif video_name == 'MOT17-12-FRCNN':
        seq_config["track_thresh"] = 0.7
    elif video_name == 'MOT17-14-FRCNN':
        seq_config["track_thresh"] = 0.67
    elif video_name in ['MOT20-06', 'MOT20-08']:
        seq_config["track_thresh"] = 0.3
    
    # Override with command line arguments if provided
    if args.track_thresh is not None:
        seq_config["track_thresh"] = args.track_thresh
    if args.track_buffer is not None:
        seq_config["track_buffer"] = args.track_buffer
    if args.match_thresh is not None:
        seq_config["match_thresh"] = args.match_thresh
    if args.min_box_area is not None:
        seq_config["min_box_area"] = args.min_box_area
    
    targs = DummyArgs()
    
    # Initialize FastTracker
    tracker = Fasttracker(targs, seq_config, frame_rate=frame_rate)
    
    # Get image files
    img_files = sorted(glob.glob(os.path.join(img_dir, f'*{img_ext}')))
    if not img_files:
        logger.error(f"No images found in {img_dir}")
        return
    
    # Determine number of frames from sequence info or image count
    seq_length = seq_info.get('seqLength', len(img_files))
    num_frames = min(seq_length, len(img_files))
    
    # Process frames
    results = []
    frames_with_dets = sorted(dets.keys())
    
    logger.info(f"Processing {video_name}: {num_frames} frames total, {len(frames_with_dets)} frames with detections")
    
    for frame_idx in tqdm(range(1, num_frames + 1), desc=video_name, leave=False):
        # Load image to get dimensions
        img_idx = frame_idx  # MOT uses 1-based frame indexing
        img_filename = f'{img_idx:06d}{img_ext}'
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            # If image doesn't exist, try to get dimensions from first available image
            if img_files:
                img = cv2.imread(img_files[0])
                if img is not None:
                    img_h, img_w = img.shape[:2]
                else:
                    img_h, img_w = seq_info.get('imHeight', 1080), seq_info.get('imWidth', 1920)
            else:
                img_h, img_w = seq_info.get('imHeight', 1080), seq_info.get('imWidth', 1920)
        else:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                img_h, img_w = seq_info.get('imHeight', 1080), seq_info.get('imWidth', 1920)
            else:
                img_h, img_w = img.shape[:2]
        
        # Get detections for this frame
        frame_dets_list = dets.get(frame_idx, [])
        if not frame_dets_list:
            # No detections for this frame - create empty array
            dets_arr = np.zeros((0, 5), dtype=np.float32)
        else:
            frame_dets = np.array(frame_dets_list, dtype=np.float32)
            # Filter by confidence
            if args.conf_thresh > 0:
                frame_dets = frame_dets[frame_dets[:, 4] >= args.conf_thresh]
            dets_arr = frame_dets if frame_dets.size > 0 else np.zeros((0, 5), dtype=np.float32)
        
        # FastTracker expects:
        # - output_results: (N, 5) array with [x1, y1, x2, y2, score] or (N, 6) with class scores
        # - img_info: (img_h, img_w) tuple - (height, width) of original image
        # - img_size: (height, width) - if detections are in original coords, use original size
        #   The tracker calculates scale = min(img_size[0]/img_h, img_size[1]/img_w) and divides bboxes by it
        #   Since public detections are already in original image coordinates, we need scale=1.0
        info_imgs = (img_h, img_w)  # (height, width) of original image
        img_size = (img_h, img_w)  # Use original image size so scale = 1.0 (no scaling needed)
        
        # Update tracker
        online_targets = tracker.update(dets_arr, info_imgs, img_size)
        
        # Collect results
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh  # (top, left, width, height)
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6 if tlwh[3] > 0 else False
            # Filter by min box area and aspect ratio (same as mot_evaluator.py)
            if tlwh[2] * tlwh[3] > seq_config["min_box_area"] and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        
        if online_tlwhs:
            results.append((frame_idx, online_tlwhs, online_ids, online_scores))
    
    # Write results
    if results:
        write_results(result_path, results)
        logger.info(f"Processed {video_name}: {len(results)} frames with tracks")
    else:
        logger.warning(f"No tracks generated for {seq_path}")


def main():
    parser = argparse.ArgumentParser("FastTracker tracking with public detections")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MOT17",
        choices=["MOT17", "MOT20"],
        help="Dataset name"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets",
        help="Root directory of datasets"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val_half"],
        help="Dataset split"
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        help="Specific sequence to process (e.g., MOT17-02-FRCNN). If None, process all sequences."
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="track_results",
        help="Folder to save tracking results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file with tracking parameters"
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.01,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--track_thresh",
        type=float,
        default=None,
        help="Tracking confidence threshold (overrides config)"
    )
    parser.add_argument(
        "--track_buffer",
        type=int,
        default=None,
        help="Track buffer size (overrides config)"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=None,
        help="Matching threshold for tracking (overrides config)"
    )
    parser.add_argument(
        "--min_box_area",
        type=float,
        default=None,
        help="Minimum box area (overrides config)"
    )
    parser.add_argument(
        "--mot20",
        action="store_true",
        help="MOT20 dataset flag"
    )
    parser.add_argument(
        "--detector_type",
        type=str,
        default="FRCNN",
        choices=["FRCNN", "DPM", "SDP", "all"],
        help="Detector type to process (FRCNN, DPM, SDP, or all). Default: FRCNN"
    )
    
    args = parser.parse_args()
    
    # Set mot20 flag based on dataset
    if args.dataset == "MOT20":
        args.mot20 = True
        # MOT20 doesn't have detector subtypes, so ignore detector_type filter
        if args.detector_type != "all":
            logger.info(f"MOT20 doesn't have detector subtypes, ignoring --detector_type {args.detector_type}")
            args.detector_type = "all"
    
    # Load config
    config = load_config(args.config)
    
    # Set dataset path (try MOT17/MOT20 first, then mot/ for compatibility)
    if args.dataset == "MOT17":
        dataset_path = os.path.join(args.data_root, "MOT17", args.split)
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(args.data_root, "mot", args.split)
    else:
        dataset_path = os.path.join(args.data_root, "MOT20", args.split)
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return
    
    # Create result folder
    os.makedirs(args.result_folder, exist_ok=True)
    
    # Get sequences
    if args.seq:
        sequences = [args.seq]
    else:
        all_sequences = sorted([d for d in os.listdir(dataset_path) 
                               if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')])
        
        # Filter by detector type if specified
        if args.detector_type != "all":
            sequences = [seq for seq in all_sequences if args.detector_type in seq]
            logger.info(f"Filtering to {args.detector_type} sequences: {len(sequences)} out of {len(all_sequences)} sequences")
        else:
            sequences = all_sequences
    
    logger.info(f"Found {len(sequences)} sequences in {dataset_path}")
    if args.detector_type != "all":
        logger.info(f"Processing only {args.detector_type} sequences")
    
    # Process each sequence
    for seq_name in sequences:
        seq_path = os.path.join(dataset_path, seq_name)
        
        # Determine detection file path
        if args.split == "val_half":
            det_path = os.path.join(seq_path, "det", "det_val_half.txt")
        else:
            det_path = os.path.join(seq_path, "det", "det.txt")
        
        if not os.path.exists(det_path):
            logger.warning(f"Detection file not found: {det_path}, skipping {seq_name}")
            continue
        
        # Output file: sequence name with .txt extension
        result_path = os.path.join(args.result_folder, f"{seq_name}.txt")
        process_sequence(seq_path, det_path, result_path, args, config)
    
    logger.info(f"Tracking complete! Results saved to {args.result_folder}")
    logger.info(f"\nTo evaluate with TrackEval, copy results to:")
    logger.info(f"  TrackEval/data/trackers/mot_challenge/{args.dataset}-{args.split}/FastTracker_public/data/")


if __name__ == '__main__':
    main()

