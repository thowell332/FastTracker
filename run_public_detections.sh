#!/usr/bin/env bash
# Script to run FastTracker with public detections on MOT17/MOT20 datasets
# This uses the public detections from the dataset instead of YOLOX detector

set -euo pipefail

# --------------------- USER SETTINGS ---------------------

DATASET="MOT17"  # Options: MOT17, MOT20
SPLIT="train"    # Options: train, val_half
DETECTOR_TYPE="FRCNN"  # Options: FRCNN, DPM, SDP, all (for MOT17, use FRCNN to avoid duplicate processing)
CONFIG_FILE="./configs/004_default.json"  # Path to config file (or None for defaults)
RESULT_FOLDER="track_results_public"  # Folder to save results

# ---------------------------------------------------------

TRACK_SCRIPT="tools/track_public.py"

echo "==============================="
echo "Running FastTracker with Public Detections"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Detector Type: $DETECTOR_TYPE"
echo "Config: $CONFIG_FILE"
echo "==============================="

# Build command
CMD="python $TRACK_SCRIPT --dataset $DATASET --split $SPLIT --result_folder $RESULT_FOLDER --detector_type $DETECTOR_TYPE"

if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
    echo "Using config file: $CONFIG_FILE"
else
    echo "Using default config"
fi

# Add MOT20 flag if needed
if [ "$DATASET" == "MOT20" ]; then
    CMD="$CMD --mot20"
fi

# Run tracking
echo "Running: $CMD"
$CMD

echo "==============================="
echo "Tracking complete!"
echo "Results saved to: $RESULT_FOLDER"
echo "==============================="

