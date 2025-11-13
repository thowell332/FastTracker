#!/usr/bin/env python3
"""
Simple evaluation script for tracking results.
Reads GT and tracker results, computes HOTA, CLEAR, and Identity metrics.
Reuses code from track.py for CLEAR metrics (motmetrics).
"""

import os
import sys
import glob
import argparse
import csv
from pathlib import Path
from collections import OrderedDict

import motmetrics as mm
import numpy as np

# Add TrackEval to path
TRACKEVAL_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TrackEval")
sys.path.insert(0, os.path.abspath(TRACKEVAL_ROOT))

try:
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.metrics import HOTA, CLEAR as TrackEvalCLEAR, Identity
    from trackeval import Evaluator
    TRACKEVAL_AVAILABLE = True
except ImportError:
    TRACKEVAL_AVAILABLE = False
    print("Warning: TrackEval not available. Only CLEAR metrics will be computed.")


def load_mot_file(file_path, min_confidence=-1):
    """Load MOT format file using motmetrics (same as track.py)."""
    return mm.io.loadtxt(file_path, fmt='mot15-2D', min_confidence=min_confidence)


def compare_dataframes(gts, ts):
    """Compare GT and tracker dataframes - reused from track.py."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
    return accs, names


def evaluate_with_motmetrics(gt_files, tracker_files, dataset_name="MOT17"):
    """Evaluate using motmetrics (CLEAR metrics) - reused from track.py."""
    mm.lap.default_solver = 'lap'
    
    # Load GT and tracker files
    gt = OrderedDict([(Path(f).parts[-3], load_mot_file(f, min_confidence=1)) for f in gt_files])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], load_mot_file(f, min_confidence=-1)) for f in tracker_files])
    
    # Compare
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)
    
    # Compute metrics (same as track.py)
    metrics_list = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                   'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                   'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics_list, generate_overall=True)
    
    # Normalize ratios (same as track.py)
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']
    }
    for divisor in div_dict:
        if divisor in summary.columns:
            for divided in div_dict[divisor]:
                if divided in summary.columns:
                    summary[divided] = (summary[divided] / summary[divisor])
    
    # Also compute motchallenge_metrics which includes idf1 (same as track.py)
    motchallenge_metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary_full = mh.compute_many(accs, names=names, metrics=motchallenge_metrics, generate_overall=True)
    
    # Add idf1 to summary if available
    if 'idf1' in summary_full.columns:
        summary['idf1'] = summary_full['idf1']
    
    return summary, gt, ts


def setup_trackeval_structure(dataset, split, results_dir, tracker_name="FastTracker_public"):
    """Set up minimal TrackEval directory structure using symlinks/copies."""
    import shutil
    
    # Create temporary TrackEval structure
    trackeval_data = os.path.join(TRACKEVAL_ROOT, "data")
    os.makedirs(trackeval_data, exist_ok=True)
    
    gt_folder = os.path.join(trackeval_data, "gt", "mot_challenge")
    trackers_folder = os.path.join(trackeval_data, "trackers", "mot_challenge")
    
    benchmark_name = f"{dataset}-{split}"
    
    # Setup GT folder
    gt_dst = os.path.join(gt_folder, benchmark_name)
    os.makedirs(gt_dst, exist_ok=True)
    
    # Setup tracker folder
    tracker_dst = os.path.join(trackers_folder, benchmark_name, tracker_name, "data")
    os.makedirs(tracker_dst, exist_ok=True)
    
    # Setup seqmaps
    seqmaps_folder = os.path.join(gt_folder, "seqmaps")
    os.makedirs(seqmaps_folder, exist_ok=True)
    seqmap_file = os.path.join(seqmaps_folder, f"{benchmark_name}.txt")
    
    # Get dataset path
    if dataset == "MOT17":
        dataset_path = os.path.join("datasets", "MOT17", split)
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join("datasets", "mot", split)
    else:
        dataset_path = os.path.join("datasets", dataset, split)
    
    # Get result files
    result_files = sorted(glob.glob(os.path.join(results_dir, "*.txt")))
    sequences = []
    
    # Copy/link GT and tracker files
    for result_file in result_files:
        seq_name = os.path.splitext(os.path.basename(result_file))[0]
        
        # GT source
        gt_src = os.path.join(dataset_path, seq_name, "gt", "gt.txt")
        if not os.path.exists(gt_src):
            continue
        
        # GT destination
        gt_seq_dst = os.path.join(gt_dst, seq_name, "gt")
        os.makedirs(gt_seq_dst, exist_ok=True)
        gt_file_dst = os.path.join(gt_seq_dst, "gt.txt")
        
        # Copy or link GT
        if os.path.exists(gt_file_dst):
            os.remove(gt_file_dst)
        try:
            os.symlink(os.path.abspath(gt_src), gt_file_dst)
        except OSError:
            # Fallback to copy if symlink fails
            shutil.copy2(gt_src, gt_file_dst)
        
        # Copy seqinfo.ini if it exists
        seqinfo_src = os.path.join(dataset_path, seq_name, "seqinfo.ini")
        if os.path.exists(seqinfo_src):
            seqinfo_dst = os.path.join(gt_dst, seq_name, "seqinfo.ini")
            if os.path.exists(seqinfo_dst):
                os.remove(seqinfo_dst)
            try:
                os.symlink(os.path.abspath(seqinfo_src), seqinfo_dst)
            except OSError:
                shutil.copy2(seqinfo_src, seqinfo_dst)
        
        # Copy tracker result
        tracker_file_dst = os.path.join(tracker_dst, f"{seq_name}.txt")
        shutil.copy2(result_file, tracker_file_dst)
        
        sequences.append(seq_name)
    
    # Write seqmap file
    with open(seqmap_file, 'w') as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")
    
    return gt_folder, trackers_folder, benchmark_name, tracker_name


def evaluate_with_trackeval(dataset, split, results_dir, tracker_name="FastTracker_public", metrics=['HOTA', 'CLEAR', 'Identity']):
    """Evaluate using TrackEval."""
    if not TRACKEVAL_AVAILABLE:
        return None
    
    # Setup directory structure
    gt_folder, trackers_folder, benchmark_name, tracker_name = setup_trackeval_structure(
        dataset, split, results_dir, tracker_name
    )
    
    # Configure dataset
    dataset_config = MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = gt_folder
    dataset_config['TRACKERS_FOLDER'] = trackers_folder
    dataset_config['BENCHMARK'] = dataset
    dataset_config['SPLIT_TO_EVAL'] = split
    dataset_config['TRACKERS_TO_EVAL'] = [tracker_name]
    dataset_config['CLASSES_TO_EVAL'] = ['person']  # Use 'person' not 'pedestrian' (matches modified TrackEval)
    dataset_config['DO_PREPROC'] = True
    dataset_config['SKIP_SPLIT_FOL'] = False
    
    # Configure evaluator
    eval_config = Evaluator.get_default_eval_config()
    eval_config['USE_PARALLEL'] = False
    eval_config['NUM_PARALLEL_CORES'] = 1
    eval_config['PRINT_RESULTS'] = False
    eval_config['TIME_PROGRESS'] = True
    
    # Create dataset and metrics
    dataset_obj = MotChallenge2DBox(dataset_config)
    
    metrics_list = []
    metrics_config = {'THRESHOLD': 0.5}
    
    if 'HOTA' in metrics:
        metrics_list.append(HOTA(metrics_config))
    if 'CLEAR' in metrics:
        metrics_list.append(TrackEvalCLEAR(metrics_config))
    if 'Identity' in metrics:
        metrics_list.append(Identity(metrics_config))
    
    if not metrics_list:
        return None
    
    # Run evaluation
    evaluator = Evaluator(eval_config)
    output_res, output_msg = evaluator.evaluate([dataset_obj], metrics_list)
    
    # evaluator.evaluate() returns (output_res, output_msg) tuple
    # output_res structure: {dataset_class_name: {tracker_name: results}}
    # Get the dataset name from the dataset object (e.g., 'MotChallenge2DBox' not 'MOT17')
    dataset_key = dataset_obj.get_name()
    
    # Return both the results and the dataset key for later access
    return output_res, dataset_key


def evaluate_all(dataset, split, results_dir, output_csv=None, use_hota=True):
    """Evaluate all sequences."""
    # Get GT and tracker files
    if dataset == "MOT17":
        dataset_path = os.path.join("datasets", "MOT17", split)
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join("datasets", "mot", split)
    else:
        dataset_path = os.path.join("datasets", dataset, split)
    
    # Get result files
    result_files = sorted(glob.glob(os.path.join(results_dir, "*.txt")))
    if not result_files:
        print(f"Error: No result files found in {results_dir}")
        return
    
    # Find corresponding GT files
    gt_files = []
    valid_result_files = []
    
    for result_file in result_files:
        seq_name = os.path.splitext(os.path.basename(result_file))[0]
        gt_file = os.path.join(dataset_path, seq_name, "gt", "gt.txt")
        if os.path.exists(gt_file):
            gt_files.append(gt_file)
            valid_result_files.append(result_file)
        else:
            print(f"Warning: GT file not found for {seq_name}: {gt_file}")
    
    if not gt_files:
        print("Error: No matching GT files found")
        return
    
    print(f"Found {len(gt_files)} sequences to evaluate")
    print(f"Dataset: {dataset}, Split: {split}")
    print("=" * 80)
    
    # Evaluate with motmetrics (CLEAR metrics)
    print("\nEvaluating with motmetrics (CLEAR metrics)...")
    clear_summary, gt_dict, ts_dict = evaluate_with_motmetrics(gt_files, valid_result_files, dataset)
    
    # Print CLEAR metrics using motmetrics render_summary (same as track.py)
    print("\n" + "=" * 80)
    print("CLEAR Metrics (MOTMetrics)")
    print("=" * 80)
    
    # Use motmetrics formatters to print summary (same as track.py)
    mh = mm.metrics.create()
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        if k in fmt:
            fmt[k] = fmt['mota']
    
    # Print using motmetrics render_summary (same as track.py)
    print(mm.io.render_summary(clear_summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))
    
    # Evaluate with TrackEval (HOTA, Identity)
    trackeval_results = None
    dataset_key = None
    if use_hota and TRACKEVAL_AVAILABLE:
        print("\n" + "=" * 80)
        print("Evaluating with TrackEval (HOTA, Identity metrics)...")
        print("=" * 80)
        
        try:
            trackeval_result = evaluate_with_trackeval(
                dataset, split, results_dir, 
                tracker_name="FastTracker_public",
                metrics=['HOTA', 'Identity']
            )
            
            if trackeval_result:
                trackeval_results, dataset_key = trackeval_result
                
                # Extract results
                print("\nHOTA Metrics:")
                print("-" * 80)
                print(f"{'Metric':<25} {'Value':>10}")
                print("-" * 80)
                
                # TrackEval results structure: output_res[dataset_class_name][tracker_name]['COMBINED_SEQ'][class][metric_name]
                try:
                    dataset_results = trackeval_results[dataset_key]
                    tracker_results = dataset_results['FastTracker_public']
                    # Access COMBINED_SEQ results: res['COMBINED_SEQ'][cls][metric_name]
                    combined_results = tracker_results['COMBINED_SEQ']
                    # Use 'person' instead of 'pedestrian' (matches modified TrackEval)
                    person_results = combined_results['person']
                    
                    # HOTA metrics
                    if 'HOTA' in person_results:
                        hota_results = person_results['HOTA']
                        hota_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']
                        for field in hota_fields:
                            if field in hota_results:
                                value = hota_results[field]
                                if isinstance(value, (list, np.ndarray)):
                                    value = value[0] if len(value) > 0 else 0.0
                                print(f"{field:<25} {value:>10.4f}")
                    
                    # Identity metrics
                    if 'Identity' in person_results:
                        print("\nIdentity Metrics:")
                        print("-" * 80)
                        print(f"{'Metric':<25} {'Value':>10}")
                        print("-" * 80)
                        identity_results = person_results['Identity']
                        identity_fields = ['IDF1', 'IDP', 'IDR']
                        for field in identity_fields:
                            if field in identity_results:
                                value = identity_results[field]
                                if isinstance(value, (list, np.ndarray)):
                                    value = value[0] if len(value) > 0 else 0.0
                                print(f"{field:<25} {value:>10.4f}")
                
                except (KeyError, TypeError) as e:
                    print(f"Warning: Could not extract TrackEval results: {e}")
                    if isinstance(trackeval_results, dict):
                        print("TrackEval results keys:", list(trackeval_results.keys()))
                        if dataset_key in trackeval_results:
                            dataset_keys = list(trackeval_results[dataset_key].keys())
                            print(f"Dataset '{dataset_key}' keys:", dataset_keys)
                            if 'FastTracker_public' in trackeval_results[dataset_key]:
                                tracker_keys = list(trackeval_results[dataset_key]['FastTracker_public'].keys())
                                print(f"Tracker 'FastTracker_public' keys:", tracker_keys)
                    else:
                        print("TrackEval results type:", type(trackeval_results))
        except Exception as e:
            print(f"Warning: TrackEval evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save to CSV if requested
    if output_csv:
        save_to_csv(clear_summary, trackeval_results, output_csv, dataset_key)
        print(f"\nResults saved to: {output_csv}")


def save_to_csv(clear_summary, trackeval_results, output_file, dataset_key):
    """Save results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Metric Family', 'Metric', 'Value'])
        
        # CLEAR metrics - use last row for overall (when generate_overall=True, last row is overall)
        clear_metrics = ['mota', 'motp', 'idf1', 'num_switches', 'num_fragmentations',
                        'num_false_positives', 'num_misses', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
        
        # Get overall values from last row of summary (last row is overall when generate_overall=True)
        if len(clear_summary) > 0:
            for metric in clear_metrics:
                if metric in clear_summary.columns:
                    try:
                        # Use iloc to get last row directly (this is the overall summary)
                        value = clear_summary.iloc[-1][metric]
                        writer.writerow(['CLEAR', metric, float(value)])
                    except (KeyError, IndexError, ValueError) as e:
                        # Skip if metric not found
                        continue
        
        # HOTA and Identity metrics from TrackEval
        if trackeval_results and dataset_key:
            try:
                dataset_results = trackeval_results[dataset_key]
                tracker_results = dataset_results['FastTracker_public']
                # Access COMBINED_SEQ results: res['COMBINED_SEQ'][cls][metric_name]
                combined_results = tracker_results['COMBINED_SEQ']
                # Use 'person' instead of 'pedestrian' (matches modified TrackEval)
                person_results = combined_results['person']
                
                # HOTA metrics
                if 'HOTA' in person_results:
                    hota_results = person_results['HOTA']
                    hota_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']
                    for field in hota_fields:
                        if field in hota_results:
                            value = hota_results[field]
                            if isinstance(value, (list, np.ndarray)):
                                value = value[0] if len(value) > 0 else 0.0
                            writer.writerow(['HOTA', field, value])
                
                # Identity metrics
                if 'Identity' in person_results:
                    identity_results = person_results['Identity']
                    identity_fields = ['IDF1', 'IDP', 'IDR']
                    for field in identity_fields:
                        if field in identity_results:
                            value = identity_results[field]
                            if isinstance(value, (list, np.ndarray)):
                                value = value[0] if len(value) > 0 else 0.0
                            writer.writerow(['Identity', field, value])
            except (KeyError, TypeError) as e:
                # Skip if structure doesn't match (better error handling)
                print(f"Warning: Could not extract TrackEval results for CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tracking results with HOTA, CLEAR, and Identity metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MOT17 train results
  python tools/eval_results.py --dataset MOT17 --split train --results_dir track_results_public
  
  # Save results to CSV
  python tools/eval_results.py --dataset MOT17 --split train --results_dir track_results_public --output_csv results.csv
  
  # Evaluate without HOTA (CLEAR metrics only)
  python tools/eval_results.py --dataset MOT17 --split train --results_dir track_results_public --no_hota
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="MOT17",
        choices=["MOT17", "MOT20"],
        help="Dataset name (default: MOT17)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="track_results_public",
        help="Directory containing tracking results (default: track_results_public)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file (optional)"
    )
    parser.add_argument(
        "--no_hota",
        action="store_true",
        help="Disable HOTA evaluation (CLEAR metrics only)"
    )
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Run evaluation
    evaluate_all(
        args.dataset, 
        args.split, 
        args.results_dir, 
        args.output_csv,
        use_hota=not args.no_hota
    )


if __name__ == "__main__":
    main()
