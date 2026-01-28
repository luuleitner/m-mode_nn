"""
Verify Label Consistency

Reads from RAW data (base_data_path/raw/) to verify label generation logic.

Compares label generation between:
1. preprocessing/label_logic/visualize.py (uses label_config.yaml)
2. visualization/data_visualization_raw-processed.py (uses label_config.yaml via processor.py)

Both should produce IDENTICAL labels since they use the same config.

Usage:
    python visualization/verify_labels.py
    python visualization/verify_labels.py --exp-path /vol/data/.../session14_W_001/10

Options:
    --exp-path    Specific experiment path to verify
"""

import os
import sys
import argparse
import numpy as np
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from preprocessing.signal_utils import apply_joystick_filters
from preprocessing.label_logic.labeling import (
    create_derivative_labels,
    create_edge_to_peak_labels,
    create_edge_to_derivative_labels
)
from preprocessing.processor import DataProcessor


def load_label_config():
    """Load unified label config from label_logic/label_config.yaml"""
    config_path = os.path.join(project_root, 'preprocessing', 'label_logic', 'label_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_labels_visualize_style(joystick_data, label_config):
    """
    Create labels the way visualize.py does it.
    Uses label_config.yaml settings.
    """
    label_axis = label_config.get('axis', 'x')
    joystick_column = 1 if label_axis == 'x' else 2
    filters_config = label_config.get('filters', {})
    label_method = label_config.get('method', 'derivative')
    threshold_percent = label_config.get('threshold_percent', 5.0)

    # Get raw data for the configured column
    raw_data = joystick_data[:, joystick_column]

    # Apply filters to position
    filtered_data = apply_joystick_filters(raw_data.copy(), filters_config, 'position')
    derivative = np.gradient(filtered_data)
    derivative = apply_joystick_filters(derivative, filters_config, 'derivative')

    # Create labels
    if label_method == 'edge_to_peak':
        labels, threshold, markers = create_edge_to_peak_labels(filtered_data, derivative, threshold_percent)
    elif label_method == 'edge_to_derivative':
        labels, pos_thresh, deriv_thresh, markers = create_edge_to_derivative_labels(filtered_data, derivative, threshold_percent)
    else:
        labels, threshold = create_derivative_labels(derivative, threshold_percent)

    return {
        'labels': labels,
        'joystick_column': joystick_column,
        'axis_name': label_axis.upper(),
        'method': label_method,
        'threshold_percent': threshold_percent,
        'filtered_data': filtered_data,
        'derivative': derivative
    }


def create_labels_processor_style(joystick_data, label_config):
    """
    Create labels the way data_visualization_raw-processed.py / processor.py does it.
    Now uses the same label_config.yaml as visualize.py.
    """
    label_axis = label_config.get('axis', 'x')
    joystick_column = 1 if label_axis == 'x' else 2
    filters_config = label_config.get('filters', {})
    label_method = label_config.get('method', 'derivative')
    threshold_percent = label_config.get('threshold_percent', 5.0)

    # Get raw data
    raw_data = joystick_data[:, joystick_column]

    # Apply filters to position
    filtered_data = apply_joystick_filters(raw_data.copy(), filters_config, 'position')
    derivative = np.gradient(filtered_data)
    derivative = apply_joystick_filters(derivative, filters_config, 'derivative')

    # Create labels
    if label_method == 'edge_to_peak':
        labels, threshold, markers = create_edge_to_peak_labels(filtered_data, derivative, threshold_percent)
    elif label_method == 'edge_to_derivative':
        labels, pos_thresh, deriv_thresh, markers = create_edge_to_derivative_labels(filtered_data, derivative, threshold_percent)
    else:
        labels, threshold = create_derivative_labels(derivative, threshold_percent)

    return {
        'labels': labels,
        'joystick_column': joystick_column,
        'axis_name': label_axis.upper(),
        'method': label_method,
        'threshold_percent': threshold_percent,
        'filtered_data': filtered_data,
        'derivative': derivative
    }


def compare_labels(labels1, labels2, name1, name2):
    """Compare two label arrays and print differences."""
    print(f"\n{'='*60}")
    print(f"LABEL COMPARISON: {name1} vs {name2}")
    print(f"{'='*60}")

    print(f"\nArray lengths: {len(labels1)} vs {len(labels2)}")

    if len(labels1) != len(labels2):
        print("WARNING: Label arrays have different lengths!")
        min_len = min(len(labels1), len(labels2))
        labels1 = labels1[:min_len]
        labels2 = labels2[:min_len]
        print(f"Comparing first {min_len} samples only")

    # Check if identical
    identical = np.array_equal(labels1, labels2)
    print(f"\nLabels identical: {identical}")

    if not identical:
        diff_mask = labels1 != labels2
        diff_count = np.sum(diff_mask)
        diff_pct = 100 * diff_count / len(labels1)
        print(f"Differences: {diff_count} samples ({diff_pct:.2f}%)")

        # Show first few differences
        diff_indices = np.where(diff_mask)[0]
        print(f"\nFirst 10 differences (index: {name1} vs {name2}):")
        for idx in diff_indices[:10]:
            print(f"  [{idx}]: {labels1[idx]} vs {labels2[idx]}")

    # Distribution comparison
    print(f"\nLabel distribution:")
    print(f"  {name1}: 0={np.sum(labels1==0)}, 1={np.sum(labels1==1)}, 2={np.sum(labels1==2)}")
    print(f"  {name2}: 0={np.sum(labels2==0)}, 1={np.sum(labels2==1)}, 2={np.sum(labels2==2)}")

    return identical


def main():
    parser = argparse.ArgumentParser(description='Verify label consistency between visualizers')
    parser.add_argument('--exp-path', type=str,
                        default='/vol/data/2025_wristus_wiicontroller_leitner/raw/session14_W_001/10',
                        help='Experiment path to verify')
    args = parser.parse_args()

    exp_path = args.exp_path

    print(f"\n{'#'*60}")
    print(f"LABEL VERIFICATION")
    print(f"{'#'*60}")
    print(f"\nExperiment: {exp_path}")

    # Load joystick data
    joystick_path = os.path.join(exp_path, '_joystick.npy')
    if not os.path.exists(joystick_path):
        print(f"ERROR: Joystick file not found: {joystick_path}")
        return 1

    joystick_data = np.load(joystick_path, allow_pickle=True)
    print(f"Joystick shape: {joystick_data.shape}")

    # Load unified label config
    label_config = load_label_config()

    # Print config
    print(f"\n{'='*60}")
    print("LABEL CONFIG (label_logic/label_config.yaml)")
    print(f"{'='*60}")
    print(f"  axis: {label_config.get('axis', 'x')}")
    print(f"  method: {label_config.get('method', 'derivative')}")
    print(f"  threshold_percent: {label_config.get('threshold_percent', 5.0)}")
    print(f"  filters.position: {list(label_config.get('filters', {}).get('position', {}).keys())}")
    print(f"  filters.derivative: {list(label_config.get('filters', {}).get('derivative', {}).keys())}")

    # Create labels using both methods (should now be identical)
    result_visualize = create_labels_visualize_style(joystick_data, label_config)
    result_processor = create_labels_processor_style(joystick_data, label_config)

    print(f"\n{'='*60}")
    print("LABEL GENERATION RESULTS")
    print(f"{'='*60}")

    print(f"\nvisualize.py style:")
    print(f"  Axis: {result_visualize['axis_name']} (column {result_visualize['joystick_column']})")
    print(f"  Method: {result_visualize['method']}")
    print(f"  Threshold: {result_visualize['threshold_percent']}%")
    print(f"  Labels shape: {result_visualize['labels'].shape}")

    print(f"\ndata_visualization_raw-processed.py style:")
    print(f"  Axis: {result_processor['axis_name']} (column {result_processor['joystick_column']})")
    print(f"  Method: {result_processor['method']}")
    print(f"  Threshold: {result_processor['threshold_percent']}%")
    print(f"  Labels shape: {result_processor['labels'].shape}")

    # Compare labels
    identical = compare_labels(
        result_visualize['labels'],
        result_processor['labels'],
        f"visualize.py ({result_visualize['axis_name']})",
        f"data_visualization_raw-processed.py ({result_processor['axis_name']})"
    )

    # Summary
    print(f"\n{'#'*60}")
    print("SUMMARY")
    print(f"{'#'*60}")

    if identical:
        print(f"\nLabels are IDENTICAL between both methods.")
        print(f"Both use unified label_config.yaml - configuration is consistent.")
    else:
        print(f"\nLabels are DIFFERENT - this indicates a bug in the implementation!")
        print(f"Both methods should produce identical results using label_config.yaml.")

    return 0 if identical else 1


if __name__ == '__main__':
    exit(main())
