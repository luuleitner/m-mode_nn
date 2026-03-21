"""
Raster Plot: Label heatmaps across experiments (matplotlib)

Grid of rows x cols, each cell = one experiment shown as a 1D heatmap
colored by label class.

Usage:
    python preprocessing/visualization/visualize_labels_raster.py
    python preprocessing/visualization/visualize_labels_raster.py --seed 7 --rows 5 --cols 6
    python preprocessing/visualization/visualize_labels_raster.py --save raster.png
"""

import numpy as np
import os
import sys
import argparse
import yaml

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_dir)

from preprocessing.signal_utils import apply_joystick_filters
from preprocessing.label_logic.label_logic import label_movements_xy
from preprocessing.processor import DataProcessor

# Load label config
with open(os.path.join(project_dir, "preprocessing", "label_logic", "label_config.yaml")) as f:
    label_config = yaml.safe_load(f)

FILTERS = label_config.get('filters', {})
SC_DEFAULTS = label_config.get('segment_classify', {})

LABEL_NAMES = {0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}
LABEL_COLORS_RGB = {
    0: '#C8C8C8',  # Noise - gray
    1: '#00C800',  # Up - green
    2: '#C80000',  # Down - red
    3: '#0064FF',  # Left - blue
    4: '#FFA500',  # Right - orange
}

classes_cfg = label_config.get('classes', {})
if classes_cfg.get('names'):
    LABEL_NAMES = {int(k): v for k, v in classes_cfg['names'].items()}

CMAP = ListedColormap([LABEL_COLORS_RGB[i] for i in range(5)])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], CMAP.N)


def process_experiment(exp_path, config):
    """Load, filter, label one experiment."""
    joystick = np.load(os.path.join(exp_path, "_joystick.npy"), allow_pickle=True)

    x_pos = apply_joystick_filters(joystick[:, 1].copy(), FILTERS, 'position')
    y_pos = apply_joystick_filters(joystick[:, 2].copy(), FILTERS, 'position')
    x_vel = apply_joystick_filters(np.gradient(x_pos), FILTERS, 'derivative')
    y_vel = apply_joystick_filters(np.gradient(y_pos), FILTERS, 'derivative')

    labels, _, _ = label_movements_xy(x_pos, y_pos, x_vel, y_vel, config)

    parts = exp_path.rstrip('/').split('/')
    name = f"{parts[-3]}/{parts[-2][-3:]}/{parts[-1][-3:]}"

    return name, labels


def visualize_raster(exp_paths, config, rows, cols, seed):
    n_exp = len(exp_paths)

    # Process all experiments once
    results = []
    for idx, path in enumerate(exp_paths):
        name, labels = process_experiment(path, config)
        print(f"  [{idx+1}/{n_exp}] {name}")
        results.append((name, labels))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 1.2),
                              squeeze=False)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        if idx < n_exp:
            name, labels = results[idx]
            ax.imshow(labels.reshape(1, -1), aspect='auto',
                     cmap=CMAP, norm=NORM, interpolation='nearest')
            ax.set_title(name, fontsize=7, pad=2)
        else:
            ax.set_visible(False)

        ax.set_yticks([])
        if r < rows - 1:
            ax.set_xticks([])

    # Shared legend
    legend_patches = [Patch(facecolor=LABEL_COLORS_RGB[i], label=LABEL_NAMES[i])
                      for i in range(5)]
    fig.legend(handles=legend_patches, loc='lower center',
              ncol=5, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Label Raster: {n_exp} experiments (seed={seed})", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Print summary
    all_labels = np.concatenate([labels for _, labels in results])
    unique, counts = np.unique(all_labels, return_counts=True)
    total = len(all_labels)

    print(f"\n{'='*50}")
    print(f"Raster: {n_exp} experiments, seed={seed}")
    for u, c in zip(unique, counts):
        print(f"  {LABEL_NAMES.get(u, u):>6}: {c:>7} ({100*c/total:.1f}%)")
    print(f"{'='*50}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Label raster plot (matplotlib)')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--cols', type=int, default=5)
    parser.add_argument('--activity-thresh', type=float, default=None)
    parser.add_argument('--smooth-window', type=int, default=None)
    parser.add_argument('--min-duration', type=int, default=None)
    parser.add_argument('--merge-gap', type=int, default=None)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    config = dict(SC_DEFAULTS)
    if args.activity_thresh is not None:
        config['activity_threshold_percent'] = args.activity_thresh
    if args.smooth_window is not None:
        config['smooth_window'] = args.smooth_window
    if args.min_duration is not None:
        config['min_duration'] = args.min_duration
    if args.merge_gap is not None:
        config['merge_gap'] = args.merge_gap

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_dir, config_path)

    processor = DataProcessor(config_file=config_path, auto_run=False)
    all_paths = processor.get_experiment_paths()
    if not all_paths:
        raise ValueError("No experiments found")

    n = min(args.rows * args.cols, len(all_paths))
    rng = np.random.default_rng(args.seed)
    selected = list(rng.choice(all_paths, size=n, replace=False))
    selected.sort()

    print(f"Selected {n} experiments (seed={args.seed})")
    fig = visualize_raster(selected, config, args.rows, args.cols, args.seed)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to: {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    exit(main())
