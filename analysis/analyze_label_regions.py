"""
Label Region Analysis - US Signal Discriminability

Computes statistics per label region and aggregates across:
- US channels (individual + combined)
- Experiments, Sessions, Participants, Global

Usage:
    python analysis/analyze_label_regions.py
    python analysis/analyze_label_regions.py --config config/config.yaml
    python analysis/analyze_label_regions.py --max-experiments 10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from preprocessing.processor import DataProcessor

LABEL_NAMES = {0: 'Noise', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}
SSIM_MIN_WIDTH = 20  # Minimum width for SSIM computation


def compute_ssim(img1, img2):
    """Compute SSIM between two same-sized images."""
    # Normalize to [0, 1] for SSIM
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 0:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    img1_norm = normalize(img1)
    img2_norm = normalize(img2)

    # Use smaller win_size if image is small
    win_size = min(7, min(img1.shape) - 1)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    try:
        from skimage.metrics import structural_similarity as ssim_func
        return ssim_func(img1_norm, img2_norm, win_size=win_size, data_range=1.0)
    except Exception:
        return np.nan


def find_label_regions(labels):
    """Find start and end indices for each labeled region."""
    regions = {i: [] for i in range(5)}

    for label_val in range(5):
        mask = labels == label_val
        diff = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        for s, e in zip(starts, ends):
            regions[label_val].append((s, e))

    return regions


def compute_region_stats(us_patch):
    """
    Compute statistics for a US region patch.

    Args:
        us_patch: [depth, width] array for single channel

    Returns:
        dict with mean, std, range
    """
    flat = us_patch.flatten()
    return {
        'mean': np.mean(flat),
        'std': np.std(flat),
        'range': np.ptp(flat)  # max - min
    }


def extract_region_features(processed_us, labels, min_region_length=10):
    """
    Extract features from all label regions in an experiment.

    Args:
        processed_us: [C, depth, pulses] array
        labels: [pulses] array
        min_region_length: skip regions shorter than this

    Returns:
        tuple: (list of feature dicts, list of cropped patches for SSIM)
    """
    regions = find_label_regions(labels)
    n_channels = processed_us.shape[0]
    features = []
    ssim_patches = []  # Store cropped patches for SSIM computation

    for label_val, region_list in regions.items():
        for region_idx, (start, end) in enumerate(region_list):
            width = end - start + 1
            if width < min_region_length:
                continue

            for ch in range(n_channels):
                us_patch = processed_us[ch, :, start:end+1]
                stats = compute_region_stats(us_patch)

                feat_idx = len(features)
                features.append({
                    'label': label_val,
                    'label_name': LABEL_NAMES[label_val],
                    'channel': ch,
                    'region_idx': region_idx,
                    'n_samples': width,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'range': stats['range'],
                    'ssim': np.nan  # Placeholder, filled later
                })

                # Store center-cropped patch for SSIM if wide enough
                if width >= SSIM_MIN_WIDTH:
                    crop_start = (width - SSIM_MIN_WIDTH) // 2
                    cropped = us_patch[:, crop_start:crop_start + SSIM_MIN_WIDTH]
                    ssim_patches.append({
                        'feat_idx': feat_idx,
                        'label': label_val,
                        'channel': ch,
                        'patch': cropped
                    })

    return features, ssim_patches


def parse_experiment_path(exp_path):
    """Extract participant, session, experiment IDs from path."""
    import re
    match = re.search(r"P(\d+)/session(\d+)/exp(\d+)", exp_path)
    if match:
        return {
            'participant': int(match.group(1)),
            'session': int(match.group(2)),
            'experiment': int(match.group(3))
        }
    return {'participant': -1, 'session': -1, 'experiment': -1}


def build_class_templates(all_ssim_patches):
    """
    Build mean templates per class per channel from cropped patches.

    Returns:
        dict: {(label, channel): mean_template_array}
    """
    from collections import defaultdict

    # Group patches by (label, channel)
    grouped = defaultdict(list)
    for p in all_ssim_patches:
        key = (p['label'], p['channel'])
        grouped[key].append(p['patch'])

    # Compute mean template for each group
    templates = {}
    for key, patches in grouped.items():
        templates[key] = np.mean(patches, axis=0)

    return templates


def compute_ssim_scores(all_features, all_ssim_patches, templates):
    """
    Compute SSIM for each patch against its class template.
    Updates the 'ssim' field in all_features in place.
    """
    for p in all_ssim_patches:
        key = (p['label'], p['channel'])
        if key in templates:
            ssim_val = compute_ssim(p['patch'], templates[key])
            all_features[p['feat_idx']]['ssim'] = ssim_val


def collect_all_features(processor, max_experiments=None):
    """
    Process all experiments and collect region features.
    Two-pass approach: collect patches, build templates, compute SSIM.

    Returns:
        DataFrame with all region-level features including SSIM
    """
    exp_paths = processor.get_experiment_paths()

    if max_experiments:
        exp_paths = exp_paths[:max_experiments]

    all_features = []
    all_ssim_patches = []

    # Pass 1: Collect features and patches
    for exp_path in tqdm(exp_paths, desc="Collecting regions"):
        try:
            data = processor.process_single_experiment(exp_path)
            ids = parse_experiment_path(exp_path)

            features, ssim_patches = extract_region_features(
                data['processed_us'],
                data['labels'],
                min_region_length=10
            )

            # Adjust patch indices to global index
            offset = len(all_features)
            for p in ssim_patches:
                p['feat_idx'] += offset

            for f in features:
                f.update(ids)

            all_features.extend(features)
            all_ssim_patches.extend(ssim_patches)

        except Exception as e:
            print(f"Error processing {exp_path}: {e}")
            continue

    # Pass 2: Build templates and compute SSIM
    if all_ssim_patches:
        print(f"\nBuilding templates from {len(all_ssim_patches)} patches...")
        templates = build_class_templates(all_ssim_patches)
        print(f"Computing SSIM scores...")
        compute_ssim_scores(all_features, all_ssim_patches, templates)

        # Report SSIM coverage
        ssim_count = sum(1 for f in all_features if not np.isnan(f['ssim']))
        print(f"SSIM computed for {ssim_count}/{len(all_features)} regions (width >= {SSIM_MIN_WIDTH})")

    return pd.DataFrame(all_features)


def add_combined_channels(df):
    """Add rows for combined (averaged) channel statistics."""
    # Group by everything except channel, average the stats
    group_cols = ['participant', 'session', 'experiment', 'label', 'label_name', 'region_idx', 'n_samples']
    stat_cols = ['mean', 'std', 'range']

    combined = df.groupby(group_cols, as_index=False)[stat_cols].mean()
    combined['channel'] = 'all'

    # Convert channel to string for concatenation
    df = df.copy()
    df['channel'] = df['channel'].astype(str)

    return pd.concat([df, combined], ignore_index=True)


def aggregate_by_experiment(df):
    """Level 1: Aggregate per experiment × label × channel."""
    group_cols = ['participant', 'session', 'experiment', 'label', 'label_name', 'channel']

    agg = df.groupby(group_cols, as_index=False).agg({
        'mean': ['mean', 'std'],
        'std': ['mean', 'std'],
        'range': ['mean', 'std'],
        'ssim': ['mean', 'count'],  # mean SSIM + count of valid SSIM values
        'region_idx': 'count'
    })

    # Flatten column names
    agg.columns = group_cols + [
        'mean_mean', 'mean_std',
        'std_mean', 'std_std',
        'range_mean', 'range_std',
        'ssim_mean', 'ssim_n',
        'region_count'
    ]

    return agg


def aggregate_by_session(df):
    """Level 2: Aggregate per session × label × channel."""
    group_cols = ['participant', 'session', 'label', 'label_name', 'channel']

    agg = df.groupby(group_cols, as_index=False).agg({
        'mean': ['mean', 'std'],
        'std': ['mean', 'std'],
        'range': ['mean', 'std'],
        'ssim': ['mean', 'count'],
        'experiment': 'nunique',
        'region_idx': 'count'
    })

    agg.columns = group_cols + [
        'mean_mean', 'mean_std',
        'std_mean', 'std_std',
        'range_mean', 'range_std',
        'ssim_mean', 'ssim_n',
        'experiment_count', 'region_count'
    ]

    return agg


def aggregate_by_participant(df):
    """Level 3: Aggregate per participant × label × channel."""
    group_cols = ['participant', 'label', 'label_name', 'channel']

    agg = df.groupby(group_cols, as_index=False).agg({
        'mean': ['mean', 'std'],
        'std': ['mean', 'std'],
        'range': ['mean', 'std'],
        'ssim': ['mean', 'count'],
        'session': 'nunique',
        'region_idx': 'count'
    })

    agg.columns = group_cols + [
        'mean_mean', 'mean_std',
        'std_mean', 'std_std',
        'range_mean', 'range_std',
        'ssim_mean', 'ssim_n',
        'session_count', 'region_count'
    ]

    return agg


def aggregate_global(df):
    """Level 4: Aggregate globally per label × channel."""
    group_cols = ['label', 'label_name', 'channel']

    agg = df.groupby(group_cols, as_index=False).agg({
        'mean': ['mean', 'std'],
        'std': ['mean', 'std'],
        'range': ['mean', 'std'],
        'ssim': ['mean', 'count'],
        'participant': 'nunique',
        'region_idx': 'count'
    })

    agg.columns = group_cols + [
        'mean_mean', 'mean_std',
        'std_mean', 'std_std',
        'range_mean', 'range_std',
        'ssim_mean', 'ssim_n',
        'participant_count', 'region_count'
    ]

    return agg


def compute_statistical_tests(df):
    """
    Compute Kruskal-Wallis test for each statistic.
    Tests whether distributions differ across label classes.

    Returns:
        DataFrame with test results per statistic per channel
    """
    results = []
    stat_cols = ['mean', 'std', 'range', 'ssim']
    channels = df['channel'].unique()

    for channel in channels:
        df_ch = df[df['channel'] == channel]

        for stat in stat_cols:
            # Group values by label, filter out NaN
            groups = []
            for l in range(5):
                vals = df_ch[df_ch['label'] == l][stat].dropna().values
                if len(vals) > 0:
                    groups.append(vals)

            if len(groups) >= 2:
                h_stat, p_value = scipy_stats.kruskal(*groups)
            else:
                h_stat, p_value = np.nan, np.nan

            results.append({
                'channel': channel,
                'statistic': stat,
                'H_statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < 0.05 if not np.isnan(p_value) else False
            })

    return pd.DataFrame(results)


def compute_pairwise_effect_sizes(df):
    """
    Compute Cohen's d between each pair of classes.

    Returns:
        DataFrame with effect sizes
    """
    results = []
    stat_cols = ['mean', 'std', 'range', 'ssim']
    channels = df['channel'].unique()
    labels = sorted(df['label'].unique())

    for channel in channels:
        df_ch = df[df['channel'] == channel]

        for stat in stat_cols:
            for i, l1 in enumerate(labels):
                for l2 in labels[i+1:]:
                    g1 = df_ch[df_ch['label'] == l1][stat].dropna().values
                    g2 = df_ch[df_ch['label'] == l2][stat].dropna().values

                    if len(g1) > 1 and len(g2) > 1:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(g1)-1)*np.var(g1) + (len(g2)-1)*np.var(g2)) / (len(g1)+len(g2)-2))
                        if pooled_std > 0:
                            d = (np.mean(g1) - np.mean(g2)) / pooled_std
                        else:
                            d = 0
                    else:
                        d = np.nan

                    results.append({
                        'channel': channel,
                        'statistic': stat,
                        'class_1': LABEL_NAMES[l1],
                        'class_2': LABEL_NAMES[l2],
                        'cohens_d': d,
                        'effect_size': 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small' if not np.isnan(d) else 'N/A'
                    })

    return pd.DataFrame(results)


def create_boxplots(df, output_dir):
    """Create box plots for visual comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Global boxplot (combined channels) - separate plots for basic stats and SSIM
    df_combined = df[df['channel'] == 'all'].copy()

    # Plot 1: Basic statistics (mean, std, range)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, stat in enumerate(['mean', 'std', 'range']):
        df_stat = df_combined.dropna(subset=[stat])
        if not df_stat.empty:
            sns.boxplot(data=df_stat, x='label_name', y=stat, ax=axes[i])
        axes[i].set_title(f'{stat.upper()} by Class')
        axes[i].set_xlabel('Class')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_global_basic_stats.png'), dpi=150)
    plt.close()

    # Plot 2: SSIM (separate since many NaN values)
    df_ssim = df_combined.dropna(subset=['ssim'])
    if not df_ssim.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_ssim, x='label_name', y='ssim', ax=ax)
        ax.set_title('SSIM by Class (All Channels)')
        ax.set_xlabel('Class')
        ax.set_ylabel('SSIM')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplot_global_ssim.png'), dpi=150)
        plt.close()
    else:
        print("No SSIM data available for global boxplot")

    # Per-channel boxplots (basic stats only)
    channels = sorted([c for c in df['channel'].unique() if c != 'all'])

    if channels:
        for stat in ['mean', 'std', 'range']:
            fig, axes = plt.subplots(1, len(channels), figsize=(5*len(channels), 5))
            if len(channels) == 1:
                axes = [axes]

            for i, ch in enumerate(channels):
                df_ch = df[df['channel'] == ch].dropna(subset=[stat])
                if not df_ch.empty:
                    sns.boxplot(data=df_ch, x='label_name', y=stat, ax=axes[i])
                axes[i].set_title(f'Channel {ch}')
                axes[i].set_xlabel('Class')
                axes[i].tick_params(axis='x', rotation=45)

            fig.suptitle(f'{stat.upper()} by Class per Channel', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'boxplot_{stat}_per_channel.png'), dpi=150)
            plt.close()

        # SSIM per channel
        fig, axes = plt.subplots(1, len(channels), figsize=(5*len(channels), 5))
        if len(channels) == 1:
            axes = [axes]

        for i, ch in enumerate(channels):
            df_ch = df[df['channel'] == ch].dropna(subset=['ssim'])
            if not df_ch.empty:
                sns.boxplot(data=df_ch, x='label_name', y='ssim', ax=axes[i])
            axes[i].set_title(f'Channel {ch}')
            axes[i].set_xlabel('Class')
            axes[i].tick_params(axis='x', rotation=45)

        fig.suptitle('SSIM by Class per Channel', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplot_ssim_per_channel.png'), dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze label regions in US data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--max-experiments', type=int, default=None,
                        help='Limit number of experiments to process')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    # Resolve paths
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(project_root, 'analysis', 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")

    # Initialize processor
    processor = DataProcessor(config_file=config_path, auto_run=False)

    # Collect features
    print("\n=== Collecting region features ===")
    df_raw = collect_all_features(processor, max_experiments=args.max_experiments)

    if df_raw.empty:
        print("No features collected. Check experiment paths.")
        return

    print(f"Collected {len(df_raw)} region measurements")
    print(f"  Participants: {df_raw['participant'].nunique()}")
    print(f"  Sessions: {df_raw['session'].nunique()}")
    print(f"  Experiments: {df_raw['experiment'].nunique()}")

    # Add combined channels
    df_raw = add_combined_channels(df_raw)

    # Save raw data
    df_raw.to_csv(os.path.join(output_dir, 'raw_region_features.csv'), index=False)
    print(f"\nSaved: raw_region_features.csv")

    # Aggregations
    print("\n=== Computing aggregations ===")

    df_by_exp = aggregate_by_experiment(df_raw)
    df_by_exp.to_csv(os.path.join(output_dir, 'summary_by_experiment.csv'), index=False)
    print(f"Saved: summary_by_experiment.csv ({len(df_by_exp)} rows)")

    df_by_session = aggregate_by_session(df_raw)
    df_by_session.to_csv(os.path.join(output_dir, 'summary_by_session.csv'), index=False)
    print(f"Saved: summary_by_session.csv ({len(df_by_session)} rows)")

    df_by_participant = aggregate_by_participant(df_raw)
    df_by_participant.to_csv(os.path.join(output_dir, 'summary_by_participant.csv'), index=False)
    print(f"Saved: summary_by_participant.csv ({len(df_by_participant)} rows)")

    df_global = aggregate_global(df_raw)
    df_global.to_csv(os.path.join(output_dir, 'summary_global.csv'), index=False)
    print(f"Saved: summary_global.csv ({len(df_global)} rows)")

    # Statistical tests
    print("\n=== Statistical tests ===")
    df_tests = compute_statistical_tests(df_raw)
    df_tests.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
    print(f"Saved: statistical_tests.csv")

    print("\nKruskal-Wallis test results (H0: all classes have same distribution):")
    for _, row in df_tests.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {row['channel']:>4} | {row['statistic']:>5} | H={row['H_statistic']:8.2f} | p={row['p_value']:.2e} {sig}")

    # Effect sizes
    df_effects = compute_pairwise_effect_sizes(df_raw)
    df_effects.to_csv(os.path.join(output_dir, 'effect_sizes.csv'), index=False)
    print(f"\nSaved: effect_sizes.csv")

    # Show large effect sizes
    large_effects = df_effects[df_effects['effect_size'] == 'large']
    if not large_effects.empty:
        print("\nLarge effect sizes (|d| > 0.8):")
        for _, row in large_effects.iterrows():
            print(f"  {row['channel']:>4} | {row['statistic']:>5} | {row['class_1']:>5} vs {row['class_2']:<5} | d={row['cohens_d']:+.2f}")

    # Create plots
    print("\n=== Creating plots ===")
    plot_dir = os.path.join(output_dir, 'plots')
    create_boxplots(df_raw, plot_dir)

    # Print global summary
    print("\n=== Global Summary (Combined Channels) ===")
    df_global_combined = df_global[df_global['channel'] == 'all']
    print(df_global_combined[['label_name', 'mean_mean', 'std_mean', 'range_mean', 'ssim_mean', 'ssim_n', 'region_count']].to_string(index=False))

    print(f"\n=== Done ===")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()