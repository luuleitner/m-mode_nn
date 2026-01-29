"""
Generate a CSV selection file from available experiments.

Usage:
    python utils/generate_selection.py [--output config/preprocessing_selection.csv]

The generated CSV can be opened in Excel/LibreOffice/Google Sheets.
Delete rows or set include=0 for experiments you want to exclude.
"""

import os
import glob
import argparse
import csv
import datetime
from pathlib import Path


def find_experiments(raw_data_path):
    """
    Find all experiment folders in the raw data directory.

    Supports new hierarchy: P*/session*/exp*

    Returns:
        List of dicts with participant, session, experiment, path info
        (sorted by participant, then session, then experiment number)
    """
    experiments = []

    # Find all experiment folders (P*/session*/exp*)
    pattern = os.path.join(raw_data_path, "P*", "session*", "exp*")
    for exp_path in glob.glob(pattern):
        if os.path.isdir(exp_path):
            # Parse path components
            exp_name = os.path.basename(exp_path)  # exp000
            session_dir = os.path.dirname(exp_path)
            session_name = os.path.basename(session_dir)  # session000
            participant_dir = os.path.dirname(session_dir)
            participant_name = os.path.basename(participant_dir)  # P000

            # Extract numeric values
            participant_num = int(participant_name[1:])  # P000 -> 0
            session_num = int(session_name[7:])  # session000 -> 0
            exp_num = int(exp_name[3:])  # exp000 -> 0

            experiments.append({
                'participant': participant_num,
                'session': session_num,
                'experiment': exp_num,
                'include': 1,  # Default to include
                'path': exp_path
            })

    # Sort by participant, then session, then experiment number
    experiments.sort(key=lambda x: (x['participant'], x['session'], x['experiment']))

    return experiments


def generate_csv(experiments, output_path, include_path=False):
    """
    Generate a CSV selection file.

    Args:
        experiments: List of experiment dicts
        output_path: Path to write CSV
        include_path: Whether to include full path column (for reference)
    """
    fieldnames = ['participant', 'session', 'experiment', 'include']
    if include_path:
        fieldnames.append('path')

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in experiments:
            row = {
                'participant': exp['participant'],
                'session': exp['session'],
                'experiment': exp['experiment'],
                'include': exp['include']
            }
            if include_path:
                row['path'] = exp['path']
            writer.writerow(row)

    print(f"Generated selection file: {output_path}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Participants: {len(set(e['participant'] for e in experiments))}")
    print(f"  Sessions: {len(set((e['participant'], e['session']) for e in experiments))}")
    print(f"\nEdit this file to exclude experiments (set include=0 or delete rows)")


def main():
    parser = argparse.ArgumentParser(description="Generate preprocessing selection CSV")
    parser.add_argument('--data-path', '-d',
                       default='/vol/data/2026_wristus_wiicontroller_sgambato/raw/day001',
                       help='Path to raw data directory')
    parser.add_argument('--output', '-o',
                       default=None,
                       help='Output CSV path (default: auto-generated in data path with timestamp)')
    parser.add_argument('--include-path', '-p', action='store_true',
                       help='Include full path column in CSV')

    args = parser.parse_args()

    # Find experiments
    experiments = find_experiments(args.data_path)

    if not experiments:
        print(f"No experiments found in: {args.data_path}")
        return 1

    # Generate output path with timestamp if not specified
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.data_path, f"selection_{timestamp}.csv")
    else:
        output_path = args.output
        if not os.path.isabs(output_path):
            script_dir = Path(__file__).parent.parent
            output_path = script_dir / output_path

    # Generate CSV
    generate_csv(experiments, str(output_path), args.include_path)

    return 0


if __name__ == '__main__':
    exit(main())
