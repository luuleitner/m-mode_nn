#!/bin/bash
#
# Two-Stage Classifier Pipeline
#
# This script runs:
#   1. Extract embeddings from trained autoencoder
#   2. Train two-stage classifier on embeddings
#
# Usage:
#   ./scripts/run_stage2_CLS2stage_pipeline.sh --config config/config.yaml --checkpoint path/to/model.pth
#   ./scripts/run_stage2_CLS2stage_pipeline.sh --config config/config.yaml --checkpoint path/to/model.pth --target-recall 0.90
#   ./scripts/run_stage2_CLS2stage_pipeline.sh --config config/config.yaml --checkpoint path/to/model.pth --tune
#

set -e  # Exit on error

# Change to project root (script can be called from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
CONFIG="/home/cleitner/code/lab/projects/ML/m-mode_nn/config/config.yaml"
CHECKPOINT=""
TARGET_RECALL="0.7"
TUNE=""
NORMALIZE=""
N_TRIALS=""
SKIP_EXTRACTION="true"
EMBEDDINGS_FILE="/vol/data/2026_wristus_wiicontroller_sgambato/day002/processed/Dataset_Envelope_CNN/Window10_Stride02_Labels_soft/latest/checkpoints/training_20260129_215547/embeddings/embeddings_latest.npz"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint|-ckpt)
            CHECKPOINT="$2"
            shift 2
            ;;
        --target-recall)
            TARGET_RECALL="$2"
            shift 2
            ;;
        --tune|-t)
            TUNE="--tune"
            shift
            ;;
        --normalize|-n)
            NORMALIZE="--normalize"
            shift
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --skip-extraction|-s)
            SKIP_EXTRACTION="true"
            shift
            ;;
        --embeddings|-e)
            EMBEDDINGS_FILE="$2"
            SKIP_EXTRACTION="true"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --config <config.yaml> [--checkpoint <model.pth>] [options]"
            echo ""
            echo "Required:"
            echo "  --config, -c        Path to config YAML file"
            echo "  --checkpoint, -ckpt Path to model checkpoint (required unless --embeddings provided)"
            echo ""
            echo "Optional:"
            echo "  --target-recall     Target detection recall (default: 0.7)"
            echo "  --tune, -t          Run hyperparameter tuning with Optuna"
            echo "  --n-trials          Number of Optuna trials (default: 30)"
            echo "  --normalize, -n     Normalize embeddings with StandardScaler"
            echo "  --skip-extraction, -s  Skip embedding extraction (requires --embeddings)"
            echo "  --embeddings, -e    Path to existing embeddings file (skips extraction)"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Checkpoint only required if NOT skipping extraction
if [[ "$SKIP_EXTRACTION" != "true" ]]; then
    if [[ -z "$CHECKPOINT" ]]; then
        echo "Error: --checkpoint is required (unless using --skip-extraction or --embeddings)"
        exit 1
    fi

    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Error: Checkpoint file not found: $CHECKPOINT"
        exit 1
    fi
fi

# Get directory of checkpoint for output (or embeddings file if skipping)
if [[ -n "$CHECKPOINT" ]]; then
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
else
    CHECKPOINT_DIR=$(dirname "$EMBEDDINGS_FILE")
fi
EMBEDDINGS_DIR="${CHECKPOINT_DIR}/embeddings"

echo "========================================================================"
echo "TWO-STAGE CLASSIFIER PIPELINE"
echo "========================================================================"
echo "Config:        $CONFIG"
echo "Checkpoint:    $CHECKPOINT"
echo "Target Recall: $TARGET_RECALL"
echo "Tune:          ${TUNE:-No}"
echo "Normalize:     ${NORMALIZE:-No}"
echo "========================================================================"
echo ""

# Step 1: Extract embeddings (or skip if requested)
if [[ "$SKIP_EXTRACTION" == "true" ]]; then
    echo "========================================================================"
    echo "STEP 1: Skipping embedding extraction (using existing)"
    echo "========================================================================"

    # If no embeddings file specified, use default location
    if [[ -z "$EMBEDDINGS_FILE" ]]; then
        EMBEDDINGS_FILE="${EMBEDDINGS_DIR}/embeddings_latest.npz"
    fi

    if [[ ! -f "$EMBEDDINGS_FILE" ]]; then
        echo "Error: Embeddings file not found: $EMBEDDINGS_FILE"
        exit 1
    fi

    echo "Using: $EMBEDDINGS_FILE"
else
    echo "========================================================================"
    echo "STEP 1: Extracting embeddings..."
    echo "========================================================================"

    python -m src.training.extract_embeddings \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        $NORMALIZE

    EMBEDDINGS_FILE="${EMBEDDINGS_DIR}/embeddings_latest.npz"

    if [[ ! -f "$EMBEDDINGS_FILE" ]]; then
        echo "Error: Embeddings file not found: $EMBEDDINGS_FILE"
        exit 1
    fi

    echo "Embeddings saved to: $EMBEDDINGS_FILE"
fi
echo ""

# Step 2: Train two-stage classifier
echo "========================================================================"
echo "STEP 2: Training two-stage classifier..."
echo "========================================================================"

TUNE_ARGS=""
if [[ -n "$TUNE" ]]; then
    TUNE_ARGS="--tune --n-trials $N_TRIALS"
fi

python -m src.training.train_two_stage \
    --config "$CONFIG" \
    --embeddings "$EMBEDDINGS_FILE" \
    --target-recall "$TARGET_RECALL" \
    $TUNE_ARGS

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
