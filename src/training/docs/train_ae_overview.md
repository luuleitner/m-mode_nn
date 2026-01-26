# train_ae.py - High Level Overview

## Document Information

| **Field** | **Value** |
|-----------|-----------|
| **Version** | 1.1 |
| **Last Updated** | 2025-10-08 |
| **Status** | ✅ Current |
| **Document Type** | High-Level Architecture Overview |
| **Companion Doc** | train_ae_flow.md (detailed flow) |

## Change Log

### Version 1.1 (2025-10-08)
- ✅ Added versioning and errata tracking
- ✅ Updated model references to CNNAutoencoder
- ✅ Enhanced component descriptions
- ✅ Added sync status tracking

### Version 1.0 (2025-10-08)
- 🔄 Initial creation with big picture flow
- ✅ Core components overview
- ✅ Decision points table
- ✅ Execution paths

## Known Issues / TODO

- 📝 Consider adding data flow volume estimates
- 🔄 Memory usage patterns not yet documented
- ⚠️  WandB integration details could be expanded
- 📈 Add performance benchmarking section

## Sync Status

| **Component** | **Last Checked** | **Status** |
|---------------|------------------|------------|
| Main Flow | 2025-10-08 | ✅ Synchronized |
| CNNAutoencoder | 2025-10-08 | ✅ Synchronized |
| TrainerAE | 2025-10-08 | ✅ Synchronized |
| Config System | 2025-10-08 | ✅ Synchronized |

## Big Picture Flow

```
                              ┌─────────────────┐
                              │    CONFIG       │
                              │  ┌───────────┐  │
                              │  │ YAML File │  │
                              │  └───────────┘  │
                              └─────────┬───────┘
                                        │
                               ┌────────▼────────┐
                               │   ENVIRONMENT   │
                               │     SETUP       │
                               └────────┬────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │          DATA LOADING        │
                         │  ┌────────┐  ┌────────────┐ │
                         │  │Pickled │  │Raw Dataset│ │
                         │  │Dataset │or│Processing  │ │
                         │  └────────┘  └────────────┘ │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │       MODEL & TRAINER       │
                         │  ┌──────────────────────┐   │
                         │  │   CNNAutoencoder     │   │
                         │  │   +                  │   │
                         │  │   TrainerAE          │   │
                         │  └──────────────────────┘   │
                         └──────────────┬──────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │    TRAINING       │
                              │      MODE         │
                              └─────┬───────┬─────┘
                                   │       │
                            ┌──────▼──┐ ┌──▼──────┐
                            │ RESTART │ │  FRESH  │
                            │  MODE   │ │  START  │
                            └──────┬──┘ └──┬──────┘
                                   └─────┬─┘
                                         │
                         ┌───────────────▼───────────────┐
                         │       TRAINING LOOP           │
                         │                               │
                         │  ╔═════════════════════════╗  │
                         │  ║ EPOCHS                  ║  │
                         │  ║  ┌─────────────────────┐║  │
                         │  ║  │ Forward → Loss      │║  │
                         │  ║  │ Backward → Update   │║  │
                         │  ║  │ Validate → Save     │║  │
                         │  ║  │ Log → Visualize     │║  │
                         │  ║  └─────────────────────┘║  │
                         │  ╚═════════════════════════╝  │
                         └───────────────┬───────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    EVALUATION       │
                              │    & REPORTING      │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │      CLEANUP        │
                              │    & FINALIZE       │
                              └─────────────────────┘
```

## Core Components

### 1. **Configuration System** 🔧
- **Input**: `config.yaml`
- **Function**: Central control for all parameters
- **Key Settings**: Model architecture, training hyperparameters, paths, WandB config

### 2. **Environment Setup** 🌐
- **Function**: Initialize compute environment
- **Actions**: Device selection, path creation, logging setup, resource allocation

### 3. **Data Pipeline** 📊
- **Pickle Path**: Fast loading from cached `.pkl` files
- **Raw Processing Path**: Create datasets from raw ultrasound data
- **Output**: Train/validation/test DataLoaders

### 4. **Model & Trainer** 🧠
- **Model**: `CNNAutoencoder` - 1D CNN for ultrasound embedding
- **Trainer**: `TrainerAE` - Complete training orchestration

### 5. **Training Execution** 🚀
- **Two Modes**:
  - **Restart**: Resume from existing checkpoint
  - **Fresh**: Start training from scratch
- **Core Loop**: Forward pass → Loss computation → Backprop → Validation

### 6. **Monitoring & Outputs** 📈
- **Real-time**: WandB logging, console metrics
- **Periodic**: Training curves, reconstruction visualizations
- **Final**: Complete evaluation report with all metrics

## Key Abstractions

### Data Flow
```
Raw Ultrasound → Tokenization → Sequencing → Batching → Model → Embeddings
```

### Training Flow
```
Setup → Load Data → Initialize → Train → Evaluate → Save
```

### Error Recovery
```
Exception → Emergency Save → Recovery Info → Cleanup
```

## Decision Points

| **Decision** | **Options** | **Impact** |
|--------------|-------------|------------|
| **Data Loading** | Pickle vs Raw | Speed vs Flexibility |
| **Training Mode** | Fresh vs Restart | Clean start vs Resume |
| **Monitoring** | WandB on/off | Cloud logging vs Local |
| **Checkpointing** | Frequency | Storage vs Recovery granularity |

## File Organization

```
train_ae.py
├── Main Flow (18-226)
├── Data Helpers (233-270)
├── Hyperparameter Utils (277-328)
├── Error Handling (334-581)
├── Callback Functions (591-728)
└── CLI & Config (734-783)
```

## Typical Execution Paths

### 🆕 **First Time Training**
```
Config → Environment → Process Raw Data → Cache → Train → Evaluate
```

### 🔄 **Subsequent Training**
```
Config → Environment → Load Cache → Train → Evaluate
```

### ↩️ **Resume Training**
```
Config → Environment → Load Cache → Find Checkpoint → Resume → Evaluate
```

### ⚠️ **Recovery Scenario**
```
Training → Exception → Emergency Save → Recovery Logs → Manual Restart
```

## Success Metrics

- **Training**: Loss convergence, validation improvement
- **Quality**: Reconstruction accuracy, embedding meaningful
- **Robustness**: Error recovery, checkpoint integrity
- **Monitoring**: Complete logging, visualization quality

## Entry Points

```bash
# Standard execution
python -m training.train_ae --config config/config.yaml

# Dry run validation
python -m training.train_ae --config config/config.yaml --dry-run

# Override parameters
python -m training.train_ae --config config/config.yaml --override ml.training.epochs=50
```