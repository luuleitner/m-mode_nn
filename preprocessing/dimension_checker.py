#!/usr/bin/env python3
"""
Dimension Checker and Guide Updater
Automatically checks data dimensions and updates the data_dimensions_guide.md
"""

import os
import yaml
import datetime
import numpy as np
from pathlib import Path
import h5py
import pandas as pd
from typing import Dict, Tuple, Optional, List

import utils.logging_config as logconf
logger = logconf.get_logger("DIMENSION_CHECKER")


class DimensionChecker:
    """Checks and validates data dimensions throughout the processing pipeline"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize with configuration file"""
        self.config_path = config_path
        self.load_config()
        self.guide_path = Path(__file__).parent / 'docs' / 'data_dimensions_guide.md'
        
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def calculate_expected_dimensions(self) -> Dict:
        """Calculate expected dimensions based on configuration"""
        
        # Extract parameters from config
        preprocess = self.config['preprocess']
        
        # Signal processing parameters
        clip_flag = preprocess['signal']['clip']['apply']
        clip_samples = preprocess['signal']['clip']['samples2keep']
        decimation_flag = preprocess['signal']['decimation']['apply']
        decimation_factor = preprocess['signal']['decimation']['factor']
        envelope_flag = preprocess['signal']['envelope']['apply']
        logcomp_flag = preprocess['signal']['logcompression']['apply']
        norm_flag = preprocess['signal']['normalization']['apply']
        
        # Tokenization parameters
        token_window = preprocess['tokenization']['window']
        token_stride = preprocess['tokenization']['stride']
        startend_flag = preprocess['tokenization']['startendID']
        
        # Sequencing parameters
        sequence_window = preprocess['sequencing']['window']
        
        # Calculate dimensions step by step
        dimensions = {
            'config': {
                'clip_enabled': clip_flag,
                'clip_samples': clip_samples,
                'decimation_enabled': decimation_flag,
                'decimation_factor': decimation_factor,
                'envelope_enabled': envelope_flag,
                'logcompression_enabled': logcomp_flag,
                'normalization_enabled': norm_flag,
                'token_window': token_window,
                'token_stride': token_stride,
                'sequence_window': sequence_window,
                'startend_markers': startend_flag
            },
            'transformations': {}
        }
        
        # Typical input dimensions (example)
        channels = 3
        raw_height = 65000  # Typical raw samples
        raw_width = 5000    # Typical A-mode lines
        
        # Track transformations
        height = raw_height
        width = raw_width
        
        dimensions['transformations']['raw'] = {
            'shape': [channels, raw_height, raw_width],
            'description': 'Raw ultrasound data'
        }
        
        # After clipping
        if clip_flag:
            height = clip_samples
            dimensions['transformations']['after_clipping'] = {
                'shape': [channels, height, width],
                'description': f'Clipped to {clip_samples} samples'
            }
        
        # After decimation
        if decimation_flag:
            height = height // decimation_factor
            dimensions['transformations']['after_decimation'] = {
                'shape': [channels, height, width],
                'description': f'Decimated by factor {decimation_factor}'
            }
        
        # Calculate tokenization
        num_tokens = (width - token_window) // token_stride + 1
        token_overlap = token_window - token_stride if token_stride < token_window else 0
        
        # Add start/end markers
        token_height = height
        if startend_flag:
            token_height = height + 2
            
        dimensions['transformations']['after_tokenization'] = {
            'shape': [num_tokens, channels, token_height, token_window],
            'num_tokens': num_tokens,
            'overlap_samples': token_overlap,
            'description': f'Tokenized with window={token_window}, stride={token_stride}'
        }
        
        # Calculate sequencing
        num_sequences = num_tokens // sequence_window
        clipped_tokens = num_tokens % sequence_window
        
        dimensions['transformations']['after_sequencing'] = {
            'shape': [num_sequences, sequence_window, channels, token_height, token_window],
            'num_sequences': num_sequences,
            'clipped_tokens': clipped_tokens,
            'description': f'Sequenced with window={sequence_window}'
        }
        
        # Calculate memory usage
        total_elements = num_sequences * sequence_window * channels * token_height * token_window
        memory_bytes = total_elements * 4  # float32
        memory_mb = memory_bytes / (1024 * 1024)
        
        dimensions['memory'] = {
            'total_elements': total_elements,
            'bytes': memory_bytes,
            'megabytes': round(memory_mb, 2)
        }
        
        return dimensions
    
    def check_processed_data(self, data_path: str) -> Dict:
        """Check actual dimensions from processed data files"""
        
        actual_dims = {}
        
        # Find processed data folders
        processed_base = Path(data_path)
        
        if not processed_base.exists():
            logger.warning(f"Processed data path does not exist: {processed_base}")
            return actual_dims
        
        # Look for H5 files
        h5_files = list(processed_base.glob("**/*.h5"))
        
        if h5_files:
            # Sample first file for dimensions
            sample_file = h5_files[0]
            
            try:
                with h5py.File(sample_file, 'r') as f:
                    data_shape = f['X'].shape if 'X' in f else None
                    label_shape = f['y'].shape if 'y' in f else None
                    
                    actual_dims['sample_file'] = str(sample_file.relative_to(processed_base))
                    actual_dims['data_shape'] = list(data_shape) if data_shape else None
                    actual_dims['label_shape'] = list(label_shape) if label_shape else None
                    
                    if data_shape:
                        actual_dims['interpretation'] = {
                            'num_sequences': data_shape[0],
                            'sequence_window': data_shape[1] if len(data_shape) > 1 else None,
                            'channels': data_shape[2] if len(data_shape) > 2 else None,
                            'height': data_shape[3] if len(data_shape) > 3 else None,
                            'token_width': data_shape[4] if len(data_shape) > 4 else None
                        }
                        
            except Exception as e:
                logger.error(f"Error reading H5 file: {e}")
        
        # Check metadata
        metadata_files = list(processed_base.glob("**/metadata.csv"))
        
        if metadata_files:
            try:
                meta_df = pd.read_csv(metadata_files[0])
                actual_dims['metadata'] = {
                    'total_tokens': len(meta_df),
                    'unique_sequences': meta_df['sequence id'].nunique() if 'sequence id' in meta_df.columns else None,
                    'unique_experiments': meta_df['experiment'].nunique() if 'experiment' in meta_df.columns else None
                }
            except Exception as e:
                logger.error(f"Error reading metadata: {e}")
                
        return actual_dims
    
    def generate_guide_content(self, expected_dims: Dict, actual_dims: Optional[Dict] = None) -> str:
        """Generate markdown content for the dimensions guide"""
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# Data Dimensions Guide

## Document Information
- **Version**: Auto-generated
- **Updated**: {timestamp}
- **Generated by**: dimension_checker.py
- **Purpose**: Complete guide to understanding data dimensions throughout the processing pipeline

---

## Current Configuration

### Processing Parameters
```yaml
"""
        
        # Add config parameters
        for key, value in expected_dims['config'].items():
            content += f"{key}: {value}\n"
        
        content += """```

---

## Data Flow & Dimension Transformations

### Processing Pipeline Dimensions
"""
        
        # Add transformation steps
        for i, (stage, info) in enumerate(expected_dims['transformations'].items(), 1):
            shape_str = ' × '.join(map(str, info['shape']))
            content += f"""
### {i}. {stage.replace('_', ' ').title()}
- **Shape**: `{info['shape']}` ({shape_str})
- **Description**: {info['description']}
"""
            if 'num_tokens' in info:
                content += f"- **Number of tokens**: {info['num_tokens']}\n"
            if 'overlap_samples' in info and info['overlap_samples'] > 0:
                content += f"- **Token overlap**: {info['overlap_samples']} samples\n"
            if 'num_sequences' in info:
                content += f"- **Number of sequences**: {info['num_sequences']}\n"
            if 'clipped_tokens' in info and info['clipped_tokens'] > 0:
                content += f"- **Tokens clipped**: {info['clipped_tokens']} (don't fit into sequences)\n"
        
        # Add memory usage
        content += f"""
---

## Memory Usage

- **Total elements per experiment**: {expected_dims['memory']['total_elements']:,}
- **Memory per experiment**: {expected_dims['memory']['megabytes']} MB (float32)
- **Memory formula**: `num_sequences × sequence_window × channels × height × width × 4 bytes`

---
"""
        
        # Add actual dimensions if available
        if actual_dims and actual_dims:
            content += """
## Actual Data Dimensions (From Processed Files)

"""
            if 'sample_file' in actual_dims:
                content += f"### Sample File: `{actual_dims['sample_file']}`\n\n"
                
            if 'data_shape' in actual_dims and actual_dims['data_shape']:
                content += f"- **Data shape**: `{actual_dims['data_shape']}`\n"
                
                if 'interpretation' in actual_dims:
                    interp = actual_dims['interpretation']
                    content += f"""  - Sequences: {interp['num_sequences']}
  - Tokens per sequence: {interp['sequence_window']}
  - Channels: {interp['channels']}
  - Height: {interp['height']}
  - Token width: {interp['token_width']}
"""
            
            if 'label_shape' in actual_dims and actual_dims['label_shape']:
                content += f"- **Label shape**: `{actual_dims['label_shape']}`\n"
            
            if 'metadata' in actual_dims:
                meta = actual_dims['metadata']
                content += f"""
### Metadata Statistics
- **Total tokens**: {meta.get('total_tokens', 'N/A')}
- **Unique sequences**: {meta.get('unique_sequences', 'N/A')}
- **Unique experiments**: {meta.get('unique_experiments', 'N/A')}
"""
        
        # Add calculation formulas
        content += """
---

## Dimension Calculation Formulas

```python
# Number of tokens
num_tokens = (signal_length - token_window) // token_stride + 1

# Number of sequences  
num_sequences = num_tokens // sequence_window

# Height after clipping
height_clipped = samples2keep if clip_flag else original_height

# Height after decimation
height_decimated = height_clipped // decimation_factor

# Height with markers
height_with_markers = height + 2 if startend_markers else height

# Final output shape
output_shape = [num_sequences, sequence_window, channels, height_final, token_window]
```

---

## Loading Processed Data

### Python Example
```python
import h5py
import numpy as np

# Load single experiment
with h5py.File('processed_file.h5', 'r') as f:
    data = f['X'][:]  # Shape: [num_sequences, 10, 3, height, 5]
    labels = f['y'][:]  # Shape: [num_sequences, 10, 1]
    
# For PyTorch
import torch
data_tensor = torch.from_numpy(data).float()
label_tensor = torch.from_numpy(labels).long()
```

### Expected Shapes by Stage
1. **After loading**: `[num_sequences, 10, 3, height, 5]`
2. **After batching**: `[batch_size, num_sequences, 10, 3, height, 5]`
3. **For model input**: Depends on model architecture

---

## Validation Checklist

- [ ] Config token_window matches data width dimension
- [ ] Config sequence_window matches second dimension 
- [ ] Height dimension accounts for decimation factor
- [ ] Height includes +2 if start/end markers enabled
- [ ] Number of sequences × sequence_window ≤ total tokens
- [ ] Metadata CSV row count matches expected tokens

---

## Common Issues and Solutions

### 1. Dimension Mismatch in Model
**Problem**: Model expects different input shape
**Solution**: Check if start/end markers are consistently enabled/disabled

### 2. Memory Errors During Training
**Problem**: Out of memory errors
**Solution**: Reduce batch size or sequence_window

### 3. Tokens Don't Fit Into Sequences
**Problem**: Many tokens are clipped
**Solution**: Adjust token_stride or sequence_window for better fit

---

*This guide is automatically generated and updated by `dimension_checker.py`*
"""
        
        return content
    
    def update_guide(self, processed_data_path: Optional[str] = None) -> bool:
        """Update the dimensions guide with current configuration"""
        
        try:
            # Calculate expected dimensions
            expected_dims = self.calculate_expected_dimensions()
            
            # Check actual dimensions if path provided
            actual_dims = None
            if processed_data_path:
                actual_dims = self.check_processed_data(processed_data_path)
            
            # Generate new content
            content = self.generate_guide_content(expected_dims, actual_dims)
            
            # Create docs directory if it doesn't exist
            self.guide_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write updated guide
            with open(self.guide_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Updated dimensions guide: {self.guide_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update guide: {e}")
            return False
    
    def validate_dimensions(self, processed_data_path: str) -> List[str]:
        """Validate that actual dimensions match expected"""
        
        issues = []
        
        expected = self.calculate_expected_dimensions()
        actual = self.check_processed_data(processed_data_path)
        
        if not actual:
            issues.append("No processed data found to validate")
            return issues
        
        # Check if dimensions match
        if 'data_shape' in actual and actual['data_shape']:
            expected_shape = expected['transformations']['after_sequencing']['shape']
            actual_shape = actual['data_shape']
            
            # Compare dimensions (allowing for variable first dimension)
            if len(expected_shape) != len(actual_shape):
                issues.append(f"Dimension mismatch: expected {len(expected_shape)}D, got {len(actual_shape)}D")
            else:
                for i, (exp, act) in enumerate(zip(expected_shape[1:], actual_shape[1:]), 1):
                    if exp != act:
                        dim_names = ['sequences', 'sequence_window', 'channels', 'height', 'width']
                        issues.append(f"{dim_names[i]} mismatch: expected {exp}, got {act}")
        
        return issues
    
    def print_summary(self, processed_data_path: Optional[str] = None):
        """Print a summary of expected and actual dimensions"""
        
        print("=" * 60)
        print("DATA DIMENSION SUMMARY")
        print("=" * 60)
        
        expected = self.calculate_expected_dimensions()
        
        print("\n📋 Configuration:")
        for key, value in expected['config'].items():
            print(f"  {key}: {value}")
        
        print("\n📊 Expected Dimensions:")
        for stage, info in expected['transformations'].items():
            print(f"\n  {stage.replace('_', ' ').title()}:")
            print(f"    Shape: {info['shape']}")
            if 'num_tokens' in info:
                print(f"    Tokens: {info['num_tokens']}")
            if 'num_sequences' in info:
                print(f"    Sequences: {info['num_sequences']}")
        
        print(f"\n💾 Memory Usage:")
        print(f"  Per experiment: {expected['memory']['megabytes']} MB")
        
        if processed_data_path:
            actual = self.check_processed_data(processed_data_path)
            
            if actual:
                print("\n✅ Actual Data:")
                if 'data_shape' in actual:
                    print(f"  Data shape: {actual['data_shape']}")
                if 'label_shape' in actual:
                    print(f"  Label shape: {actual['label_shape']}")
                if 'metadata' in actual:
                    print(f"  Total tokens: {actual['metadata'].get('total_tokens', 'N/A')}")
                
                # Validate
                issues = self.validate_dimensions(processed_data_path)
                if issues:
                    print("\n⚠️  Validation Issues:")
                    for issue in issues:
                        print(f"  - {issue}")
                else:
                    print("\n✅ All dimensions validated successfully!")
        
        print("\n" + "=" * 60)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check and update data dimensions guide')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--data-path', help='Path to processed data for validation')
    parser.add_argument('--update', action='store_true', help='Update the dimensions guide')
    parser.add_argument('--validate', action='store_true', help='Validate dimensions')
    parser.add_argument('--summary', action='store_true', help='Print dimension summary')
    
    args = parser.parse_args()
    
    checker = DimensionChecker(args.config)
    
    if args.update:
        success = checker.update_guide(args.data_path)
        if success:
            print("✅ Dimensions guide updated successfully!")
        else:
            print("❌ Failed to update dimensions guide")
    
    if args.validate and args.data_path:
        issues = checker.validate_dimensions(args.data_path)
        if issues:
            print("⚠️  Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ All dimensions valid!")
    
    if args.summary or (not args.update and not args.validate):
        checker.print_summary(args.data_path)


if __name__ == '__main__':
    main()