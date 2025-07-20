# Issue 1: Incorrect Model Configuration Parameters

## Problem Description

The current implementation in `midi_to_frames.py` uses incorrect model configuration parameters that don't match the official PianoMotion10M architecture. This leads to model architecture mismatches and potential loading failures.

### Specific Issues:

1. **Wrong `bs_dim` for Piano2Posi**: 
   - Current code uses `bs_dim = 6` for Piano2Posi
   - But the diffusion args incorrectly sets `bs_dim = 96` 
   - Piano2Posi should use `bs_dim = 6` (xyz coordinates for both hands)

2. **Inconsistent `num_layer` configuration**:
   - Current code uses `num_layer = 8` for both models
   - Official documentation shows different values for base vs large models
   - Base model: 4 layers, Large model: 8 layers

3. **Missing proper model size distinction**:
   - The code doesn't distinguish between base and large model configurations
   - Using large model checkpoint requires large model parameters

4. **Incorrect feature dimensions**:
   - Current code assumes fixed dimensions without checking model variant
   - Different models (base vs large) have different parameter requirements

### Code Location:
- File: `midi_to_frames.py`
- Lines: 49-75 (Piano2PosiArgs and DiffusionArgs classes)

## Recommended Fixes

### 1. Create Proper Model Configuration Classes

Replace the existing args classes with properly configured ones:

```python
class Piano2PosiArgs:
    def __init__(self, model_type='large'):
        self.feature_dim = 512
        self.bs_dim = 6  # CRITICAL: Position predictor always uses 6 (xyz for both hands)
        self.loss_mode = 'naive_l2'
        self.encoder_type = 'transformer'
        self.hidden_type = 'audio_f'
        self.wav2vec_path = "facebook/hubert-large-ls960-ft"  # For large model
        self.max_seq_len = 1500
        self.period = 1500
        self.latest_layer = "tanh"
        
        # Configure based on model type
        if model_type == 'large':
            self.num_layer = 8
            self.wav2vec_path = "facebook/hubert-large-ls960-ft"
        elif model_type == 'base':
            self.num_layer = 4
            self.wav2vec_path = "facebook/hubert-base-ls960"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class DiffusionArgs:
    def __init__(self, model_type='large'):
        self.timesteps = 1000
        self.train_sec = 4
        self.xyz_guide = True
        self.remap_noise = True
        self.bs_dim = 96  # Gesture generator always uses 96 (hand rotation parameters)
        self.encoder_type = 'transformer'
        self.experiment_name = "midi_inference"
        
        # Configure based on model type
        if model_type == 'large':
            self.unet_dim = 256
            self.num_layer = 8
        elif model_type == 'base':
            self.unet_dim = 128  # Smaller for base model
            self.num_layer = 4
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

### 2. Update Model Instantiation

Modify the model creation code to use proper configurations:

```python
# Determine model type from checkpoint name
model_type = 'large' if 'large' in CHECKPOINT_PATH else 'base'

# Create model components with correct configuration
piano2posi_args = Piano2PosiArgs(model_type)
diffusion_args = DiffusionArgs(model_type)

# Set conditional dimension based on wav2vec model
if 'large' in piano2posi_args.wav2vec_path:
    cond_dim = 1024
elif 'base' in piano2posi_args.wav2vec_path:
    cond_dim = 768
else:
    raise ValueError("Unknown wav2vec model type")
```

### 3. Add Configuration Validation

Add validation to ensure configuration consistency:

```python
def validate_model_config(piano2posi_args, diffusion_args, checkpoint_path):
    """Validate that model configuration matches checkpoint."""
    
    # Check if model type in checkpoint matches configuration
    if 'large' in checkpoint_path.lower():
        assert piano2posi_args.num_layer == 8, "Large model should have 8 layers"
        assert diffusion_args.unet_dim == 256, "Large model should have unet_dim=256"
    elif 'base' in checkpoint_path.lower():
        assert piano2posi_args.num_layer == 4, "Base model should have 4 layers"
        assert diffusion_args.unet_dim == 128, "Base model should have unet_dim=128"
    
    # Validate critical dimensions
    assert piano2posi_args.bs_dim == 6, "Piano2Posi must use bs_dim=6"
    assert diffusion_args.bs_dim == 96, "Diffusion model must use bs_dim=96"
    
    print("✓ Model configuration validation passed")
```

### 4. Complete Fixed Code Structure

```python
def main():
    print("=== PianoMotion10M MIDI to Hand Motion Inference ===")
    
    # Determine model type from checkpoint
    model_type = 'large' if 'large' in CHECKPOINT_PATH else 'base'
    print(f"Using {model_type} model configuration")
    
    # Create proper configurations
    piano2posi_args = Piano2PosiArgs(model_type)
    diffusion_args = DiffusionArgs(model_type)
    
    # Validate configuration
    validate_model_config(piano2posi_args, diffusion_args, CHECKPOINT_PATH)
    
    # Set conditional dimension
    cond_dim = 1024 if 'large' in piano2posi_args.wav2vec_path else 768
    
    # Create models with correct parameters
    piano2posi = Piano2Posi(piano2posi_args)
    
    unet = Unet1D(
        dim=diffusion_args.unet_dim,
        dim_mults=(1, 2, 4, 8),
        channels=diffusion_args.bs_dim,
        remap_noise=diffusion_args.remap_noise,
        condition=True,
        guide=diffusion_args.xyz_guide,
        guide_dim=6 if diffusion_args.xyz_guide else 0,
        condition_dim=cond_dim,
        encoder_type=diffusion_args.encoder_type,
        num_layer=diffusion_args.num_layer
    )
    
    # Continue with model creation...
```

## Testing the Fix

1. **Verify checkpoint loading**: The model should load without dimension mismatch errors
2. **Check model architecture**: Print model summary to verify layer counts
3. **Validate inference**: Run a small test to ensure the model produces outputs of expected dimensions

## Impact of Fix

- Eliminates model architecture mismatches
- Ensures compatibility with different model variants (base/large)
- Prevents silent errors from incorrect parameter configurations
- Makes the code more maintainable and extensible 