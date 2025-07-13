# Segmentation Fault Fix for integrated_gradients_4.py

## Problem Description
The script `integrated_gradients_4.py` was experiencing a segmentation fault when trying to import TensorFlow:
```
zsh: segmentation fault  python integrated_gradients_4.py --check-hash-based-pycs always
```

## Root Cause
The segmentation fault was caused by **Python 3.13 compatibility issues** with TensorFlow. The current system was using Python 3.13.5, which is too new for stable TensorFlow support.

## Solution
Created a new conda environment with Python 3.11 and properly installed TensorFlow:

### Step 1: Create New Environment
```bash
conda create -n tf-env python=3.11 -y
```

### Step 2: Activate Environment
```bash
conda activate tf-env
```

### Step 3: Install Required Packages
```bash
pip install tensorflow tensorflow-hub matplotlib numpy
```

### Step 4: Run the Script
```bash
python integrated_gradients_4.py
```

## Results
- ✅ Script runs successfully without segmentation faults
- ✅ All 13 visualization images generated correctly
- ✅ TensorFlow 2.19.0 working properly
- ✅ All integrated gradients computations completed

## Environment Details
- **Python Version**: 3.11.13 (in tf-env)
- **TensorFlow Version**: 2.19.0
- **System**: macOS ARM64 (Apple Silicon)
- **Conda Environment**: tf-env

## Usage Instructions
To run the script in the future:
```bash
# Activate the environment
conda activate tf-env

# Run the script
python integrated_gradients_4.py
```

## Key Points
- Python 3.13 is too new for TensorFlow compatibility
- Python 3.11 is the recommended version for stable TensorFlow usage
- The script generates comprehensive visualizations of integrated gradients
- All output images are saved in the `images/` directory
