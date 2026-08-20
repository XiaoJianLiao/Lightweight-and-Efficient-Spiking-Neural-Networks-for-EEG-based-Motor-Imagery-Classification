# Dataset Description: Motor Imagery with Three Classes (Hand, Elbow, Rest)

## Output File Structure
- **File Format**: MATLAB .mat file
- **File Name**: `sub-011_eeg.mat`
- **Main Variables**:
  - `all_data`: 3D array of shape [trials, channels, time_points]
  - `all_label`: 1D array of shape [trials] containing class labels

## Data Specifications

### Dimensions & Properties
- **Number of Trials**: 965 (300 Hand, 300 Elbow, 365 Rest)
- **Number of Channels**: 62 (EEG channels without EOG and EMG)
- **Time Points per Trial**: ~360 (4 seconds at 90Hz sampling rate)
- **Classes**:
  - 0: Rest state
  - 1: Elbow movement imagery
  - 2: Hand movement imagery

### Preprocessing Applied
- Common Average Reference (CAR)
- Bandpass filtered: 2-45 Hz
- Resampled to 90 Hz
- Artifact removal using AAR (Automatic Artifact Removal)
- 4-second trial segmentation aligned with motor imagery task

## Parameters for Data Loader

```python
# Data loading parameters
batch_size = 64
shuffle = True
train_test_split_ratio = 0.2

# Data loader function call example
from sklearn.model_selection import train_test_split
import scipy.io as scio

data = scio.loadmat('sub-011_eeg.mat')
all_data = data['all_data']  # Shape: [trials, channels, time_points]
all_label = data['all_label'].ravel()  # Shape: [trials]

# Split data into training and testing sets
train_data, test_data, train_label, test_label = train_test_split(
    all_data, all_label, test_size=train_test_split_ratio, random_state=42
)

# For PyTorch data loader
train_loader = data_loader(train_data, train_label, batch=batch_size, shuffle=True)
test_loader = data_loader(test_data, test_label, batch=batch_size, shuffle=False)
```

## Recommended ANN Model Configuration

- **Input Shape**: [1, channels, time_points] (1 is for batch dimension)
- **Output Shape**: 3 classes (for Rest, Elbow, Hand)
- **Suggested Architecture**: Temporal-Spatial CNN (like LENet in your script)
  - Temporal convolutions for feature extraction across time
  - Spatial convolutions for relationships between channels
  - Pooling layers to reduce dimensions
  - Fully connected output layer for classification

## Notes for ANN to SNN Conversion
- Ensure ReLU activations are used (compatible with spiking neurons)
- Avoid using batch normalization layers if possible (harder to convert)
- Consider model quantization constraints during training
- Time steps (T) for the SNN: 100 (as in your script)

This dataset is suitable for the LENet architecture in your provided code, which was specifically designed for motor imagery classification tasks.