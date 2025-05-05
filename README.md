# 'NoiseVision' -- Urban Noise Pollution Detection with Mask R-CNN

![image](https://github.com/user-attachments/assets/76459f0f-694b-4792-a12b-5ec394487289)


**NoiseVision** is a deep learning project that uses Mask R-CNN to detect and segment urban noise pollution in spectrogram images. By converting audio signals into visual spectrograms, this system can identify and localize different types of urban noises automatically. This is the final project for MUSA 6950 AI For Sustainbility.

--------

## 📋 **Table of Contents**
- Background
- Dataset
- Audio Preprocessing
- Environment Setup
- Data Preparation
- Model Architecture
- Training
- Inference
- Results

--------

## **🎯 Background**
Urban noise pollution affects millions daily. Traditional audio analysis methods process temporal data, but this project innovates by:
- Converting real world audio recordings to visual mel spectrograms image
- Applying computer vision techniques (Mask R-CNN) for noise detection
- Providing both classification and spatial localization of noise sources

--------
## **📊 Dataset**
_ESC-50 (Environmental Sound Classification 50)_

- More detailed info about this [dataset](https://datasets.activeloop.ai/docs/ml/datasets/esc-50-dataset/)
- 2,000 short audio recordings of real-world environments
- 5 seconds duration per clip
- 50 semantic sound classes
- 40 audio recordings of each class
- Designed for benchmarking environmental sound classification methods

--> Category Used For This Project: Exterior/urban noises (excluded the "church bell" category)

--------
## **🛒 Classes**
The dataset includes various urban noise sources:
- Car horn
- Airplane
- Chainsaw
- Engine
- Fireworks
- Handsaw
- Helicopter
- Siren
- Train

--------
## **🎵 Audio Preprocessing**

### Audio → Mel Spectrogram Conversion
Mel Spectrogram Generation: Converting raw audio to visual representations for computer vision processing.

**_1. Why Mel Spectrograms?_**
- Time-Frequency Representation: Captures both temporal (when) and spectral (what) characteristics
- Human Perception: Mel scale matches human auditory perception by compressing detail at high frequencies and expanding detail at low frequencies
- Visual Analysis: Transforms audio processing into computer vision problem
- Preservation of Information: Maintains timing, frequency, and amplitude details

**_2. Mel Scale Transformation_**
The mel scale is a non-linear frequency scale that better represents human pitch perception:
`mel = 2595 * log10(1 + f/700)`
Where `f` is frequency in `Hz`.

**Key Properties:**
- Equal distances on mel scale correspond to equal perceived pitch differences
- Higher resolution at lower frequencies (where humans are more sensitive)
- Lower resolution at higher frequencies (mimicking human hearing)

**_3. Conversion Process_**

**Step-by-Step Pipeline:** Raw Audio Input - 16kHz mono audio files (5 seconds each)

**Short-Time Fourier Transform (STFT):**
- Window size: 1024 samples (64ms at 16kHz)
- Hop length: 512 samples (32ms overlap)
- Hamming window function


- Mel Filter Bank: 128 mel bands spanning full frequency range
- Power Spectrum: Calculate magnitude squared
- Log Scale: Apply log transform for better visualization

_**4. Visualization Example**_

<img width="118" alt="Screenshot 2025-05-04 at 12 57 52 PM" src="https://github.com/user-attachments/assets/348bff55-e8f5-4133-8235-f83178d81a2c" />

**Mel Spectrogram Features:**

- X-axis: Time (seconds)
- Y-axis: Mel frequency bins (low to high)
- Color: Magnitude (dB)

- Yellow: High energy
- Black: Low energy
- Red: Medium energy


## 🔧 Environment Setup for Modeling

### Dependencies [GPU recommended (CUDA-enabled)]

```
# Audio processing
librosa
soundfile

# Core ML libraries
torch
torchvision
torchaudio

# Visualization and data processing
matplotlib
pandas
pillow
distinctipy

# PyTorch utilities
torchtnt==0.2.0
tqdm
tabulate

# Custom utility packages
cjm_pandas_utils
cjm_psl_utils
cjm_pil_utils
cjm_pytorch_utils
cjm_torchvision_tfms
```


## **📁 Data Preparation**

_**1. Annotation Format**_

LabelMe JSON format containing:
- Image metadata
- Polygon annotations for noise regions
- Class labels for each polygon

_**2. Train/Validation Split**_
- 80% training data
- 20% validation data
- Random shuffle for fair distribution


## **🤖 Model Architecture**

**_Mask R-CNN Configuration_**

- Base Model: ResNet-50 FPN v2 (COCO pre-trained)

**🔴 Modifications:**
- Custom FastRCNNPredictor for urban noise classes
- Custom MaskRCNNPredictor for segmentation
- Output: Bounding boxes and segmentation masks for noise regions


## **🚄 Training**

**_Data Augmentation_**
- Random IoU Crop
- Color Jitter (brightness, contrast, saturation, hue)
- Random Grayscale
- Random Horizontal Flip

_**Training Configuration**_
- Optimizer: AdamW
- Learning Rate: 5e-4
- Scheduler: OneCycleLR
- Epochs: 40
- Batch Size: 4
- Image Size: 512x512


## **📈 Results**

_**Training Metrics**_

Final Model Performance:
- Final Training Loss: 0.083785
- Final Validation Loss: 0.158384
- Best Validation Loss: 0.147019 (achieved at Epoch 0)
- Final Loss Difference: 0.074599

![image](https://github.com/user-attachments/assets/effd5c4f-1d74-43a4-8b35-582b9d04c751)

1. **Training Progress**: The model shows steady improvement with training loss decreasing from ~0.089 to 0.084 over 40 epochs
2. **Validation Behavior**: Validation loss started at 0.147 (best value) and stabilized around 0.158
3. **Loss Difference**: The gap between validation and training loss remains relatively stable, indicating good generalization without severe overfitting
4. **Learning Pattern**: The validation-training loss difference (right plot) shows some fluctuation but maintains a consistent range around 0.04-0.08


_**Model Performance**_
- Consistent learning with gradual improvement
- Good generalization demonstrated by stable train-validation gap
- No severe overfitting observed despite 40 epochs of training


## **🔄 Future Work**

- Extend to more noise categories
- Implement real-time audio processing
- Develop web interface API for demo (maybe)
