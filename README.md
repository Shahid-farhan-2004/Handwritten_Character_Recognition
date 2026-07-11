# Handwritten Character Recognition using CNN (EMNIST Balanced)

## Overview

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify handwritten characters from the **EMNIST Balanced** dataset.

The model learns to recognize **47 different classes**, including digits and uppercase/lowercase letters, by extracting image features through convolutional layers.

---

## Features

- Uses the **EMNIST Balanced** dataset
- Convolutional Neural Network (CNN)
- ReLU activation function
- Max Pooling layers
- Fully Connected layers
- Adam optimizer
- CrossEntropyLoss for multi-class classification
- Accuracy evaluation on the test dataset

---

## Technologies Used

- Python
- PyTorch
- Torchvision

---

## Dataset

Dataset: **EMNIST Balanced**

- Image Size: **28 × 28**
- Color Channels: **1 (Grayscale)**
- Number of Classes: **47**

The dataset is downloaded automatically using:

```python
datasets.EMNIST(
    root="./data",
    split="balanced",
    train=True,
    download=True
)
```

---

## Data Preprocessing

The following preprocessing steps are applied:

- Rotate images by **-90°**
- Convert images to tensors

```python
transform = transforms.Compose([
    transforms.RandomRotation((-90, -90)),
    transforms.ToTensor()
])
```

### Why Rotation?

EMNIST images are stored in a rotated orientation.

Applying a **-90° rotation** makes the characters upright so that the CNN learns from correctly oriented images.

---

## CNN Architecture

```
Input Image (1 × 28 × 28)
        │
        ▼
Conv2D (1 → 32, Kernel = 3×3, Padding = 1)
        │
        ▼
ReLU
        │
        ▼
MaxPool2D (2×2)
        │
        ▼
Conv2D (32 → 64, Kernel = 3×3, Padding = 1)
        │
        ▼
ReLU
        │
        ▼
MaxPool2D (2×2)
        │
        ▼
Flatten
        │
        ▼
Linear (3136 → 256)
        │
        ▼
ReLU
        │
        ▼
Linear (256 → 47)
        │
        ▼
CrossEntropyLoss
```

---

## Model Components

### Convolution Layers

Extract image features such as:

- Edges
- Curves
- Corners
- Character strokes

```python
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
```

---

### ReLU Activation

ReLU introduces **non-linearity**, allowing the CNN to learn complex image patterns.

```python
F.relu(x)
```

---

### Max Pooling

Pooling reduces the spatial size of feature maps.

Benefits:

- Faster training
- Reduced memory usage
- Less overfitting

```python
self.pool = nn.MaxPool2d(2, 2)
```

---

### Flatten Layer

Converts the 3D feature map into a 1D vector before passing it to the fully connected layer.

```python
x = torch.flatten(x, 1)
```

---

### Fully Connected Layers

The fully connected layers perform the final classification.

```python
self.fc1 = nn.Linear(64 * 7 * 7, 256)
self.fc2 = nn.Linear(256, 47)
```

---

## Loss Function

The project uses:

```python
nn.CrossEntropyLoss()
```

This loss function is designed for **multi-class classification**.

> **Important:** Do **not** apply `Softmax` or `ReLU` to the final layer before `CrossEntropyLoss`, as the loss function internally handles the required computations.

---

## Optimizer

Adam optimizer is used.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Advantages:

- Fast convergence
- Adaptive learning rate
- Good default optimizer for CNNs

---

## Training Process

For each epoch:

1. Load a batch of images
2. Perform forward propagation
3. Compute loss
4. Perform backpropagation
5. Update model weights

---

## Testing

After training:

- Predictions are generated
- Predicted labels are compared with actual labels
- Accuracy is calculated

---

## Project Workflow

```
EMNIST Dataset
        │
        ▼
Image Rotation
        │
        ▼
Tensor Conversion
        │
        ▼
CNN Model
        │
        ▼
Forward Pass
        │
        ▼
Loss Calculation
        │
        ▼
Backward Pass
        │
        ▼
Weight Update
        │
        ▼
Prediction
        │
        ▼
Accuracy
```

---

# ⚠️ Common Issue: SSL Certificate Verification Error

While downloading the EMNIST dataset, you may encounter an error similar to:

```text
ssl.SSLCertVerificationError:
certificate verify failed:
certificate has expired
```

or

```text
urllib.error.URLError:
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

## Why does this happen?

This error is **not related to the CNN model or training code**.

It occurs because Python attempts to establish a secure HTTPS connection to download the EMNIST dataset, but the server's SSL certificate cannot be verified.

Possible reasons include:

- Incorrect system date or time
- Expired or outdated SSL certificates
- Outdated Python certificate store
- Firewall, antivirus, or proxy intercepting HTTPS traffic
- Temporary server-side certificate issues

---

## How to Fix It

### 1. Verify System Date and Time

Ensure your operating system has the correct:

- Date
- Time
- Time Zone

---

### 2. Upgrade the Certificate Bundle

```bash
python -m pip install --upgrade certifi
```

---

### 3. Update PyTorch and Torchvision

```bash
pip install --upgrade torch torchvision
```

---

### 4. Use a Local Dataset Folder

Recommended:

```python
root="./data"
```

instead of

```python
root="/data"
```

---

### 5. Download the Dataset Manually (if required)

If automatic downloading continues to fail:

1. Download the EMNIST dataset manually from the official source.
2. Extract it into the `./data` directory.
3. Set:

```python
download=False
```

---

## Important Note

An SSL certificate error **does not indicate a problem with the neural network implementation**.

The program stops **before** the model begins training because the dataset cannot be downloaded.

Once the dataset is available locally or the SSL issue is resolved, the code should execute and train normally.

---

## Future Improvements

- Add Batch Normalization
- Add Dropout to reduce overfitting
- Save and load trained models
- Display training and validation accuracy
- Plot loss and accuracy graphs
- Implement learning rate scheduling
- Add confusion matrix visualization

---

## License

This project is intended for educational and learning purposes using the PyTorch deep learning framework.
