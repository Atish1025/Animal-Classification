
---

## 🐾 Animal Classification using MobileNetV2

This project is a deep learning-based image classification system that identifies different animals from images using transfer learning with **MobileNetV2**. It is built with **TensorFlow** and **Keras**, and is trained on a custom image dataset organized by animal classes.

### 📂 Project Structure

```
Animal-Classification/
├── dataset/               # Contains subfolders for each animal class
├── Animal.py              # Main training and evaluation script
├── README.md              # Project description (this file)
```

### 🚀 Features

* Utilizes **MobileNetV2** pretrained on ImageNet for efficient transfer learning.
* Augments data using flipping and zooming for better generalization.
* Visualizes training and validation accuracy curves.
* Outputs classification report and confusion matrix for model evaluation.

### 🧠 Model Architecture

* Pretrained **MobileNetV2** base (frozen layers)
* Global Average Pooling
* Dense layer with ReLU activation
* Final softmax layer for multi-class classification

### 📊 Training Configuration

* Image size: `224x224`
* Batch size: `32`
* Optimizer: `Adam`
* Loss function: `categorical_crossentropy`
* Epochs: `10`
* Validation split: `20%`

### 📈 Example Output

* Classification report with precision, recall, and F1-score
* Accuracy vs Epoch plot for both training and validation sets

### 🛠 Requirements

* Python 
* TensorFlow
* NumPy
* Matplotlib
* scikit-learn

Install dependencies with:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 📌 How to Run

1. Prepare your dataset in the `dataset/` folder, with one subfolder per animal class (e.g., `cat/`, `dog/`, `lion/`, etc.).
2. Run the training script:

   ```bash
   python Animal.py
   ```

---


