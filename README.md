#  Handwritten Character Recognition

> Recognizing handwritten digits using Deep Learning CNN — CodeAlpha Internship Task 3

---

##  Objective
Build a Deep Learning model that can **recognize handwritten digits (0-9)** from images using a Convolutional Neural Network (CNN).

---

##  How It Works
- MNIST dataset loaded directly from Keras — 70,000 handwritten digit images
- Images normalized by dividing pixel values by 255
- Data split into 60,000 training and 10,000 testing images
- CNN model built with Conv2D, MaxPooling, Flatten and Dense layers
- Model trained for 5 epochs
- Predictions tested on random images and verified visually

---

##  What is CNN?
Convolutional Neural Network is a Deep Learning algorithm that:
- Scans images layer by layer
- Learns edges, shapes and patterns automatically
- Much more powerful than traditional ML for image recognition

---

##  Tech Stack
- Python
- TensorFlow
- Keras
- NumPy
- matplotlib

---

## 📁 Project Structure
```
handwritten_recognition/
├── handwritten_model.py   → Build and train CNN model
├── predict.py             → Load model and test predictions
├── digit_cnn.keras        → Saved trained CNN model
├── accuracy_plot.png      → Training accuracy plot
└── predictions.png        → Sample prediction results
```

---

##  How to Run
```
pip install tensorflow keras matplotlib numpy
python handwritten_model.py
python predict.py
```

---

##  Results
- **Test Accuracy: 99.21%**
- Model correctly identifies handwritten digits with 99% accuracy
- Green title = Correct prediction ✅
- Red title = Wrong prediction ❌

---

##  Sample Predictions
Model tested on 10 random handwritten digit images from MNIST test set. Nearly all predictions were correct showing the power of CNN for image recognition!
