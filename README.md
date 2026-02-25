🧠 Breast Cancer Classification using Neural Network

### What Does This Model Do?
In your project:
The model takes 30 medical features of a tumor as input
and predicts whether the tumor is:
Malignant (Cancerous)
Benign (Non-cancerous)
So basically:
👉 Input → Patient tumor measurements
👉 Output → Cancer type prediction

### How Does the Model Work?
It looks at training data
It learns patterns between features and labels
It adjusts internal weights (mathematical parameters)
After training, it can predict on new unseen data

### Real Life Example
Think of it like this:
A doctor studies thousands of past reports.
After experience, the doctor can look at new reports and say whether the tumor is dangerous or not.
Your neural network model does the same thing — but mathematically.

📌 Project Overview
This project implements a Deep Learning model to classify breast cancer tumors as Malignant or Benign using a Neural Network built with TensorFlow and Keras.
The dataset used is the built-in Breast Cancer dataset available in scikit-learn.
The objective of this project is to understand the complete Deep Learning workflow — from data preprocessing to model training and prediction.

🚀 Technologies Used
Python (3.10)
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook

📊 Dataset Information
Dataset: Breast Cancer Wisconsin Dataset
Source: scikit-learn built-in dataset
Total Features: 30
Target Classes:
0 → Malignant
1 → Benign

### Project Workflow

1️⃣ Import Required Libraries
2️⃣ Load Dataset from scikit-learn
3️⃣ Convert Dataset into Pandas DataFrame
4️⃣ Data Preprocessing & Feature Standardization
5️⃣ Train-Test Split
6️⃣ Build Neural Network using Keras Sequential API
7️⃣ Compile Model (Optimizer, Loss, Metrics)
8️⃣ Train Model
9️⃣ Evaluate Model Performance
🔟 Make Predictions using model.predict()
1️⃣1️⃣ Convert Probabilities to Class Labels using argmax

#### Model Architecture
Input Layer: 30 features
Hidden Layer: Dense (20 neurons, ReLU activation)
Output Layer: Dense (2 neurons, Sigmoid activation)
Loss Function: sparse_categorical_crossentropy
Optimizer: Adam
Metric: Accuracy

### Model Training
Validation Split: 0.1
Epochs: 10
Data Standardized using StandardScaler

🎯 Key Learnings
Importance of environment management (Python version compatibility)
Understanding neural network architecture
Difference between prediction probability and predicted class
How argmax converts probabilities into final labels
Complete Deep Learning pipeline implementati

💻 How to Run the Project
Create a virtual environment (recommended Python 3.10)
Install required libraries:
(Code)= 

pip install numpy pandas matplotlib scikit-learn tensorflow

Open Jupyter Notebook
Run all cells in order
