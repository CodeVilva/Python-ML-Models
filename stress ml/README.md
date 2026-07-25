# 🧠 Stress Detection ML

A lightweight Machine Learning-based **text classification system** that analyzes text messages and predicts whether the content indicates a stressed or non-stressed state.

The model uses **TF-IDF text vectorization** combined with a **Multinomial Naive Bayes classifier**.

A Flask REST API is included so that Java web applications can send text through JSON and receive the model's prediction.

---

## 🚀 Overview

The system accepts a text message as input and classifies it using a trained Machine Learning model.

Example:

```text
Input:
"deadline pressure is high"

Output:
Stress classification
```

The model is designed to be used as an ML component inside larger Java web applications.

---

## 🧠 Machine Learning Pipeline

The project uses a two-stage text classification pipeline:

```text
Text Message
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Multinomial Naive Bayes
     │
     ▼
Stress Classification
```

### TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** converts text into numerical feature vectors based on the importance of words within the dataset.

### Multinomial Naive Bayes

The transformed text is classified using a **Multinomial Naive Bayes** classifier.

---

## 📊 Dataset

The model uses:

```text
stress_data.csv
```

The dataset contains text samples and their corresponding stress labels.

Expected columns:

```text
text
stress
```

Example structure:

| text                      | stress       |
| ------------------------- | ------------ |
| deadline pressure is high | stressed     |
| I feel relaxed today      | not stressed |
| I am tired and frustrated | stressed     |

The exact labels depend on the dataset used during training.

---

## 🔄 Training Workflow

```text
Stress Dataset
      │
      ▼
Text Samples
      │
      ▼
TF-IDF Vectorizer
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Trained Pipeline
      │
      ▼
stress_model.pkl
```

The vectorizer and classifier are stored together as a single Scikit-learn Pipeline.

---

## 📁 Project Structure

```text
stress_ml/
│
├── stress_data.csv
├── train_model.py
├── test_model.py
├── app.py
└── stress_model.pkl
```

---

## 📦 Saved Model

The trained pipeline is saved as:

```text
stress_model.pkl
```

It contains both:

```text
TF-IDF Vectorizer
        +
Multinomial Naive Bayes Classifier
```

This makes the inference process simple because raw text can be passed directly to the loaded pipeline.

---

## 🔌 REST API

A Flask REST API is included for external application integration.

### Endpoint

```text
POST /predict
```

### Port

```text
5000
```

### Request

```json
{
    "message": "I am feeling tired and frustrated"
}
```

### Response

```json
{
    "stress": "stressed"
}
```

The returned value depends on the labels contained in the training dataset.

---

## ☕ Java Web Application Integration

The API can be integrated into Java applications using HTTP and JSON.

```text
┌───────────────────────────┐
│    Java Web Application   │
│                           │
│ JSP / Servlet / Java      │
│                           │
│ User enters a message     │
└─────────────┬─────────────┘
              │
              │ JSON
              ▼
┌───────────────────────────┐
│       Flask API           │
│        /predict           │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    TF-IDF + Naive Bayes   │
│        ML Pipeline        │
└─────────────┬─────────────┘
              │
              │ JSON
              ▼
       Stress Prediction
```

This allows the Java application to use Python's Machine Learning ecosystem without implementing the NLP model directly in Java.

---

## ▶️ Running the Project

### 1. Install dependencies

```bash
pip install pandas scikit-learn flask joblib
```

### 2. Train the model

```bash
python train_model.py
```

This creates:

```text
stress_model.pkl
```

### 3. Test the model

```bash
python test_model.py
```

The test script contains sample messages for checking model predictions.

### 4. Start the Flask API

```bash
python app.py
```

The API runs on:

```text
http://localhost:5000
```

---

## 🧪 Example Test Messages

The project currently includes examples such as:

```text
deadline pressure is high
I feel relaxed today
I am tired and frustrated
```

These can be replaced or expanded with additional test messages.

---

## 🛠️ Technologies

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Flask
* Joblib
* JSON
* REST API

---

## 🎯 Purpose

This project was developed to explore basic **Natural Language Processing (NLP)** and text classification while creating a reusable Python ML component for Java web applications.

It demonstrates how a Java application can send text data to a Python Machine Learning service and receive a classification result through JSON.

---

## 🔮 Future Improvements

* Add larger and more diverse datasets
* Evaluate using accuracy, precision, recall and F1 score
* Improve text preprocessing
* Compare Naive Bayes with Logistic Regression and other NLP classifiers
* Add confidence/probability scores
* Improve API validation
* Integrate with Java Servlet/JSP applications
* Add authentication to the API
* Explore transformer-based NLP models
* Improve classification for different stress levels

---

## ⚠️ Disclaimer

This project is intended for educational and experimental purposes.

Text-based stress classification is **not a medical or psychological diagnostic system**. Predictions should not be treated as a professional assessment.

---

## 👨‍💻 Author

**Vilva**

Java Full-Stack Developer | Python & Machine Learning Enthusiast
