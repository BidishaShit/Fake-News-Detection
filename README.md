
# Fake News Detection Using LSTM Deep Learning

## Overview

This project implements a **Fake News Detection System** using **Natural Language Processing (NLP)** and a **Long Short-Term Memory (LSTM)** neural network. The model is trained on real and fake news datasets to classify news articles as either **REAL** or **FAKE**.

The workflow includes:

* Data loading and preprocessing
* Text cleaning and normalization
* Tokenization and sequence padding
* LSTM-based deep learning model training
* Model evaluation and visualization
* Model saving/loading
* Predicting custom news articles

---

## Dataset

The project uses two CSV files from Kaggle - https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

* `Fake.csv` – Contains fake news articles
* `True.csv` – Contains real news articles

Each dataset includes:

* Title
* Text
* Subject
* Date

Labels are assigned as:

| Label | Meaning   |
| ----- | --------- |
| 0     | Fake News |
| 1     | Real News |

---

## Technologies Used

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* TensorFlow / Keras

### Deep Learning

* Embedding Layer
* LSTM Layer
* Dropout Regularization
* Sigmoid Output Layer

---

## Project Workflow

### 1. Data Loading

Load both datasets and assign labels.

```python
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
```

---

### 2. Data Preprocessing

The datasets are merged and cleaned:

* Remove unnamed columns
* Remove missing values
* Shuffle records
* Combine title and text fields

```python
df['content'] = df['title'] + " " + df['text']
```

---

### 3. Text Cleaning

The text preprocessing pipeline performs:

* Lowercasing
* Removal of special characters
* Tokenization
* Stopword removal
* Porter stemming

Example:

```text
Original:
"The President announced new economic policies."

Processed:
"presid announc new econom polici"
```

---

### 4. Train-Test Split

Dataset split:

* 80% Training
* 20% Testing

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

### 5. Tokenization and Padding

Text is converted into numerical sequences.

Parameters:

```python
vocab_size = 10000
max_length = 300
```

Padding ensures all sequences have equal length.

---

### 6. LSTM Model Architecture

```text
Embedding Layer
        ↓
LSTM (128 Units)
        ↓
Dropout (0.5)
        ↓
Dense (1 Unit, Sigmoid)
```

Model configuration:

```python
loss='binary_crossentropy'
optimizer='adam'
metrics=['accuracy']
```

---

### 7. Early Stopping

Training uses Early Stopping to prevent overfitting.

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

---

### 8. Model Training

```python
model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64
)
```

---

### 9. Model Evaluation

Performance is measured using:

* Test Loss
* Test Accuracy
* Confusion Matrix

```python
loss, accuracy = model.evaluate(
    X_test_pad,
    y_test
)
```

---

### 10. Confusion Matrix Visualization

A heatmap is generated using Seaborn.

```python
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
```

This provides insight into:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

### 11. Model Saving and Loading

Save trained model:

```python
model.save('fake_news_model.h5')
```

Load model later:

```python
loaded_model = load_model(
    'fake_news_model.h5'
)
```

---

### 12. Custom News Prediction

Example:

```python
sample_news = "Aliens officially elected a cat as president of the Moon."
```

Prediction pipeline:

1. Preprocess text
2. Convert to sequence
3. Pad sequence
4. Predict class

Output:

```text
FAKE News
```

or

```text
REAL News
```

---

## Project Structure

```text
Fake-News-Detection/
│
├── Fake.csv
├── True.csv
├── fake_news_model.h5
├── python Fake News Detection System.ipynb
├── README.md
│
└── outputs/
    └── confusion_matrix.png
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/BidishaShit/Fake-News-Detection.git
cd Fake-News-Detection
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow
```

### Download NLTK Resources

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

---

## Running the Project

Execute:

```bash
python Fake News Detection System.ipynb
```

The script will:

1. Load data
2. Preprocess text
3. Train the model
4. Evaluate performance
5. Save the trained model
6. Predict sample news


---

## Results

The model learns semantic patterns from news articles and can effectively distinguish between fake and real news content. Performance depends on dataset quality, preprocessing techniques, and hyperparameter tuning.

---

## Author

Developed as a Deep Learning and Natural Language Processing project for Fake News Classification using TensorFlow/Keras and LSTM networks.

