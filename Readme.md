<p align="center">
  <img src="https://em-content.zobj.net/thumbs/240/apple/354/credit-card_1f4b3.png" width="110" alt="Credit Card Fraud Detection Logo" style="margin: 10px; border-radius: 12px;">
</p>

# 💳 Credit Card Fraud Detection – Machine Learning Project

This project uses machine learning to detect fraudulent credit card transactions. Since fraud cases are rare and often subtle, the problem is treated as an **anomaly detection** or **binary classification** task with strong emphasis on handling class imbalance.

---

## 🎯 Objective

- Build a model that accurately classifies transactions as **fraudulent** or **legitimate**
- Address the extreme class imbalance (fraud cases <1%)
- Evaluate the model using suitable metrics like **Precision**, **Recall**, and **F1-score**

---

## 📂 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**:
  - Transactions made by European cardholders in 2013
  - Contains 284,807 transactions, with only 492 frauds (~0.17%)
  - Features: `Time`, `Amount`, and 28 anonymized features (`V1` to `V28`) derived from PCA
  - **Target**: `Class` (0 = Legitimate, 1 = Fraudulent)

> 📌 Place the dataset as `creditcard.csv` inside the `data/` folder.

---

## 🚀 Project Workflow

1. **Data Exploration**
   - Check data distribution, missing values, class imbalance
   - Visualize fraud vs non-fraud distribution

2. **Data Preprocessing**
   - Scale `Amount` and `Time` using standardization
   - Handle imbalance using:
     - **Under-sampling** (majority class)
     - **Over-sampling** (SMOTE)
     - Or use **class weights**

3. **Model Building**
   - Experiment with classification models:
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - XGBoost / LightGBM
     - Isolation Forest (for anomaly detection)
   - Use stratified train-test split to preserve class ratio

4. **Model Evaluation**
   - Metrics:
     - Precision
     - Recall
     - F1 Score
     - AUC-ROC
     - Confusion Matrix
   - Emphasize recall (detecting as many frauds as possible)

5. **(Optional) Web App Integration**
   - Create a simple web form with **Flask** or **Streamlit**
   - Upload transaction data and predict fraud likelihood

---

## 🛠️ Technologies Used

| Tool / Library     | Purpose                                        |
|--------------------|------------------------------------------------|
| pandas             | Data manipulation                             |
| numpy              | Numerical operations                          |
| scikit-learn       | ML models, preprocessing, evaluation          |
| imbalanced-learn   | SMOTE and resampling techniques               |
| xgboost / lightgbm | High-performance classifiers                  |
| matplotlib / seaborn | Visualizations                             |
| flask / streamlit  | (Optional) Web interface for predictions      |

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv # Dataset
├── notebooks/
│   └── fraud_detection.ipynb # Jupyter notebook with full analysis
├── models/
│   └── model.pkl # Trained ML model
├── app/
│   ├── app.py # Flask or Streamlit app (optional)
│   ├── templates/
│   │   └── index.html # Web interface (optional)
├── outputs/
│   └── metrics/
│   └── plots/
├── requirements.txt
└── README.md
```

---

## 📊 Model Evaluation Metrics

Since fraud detection deals with rare classes, the following metrics are emphasized:

- **Recall**: Correctly detecting fraudulent transactions
- **Precision**: Accuracy of flagged frauds
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Trade-off between true positive rate and false positive rate
- **Confusion Matrix**: Visual representation of classification results

---

## 📸 Output Visualizations

**Confusion Matrices (for different models and sampling strategies):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/e785cb27-3fc7-4e54-85a4-1f8a9a38da94" width="250" alt="Confusion Matrix 1" style="margin: 8px; border-radius: 8px;">
  <img src="https://github.com/user-attachments/assets/9892d8de-0c8e-45a7-af6b-623b8dc080ab" width="250" alt="Confusion Matrix 2" style="margin: 8px; border-radius: 8px;">
  <img src="https://github.com/user-attachments/assets/ff84f0fa-2ae8-4348-9816-c7475b7c51bf" width="250" alt="Confusion Matrix 3" style="margin: 8px; border-radius: 8px;">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/c0dbc3a4-5f1d-483e-ad0e-a2761d7b3f4c" width="250" alt="Confusion Matrix 4" style="margin: 8px; border-radius: 8px;">
  <img src="https://github.com/user-attachments/assets/0566c5e5-7f3a-423e-bea8-0f6fcecec8b9" width="250" alt="Confusion Matrix 5" style="margin: 8px; border-radius: 8px;">
  <img src="https://github.com/user-attachments/assets/a5c8b9d0-4b94-45ef-9c2e-c043cf71eac3" width="250" alt="Confusion Matrix 6" style="margin: 8px; border-radius: 8px;">
</p>

**Other Result Visualizations:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/44587823-95be-4354-b858-86a7c7ce60b2" width="320" alt="AUC ROC Curve" style="margin: 8px; border-radius: 8px;">
  <img src="https://github.com/user-attachments/assets/59f7dd89-c501-4fdc-a962-71eb659d2bd8" width="320" alt="Feature Importance" style="margin: 8px; border-radius: 8px;">
</p>

---

## 📄 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
flask
streamlit
```

---

## 💡 Future Enhancements

🔐 Model Explainability: Use SHAP or LIME to explain why a transaction is flagged  
⏱️ Real-Time Detection: Deploy as an API to classify transactions on the fly  
📈 Dashboard: Visualize detection statistics with Streamlit or Dash  
📊 Time-Based Analysis: Use time series modeling for behavior trends  

---

## 👩‍💻 Author

Developed by Rakhi Yadav  
Let’s connect and collaborate

---
