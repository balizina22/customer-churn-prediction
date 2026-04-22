# customer-churn-prediction

# 📊 Customer Churn Prediction

## 🎯 Objective
Predict customer churn using Machine Learning and Deep Learning models.

---

## 📂 Project Structure
- `notebooks/` → EDA, Data Preparation, Modeling
- `data/` → Clean dataset
- `models/` → Trained models
- `images/` → Visualizations
- `app.py` → Streamlit dashboard

---

## ⚙️ Models Used
- Logistic Regression
- Random Forest
- Gradient Boosting
- MLP (Deep Learning)

---

## 📈 Evaluation Metrics
Due to class imbalance:
- Recall
- F1-score
- ROC-AUC

---

## 🧠 Key Insights
- Payment failures increase churn risk
- Low NPS strongly correlated with churn
- Long-tenure customers are less likely to churn

---

## 🖥️ Run the App
```bash
streamlit run app.py
