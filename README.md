# 🚴 Food Delivery Time Predictor (AI Powered)

An **AI-powered food delivery time prediction system** built with **Machine Learning and Streamlit**.  
This project estimates the **expected delivery time** based on factors like **distance, weather, traffic, courier experience, and preparation time**.

The application uses a **Random Forest Regression model** to predict delivery time and provides a **modern interactive UI built with Streamlit**.

---

## 🌟 Features

- 🤖 **AI-Based Prediction**
- 📊 Predict delivery time using multiple real-world factors
- 🌦️ Weather and traffic based estimation
- 🧑‍💼 Courier experience consideration
- ⚡ Instant prediction using Machine Learning
- 🎨 Modern interactive UI using Streamlit
- 📱 Responsive layout with clean design

---

## 🧠 Machine Learning Model

The model used in this project:

- **Algorithm:** Random Forest Regressor
- **Library:** Scikit-learn
- **Model File:** `optimized_rf_model.pkl`

The model predicts the **Estimated Delivery Time (minutes)** using multiple delivery-related factors.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|--------|
| Python | Backend logic |
| Streamlit | Web application |
| Scikit-learn | Machine Learning |
| Pandas | Data processing |
| Pickle | Model storage |
| HTML/CSS | UI Styling |

---

## 📂 Project Structure

```
Food-Delivery-Time-Predictor
│
├── app.py
├── optimized_rf_model.pkl
├── label_encoders.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/food-delivery-time-predictor.git
```

### Go to project directory

```bash
cd food-delivery-time-predictor
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Model Input Features

| Feature | Description |
|-------|-------------|
| Distance_km | Distance between restaurant and customer |
| Weather | Weather condition |
| Traffic_Level | Traffic density |
| Time_of_Day | Morning / Afternoon / Evening / Night |
| Vehicle_Type | Delivery vehicle |
| Preparation_Time_min | Time to prepare food |
| Courier_Experience_yrs | Delivery partner experience |

---

## 🎯 Prediction Output

The model predicts:

```
Estimated Delivery Time (minutes)
```

Displayed in a **beautiful prediction card inside the Streamlit UI**.

---

## 🚀 Future Improvements

- 📍 Google Maps API integration
- 🧠 Deep learning models
- 📊 Delivery analytics dashboard
- 📱 Mobile responsive UI
- 📈 Model explainability (SHAP)

