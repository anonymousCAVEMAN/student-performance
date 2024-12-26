# 🚗 Car Sales Prediction - Data Science Project

Welcome to the **Car Sales Prediction Project**! 🚀 This project aims to predict the maths score of students using **Linear Regression** and advanced regression techniques. We ensure robust analysis by performing extensive EDA, outlier detection, and assumption checks.

---

## 🧰 Project Structure
```
📂 project-root/
├── 📁 artifacts/                     # saved pickle files
├── 📁 data/                          # Dataset directory
├── 📁 src                            
│   ├── 📁 components                 # Data ingestion & preprocessing
         ├── data_ingestion.py        # data reading and splitting
│        ├── data_transformation.py   # feature transformation
│        └── model_trainer.py         # Model training (Linear, Ridge, Lasso, ElasticNet)
|   ├── 📁 pipelines
│        ├── predict_pipeline.py      # predict pipelie which communicates with components to Flask
│   ├── exception.py                  # Inference and prediction
    ├── logger.py                     #logging
│   └── utils.py                      # utils for saveling loading models
├── 📁 template/                      #Flask templates
└── README.md                         # You're here!
```

---

## 🔍 Key Features
- **Car Price Prediction** – Estimate used car sales value with high precision.
- **Linear Regression Pipeline** – Basic to advanced regression models.
- **EDA and Outlier Detection** – Visualizations and anomaly detection.
- **Model Evaluation** – R2 Score, Multicollinearity checks, VIF scores.
- **Streamlit & FastAPI** – Interactive user dashboards and REST API.

---

## 📦 Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart
1. **Clone the Repository**
   ```bash
   git clone https://github.com/username/car-sales-prediction.git
   cd car-sales-prediction
   ```
2. **Data preparation - Model training**
   ```bash
   python src/data_ingestion.py
   ```
3. **Perform EDA**
   ```bash
   python notebooks/eda.ipynb
   ```

6. **Launch Flask App**
   ```bash
   python app.py
   ```

---

## 📊 Model Performance
| Model                 | R2 Score  |
|----------------------|-----------|
| Linear Regression     | 88%       |
| Ridge Regression      | 90%       |
| Lasso Regression      | 91%       |
| Polynomial (2nd Order)| 93%       |
| Polynomial (5th Order)| **95%**   |

---

## 🧪 Assumptions & Analysis
- **No NaN Values** – The dataset is clean with no missing values.
- **Duplicates** – Some duplicate values exist but were retained to reflect real-world conditions.
- **Outliers** – Detected and removed where necessary to improve model performance.
- **Model Assumptions** – Checked for linearity, no autocorrelation, multicollinearity, and homoscedasticity.
- **Polynomial Regression** – Applied up to 5th order with ElasticNet to achieve optimal performance.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


