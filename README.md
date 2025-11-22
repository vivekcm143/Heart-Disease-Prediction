# Heart Disease Prediction App

A Streamlit-based web application that predicts the likelihood of heart disease using multiple machine learning models trained on cardiovascular health parameters.

## Features

- **Multiple ML Models**: Compare predictions from 4 different algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- **Interactive Interface**: Easy-to-use sidebar for entering health parameters
- **Feature Importance**: View Random Forest feature importance analysis
- **Real-time Predictions**: Instant classification results
- **Standardized Input**: Automatic feature scaling and encoding

## Project Structure

```
Heart-Disease-Prediction/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models/
│   ├── heart_disease_lr_model_5.pkl    # Logistic Regression model
│   ├── heart_disease_dt_model_5.pkl    # Decision Tree model
│   ├── heart_disease_rf_model_5.pkl    # Random Forest model
│   ├── heart_disease_svm_model_5.pkl   # SVM model
│   ├── scaler_5.pkl                    # Feature scaler
│   └── columns.pkl                     # Training columns
└── README.md                       # Project documentation
```

## Installation

1. **Clone the repository**
   ```
   git clone https://github.com/vivekcm143/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Verify model files**
   Ensure all model files are present in the `models/` directory:
   - `heart_disease_lr_model_5.pkl`
   - `heart_disease_dt_model_5.pkl`
   - `heart_disease_rf_model_5.pkl`
   - `heart_disease_svm_model_5.pkl`
   - `scaler_5.pkl`
   - `columns.pkl`

## Usage

1. **Run the Streamlit app**
   ```
   streamlit run app.py
   ```

2. **Enter patient information**
   Use the sidebar to input health parameters:
   - Age
   - Sex
   - Chest pain type
   - Blood pressure
   - Cholesterol levels
   - And more...

3. **Get predictions**
   Click the "Predict" button to see predictions from all four models

4. **View insights** (Optional)
   Check "Show feature importance" to see which features most influence the Random Forest model's predictions

## Input Parameters

| Parameter | Description | Range/Options |
|-----------|-------------|---------------|
| **Age** | Patient's age in years | 1-120 |
| **Sex** | Biological sex | Male/Female |
| **cp** | Chest pain type | 0-3 |
| **trestbps** | Resting blood pressure (mm Hg) | 90-200 |
| **chol** | Serum cholesterol (mg/dl) | 100-600 |
| **fbs** | Fasting blood sugar > 120 mg/dl | 0 (No) / 1 (Yes) |
| **restecg** | Resting ECG results | 0-2 |
| **thalach** | Maximum heart rate achieved | 60-220 |
| **exang** | Exercise induced angina | 0 (No) / 1 (Yes) |
| **oldpeak** | ST depression induced by exercise | 0.0-6.0 |
| **slope** | Slope of peak exercise ST segment | 0-2 |
| **ca** | Number of major vessels (0-4) | 0-4 |
| **thal** | Thalassemia | 0-3 |

## Model Details

### Training Dataset
- **Source**: Cleveland Heart Disease Dataset (UCI Machine Learning Repository)
- **Features**: 13 cardiovascular health parameters
- **Target**: Binary classification (Heart Disease / No Heart Disease)

### Machine Learning Models

1. **Logistic Regression**
   - Linear classification model
   - Good baseline performance

2. **Decision Tree**
   - Non-linear decision boundaries
   - Interpretable rules

3. **Random Forest**
   - Ensemble of decision trees
   - Feature importance analysis available
   - Generally highest accuracy

4. **Support Vector Machine (SVM)**
   - Kernel-based classification
   - Effective in high-dimensional spaces

### Preprocessing
- **Feature Scaling**: StandardScaler normalization
- **Encoding**: One-hot encoding for categorical features
- **Feature Engineering**: Dummy variables with `drop_first=True`

## Technical Implementation

### Data Pipeline
```
1. User Input → DataFrame
2. Convert categorical to numeric (sex)
3. One-hot encode multi-class features
4. Align columns with training data
5. Scale features using saved scaler
6. Predict using all 4 models
```

### Feature Importance
The Random Forest model provides feature importance scores showing which health parameters most influence predictions.

## Requirements

- Python 3.8+
- 2GB+ RAM
- Modern web browser

## Troubleshooting

**Issue**: Model files not found
- **Solution**: Ensure all `.pkl` files are in the `models/` directory

**Issue**: Column mismatch error
- **Solution**: Verify `columns.pkl` matches the training configuration

**Issue**: Scaling error
- **Solution**: Check that `scaler_5.pkl` is properly loaded

## Future Enhancements

- [ ] Add probability scores for each prediction
- [ ] Implement model performance metrics dashboard
- [ ] Add data visualization (charts/graphs)
- [ ] Export prediction reports as PDF
- [ ] Add batch prediction from CSV files
- [ ] Deploy to cloud (Heroku/Streamlit Cloud)
- [ ] Add SHAP explanations for predictions

## Dataset Information

The models were trained on the Cleveland Heart Disease dataset, which contains:
- 303 patient records
- 13 clinical features
- Binary classification target (presence/absence of heart disease)

**Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Model Performance

*(Add your model performance metrics here after training)*

Example:
```
Model                Accuracy    Precision    Recall    F1-Score
-------------------------------------------------------------------
Logistic Regression  85%         83%          87%       85%
Decision Tree        78%         76%          80%       78%
Random Forest        88%         86%          90%       88%
SVM                  84%         82%          86%       84%
```

## License

MIT License

## Author

**Vivek C M**
- GitHub: [@vivekcm143](https://github.com/vivekcm143)
- Education: BE in Artificial Intelligence and Machine Learning, VTU Belagavi

## Acknowledgments

- UCI Machine Learning Repository for the Cleveland Heart Disease dataset
- Scikit-learn library for machine learning implementations
- Streamlit for the web framework

## Contact

For questions, improvements, or collaboration:
- Open an issue on GitHub
- Submit a pull request
- Contact via GitHub profile

---

⭐ **Star this repository if you find it helpful!**

## Screenshots

*(Add screenshots of your app here)*

1. **Main Interface**
2. **Prediction Results**
3. **Feature Importance Visualization**
```

***

## Additional Files to Create

### .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Streamlit
.streamlit/

# Jupyter Notebook
.ipynb_checkpoints/

# Models (optional - comment out if you want to include models)
# models/*.pkl

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

***

## Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/vivekcm143/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

***
