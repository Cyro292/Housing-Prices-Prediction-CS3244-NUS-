# Housing Prices Prediction - CS3244 Assignment

A machine learning project for predicting Singapore HDB resale flat prices using various regression techniques and feature engineering approaches.

## 📋 Project Overview

This project analyzes Singapore Housing Development Board (HDB) resale flat transaction data spanning from 1990 to present day. The goal is to build accurate machine learning models to predict resale flat prices based on various property characteristics and market factors.

**Modeling Approaches:**
- **Linear Regression**: Baseline model for linear price relationships
- **Decision Trees (Random Forest)**: Tree-based ensemble for non-linear patterns and feature interactions

## 🏗️ Project Structure

```
scripts/
├── data/                           # Dataset files
│   ├── Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv
│   ├── Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv
│   ├── Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv
│   ├── Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv
│   └── Resale flat prices based on registration date from Jan-2017 onwards.csv
├── lib/                            # Core utility modules
│   ├── utils.py                    # Data loading and preprocessing functions
│   └── eval.py                     # Model evaluation metrics
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── base_model.ipynb           # Baseline model implementation
│   ├── feature_tests/             # Individual feature analysis
│   ├── linear_optimisation/       # Linear model optimization
│   ├── pca.ipynb                  # Principal Component Analysis
│   └── random/                    # Random experiments and testing
├── tests/                          # Unit tests
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
└── setup.py                      # Package setup configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Housing-Prices-Prediction-CS3244-NUS-
   ```

2. **Navigate to the scripts directory:**
   ```bash
   cd scripts
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install the package in development mode:
   ```bash
   pip install -e .
   ```

### Quick Start

1. **Run the baseline model:**
   ```bash
   python main.py
   ```

2. **Explore notebooks:**
   ```bash
   jupyter notebook notebooks/
   ```

## 📊 Dataset

The dataset contains Singapore HDB resale flat transactions with the following key features:

- **Temporal**: `month` - Transaction date
- **Location**: `town`, `block`, `street_name` - Geographic information
- **Property**: `flat_type`, `flat_model`, `floor_area_sqm` - Property characteristics
- **Building**: `storey_range`, `lease_commence_date`, `remaining_lease` - Building details
- **Target**: `resale_price` - Sale price (target variable)

**Dataset Statistics:**
- Total records: ~949,000 transactions
- Time span: 1990 - Present
- Features: 10 input features + 1 target variable

## 🔧 Core Features

### Data Processing (`lib/utils.py`)

- **Data Loading**: Load and combine multiple CSV files
- **Data Cleaning**: Handle missing values and data inconsistencies
- **Feature Engineering**: 
  - Convert categorical variables to numerical representations
  - Create derived features (flat age, relative month)
  - Normalize numerical features using min-max scaling
- **PCA Optimization**: Dimensionality reduction for improved performance

### Model Evaluation (`lib/eval.py`)

- **Regression Metrics**: MAE, RMSE, R² score
- **Comprehensive Evaluation**: Automated metric calculation and reporting

### Key Functions

```python
# Load all resale data
X, y = load_all_resale_data()

# Clean and normalize data
X_clean, y_clean = get_cleaned_normalized_data(X, y)

# Split data for training/testing
X_train, X_test, y_train, y_test = get_train_split_data(X_clean, y_clean)

# Evaluate model performance
metrics = get_regression_metrics(y_test, y_pred)
```

## 🤖 Models Implemented

**Linear Regression**
- Baseline model using linear relationships between features
- Fast training and prediction
- Provides interpretable coefficients

**Decision Trees (Random Forest)**
- Tree-based ensemble method for non-linear relationships
- Handles complex feature interactions automatically
- Robust to outliers and mixed data types

## 📈 Model Performance

**Linear Regression (Optimized with PCA + Polynomial Features):**
- **R² Score**: 92.1% (test) / 91.5% (train)
- **Polynomial Degree**: 3 (cubic features)
- **RMSE**: ~63,523 SGD
- **MAE**: ~48,225 SGD

**Decision Trees (Random Forest):**
- **R² Score**: 94% (significantly outperformed linear regression)
- **Better handling of non-linear relationships and feature interactions**

### 🔧 Optimization Techniques Used

**Principal Component Analysis (PCA):**
- Reduced feature dimensionality from ~2,800 to 50 components
- Maintained 87%+ performance while dramatically reducing computational complexity
- Optimal PCA components: 50 (explains sufficient variance)

**Polynomial Feature Engineering:**
- Applied cubic polynomial features (degree 3) to capture non-linear relationships
- Significantly improved linear regression performance from 87% to 92.1% R²
- Captures feature interactions and non-linear patterns in housing data

**Feature Engineering:**
- Categorical encoding (one-hot encoding for town, flat_model, block)
- Numerical normalization (min-max scaling)
- Derived features (flat age, relative month)
- Data cleaning and missing value handling

**Model Optimization:**
- Learning curve analysis to prevent overfitting
- Cross-validation for robust performance estimation
- Hyperparameter tuning for Random Forest

## 🧪 Analysis Notebooks

### Feature Analysis (`notebooks/feature_tests/`)
- Individual feature impact analysis
- Correlation studies
- Feature importance evaluation

### Model Optimization (`notebooks/linear_optimisation/`)
- Hyperparameter tuning
- Learning curve analysis
- Model complexity analysis

### Advanced Techniques (`notebooks/`)
- Principal Component Analysis (PCA)
- Feature selection experiments
- Model comparison studies

## 🛠️ Dependencies

### Core Libraries
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization

### Additional Libraries
- **category_encoders**: Categorical variable encoding
- **scipy**: Scientific computing
- **joblib**: Model persistence

## 📝 Usage Examples

### Basic Model Training
```python
from lib.utils import load_all_resale_data, get_cleaned_normalized_data, get_train_split_data
from lib.eval import get_regression_metrics
from sklearn.linear_model import LinearRegression

# Load and preprocess data
X, y = load_all_resale_data()
X_clean, y_clean = get_cleaned_normalized_data(X, y)
X_train, X_test, y_train, y_test = get_train_split_data(X_clean, y_clean)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
metrics = get_regression_metrics(y_test, y_pred)
print(f"R² Score: {metrics['r2']:.3f}")
```

### Custom Feature Selection
```python
# Load with specific features
X, y = load_all_resale_data(
    include_features=['floor_area_sqm', 'storey_range', 'flat_type'],
    exclude_features=['street_name']
)
```

## 🧪 Testing

Run the test suite:
```bash
cd scripts
python -m pytest tests/
```

## 📚 Course Information

- **Course**: CS3244 Machine Learning (NUS)
- **Assignment**: Assignment 1 - Regression Analysis
- **Focus**: Feature engineering, model optimization, and evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of academic coursework for CS3244 at the National University of Singapore.

## 📞 Contact

For questions or issues related to this project, please open an issue in the repository.

---

**Note**: This project is for educational purposes as part of CS3244 Machine Learning coursework at NUS.
