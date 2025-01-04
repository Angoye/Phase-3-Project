# Phase_3_House-Prices_Feature_Engineering

## Prices - Advanced Regression Techniques

### Overview

This project leverages the **Ames Housing dataset** to build a regression model aimed at predicting house prices. The focus of this analysis is on **data preprocessing**, **feature engineering**, and developing a robust regression model that accurately estimates the sale price of houses. The dataset includes various features such as numerical, categorical, and ordinal variables, which are essential for understanding and predicting house prices.

### Objectives

The primary objective of this project is to predict the **SalePrice** of houses using a variety of features. Specific goals include:

1. **Predict the SalePrice of houses** based on multiple features:
   - **Numerical Features**: LotArea, GrLivArea, TotalBsmtSF.
   - **Categorical Features**: Neighborhood, GarageType, HouseStyle.
   - **Ordinal Features**: OverallQual, ExterQual.
   
2. **Implement data preprocessing techniques** for machine learning:
   - Clean the dataset by handling missing values.
   - Convert categorical variables into numerical values.
   - Scale or normalize data as needed.
   
3. **Build a regression model** using the processed data and evaluate its performance.

### Key Features

- **Numerical Features**: These are continuous and discrete variables representing property details such as:
  - LotArea (Lot size)
  - GrLivArea (Above ground living area)
  - TotalBsmtSF (Total basement square feet)

- **Categorical Features**: These are descriptive categories for attributes like:
  - Neighborhood (The location of the property)
  - GarageType (Type of garage attached to the house)
  - HouseStyle (Style of the house)

- **Ordinal Features**: These features have a defined order and include:
  - OverallQual (Overall quality of the house)
  - ExterQual (Quality of the exterior materials)

### Project Steps

1. **Problem Understanding**
   - Define the project’s objectives and dataset structure.
   - Identify the key features required for model training and prediction.

2. **Loading the Data**
   - Load the Ames Housing dataset into a Pandas DataFrame for further exploration.
   - Explore the dataset to understand its structure and contents.

3. **Data Preprocessing**
   - **Dropping Irrelevant Columns**: Eliminate any columns that do not contribute to the prediction of house prices.
   - **Handling Missing Values**: Use techniques like `MissingIndicator` and `SimpleImputer` to manage missing values.
   - **Encoding Categorical Features**: Convert categorical variables into numeric form using techniques such as `OrdinalEncoder` and `OneHotEncoder`.
   - **Feature Scaling**: Normalize or standardize numerical features to ensure uniformity in scale for better model performance.

4. **Model Preparation**
   - Ensure that all features are in a numerical format compatible with scikit-learn models.
   - Split the dataset into training and test sets for model evaluation.

5. **Model Training and Evaluation**
   - Use regression models to train on the prepared data.
   - Evaluate the model's performance using metrics such as **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-squared**.

6. **Test Data Preparation**
   - Apply the same preprocessing steps to test data for evaluation purposes.
   - Predict house prices using the trained model on the test set.

### Requirements

To run this project, you need the following libraries:

- **Python 3.x**
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For building and evaluating machine learning models.
- **NumPy**: For numerical operations.

### Installation

Follow these steps to set up and run the project:

1. **Clone the repository** to your local machine:
    ```bash
    git clone https://github.com/your-repository-link
    ```

2. **Install the required packages** using pip:
    ```bash
    pip install pandas scikit-learn numpy
    ```

### Usage

1. **Running the Project**
   - Open the Jupyter Notebook file (`index.ipynb`).
   - Run each cell sequentially to:
     - Load the data.
     - Preprocess the data (handling missing values, encoding categorical variables).
     - Build and evaluate the regression model.
   - Make sure the dataset is available in the same directory as the notebook.

2. **Model Evaluation**
   - After running the notebook, the performance of the model will be displayed using evaluation metrics like MAE, RMSE, and R-squared.
   - You can further adjust preprocessing steps or try different regression models to improve accuracy.
  
# Key Metrics

MAE: 2,758.93

MSE: 17,047,096.68

RMSE: 4,130.74

R-Squared (R2): 0.916

# Feature Importance
The feature importance of the model has been plotted to identify the most influential variables in predicting the sale price.

# Recommendations
Hyperparameter Tuning: It’s recommended to perform hyperparameter tuning (e.g., using GridSearchCV) to further optimize the model.
Cross-Validation: Implement cross-validation to improve the model's robustness and reduce overfitting.

### Acknowledgments

- **Dataset**: This project uses the **Ames Housing dataset**, which is a comprehensive real estate dataset used for regression and predictive modeling tasks.
  - Dataset source: [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

- **Machine Learning Libraries**: The project utilizes popular Python libraries such as Pandas and Scikit-learn for data manipulation, feature engineering, and regression modeling.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

