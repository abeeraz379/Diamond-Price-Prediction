# Diamond-price-prediction 
# By: ABeer Al-Zebda|Machine Learning Engineer
A supervised machine learning project that predicts diamond prices using regression techniques. The model is trained on the classic Diamonds dataset (53,940 samples) and leverages both numerical and categorical features to produce accurate price estimates.

# Overview
Diamond pricing is influenced by multiple physical and quality attributes. This project builds a full ML pipeline — from raw data preprocessing to final prediction — using Linear Regression. A key challenge with this dataset is that diamond prices are right-skewed and Linear Regression has no lower bound constraint, meaning it can freely predict negative values. To solve this, we apply np.log() on the target variable during training, then use np.exp() on predictions to convert them back to real prices — guaranteeing all outputs are always positive.

## Dataset
- link: https://www.kaggle.com/datasets/shivam2503/diamonds
- Rows: 53,940 | Features: 9 | Target: price (USD)

## Features
- Numerical: carat, depth, table, x, y, z  
- Categorical: cut, color, clarity

## Data Dictionary
- price: price in US dollars (\$326--\$18,823)
- carat : weight of the diamond (0.2--5.01)
- cut : quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- color : diamond colour, from J (worst) to D (best)
- clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- x :length in mm (0--10.74)
- y :width in mm (0--58.9)
- z :depth in mm (0--31.8)
- depth :total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- table: width of top of diamond relative to widest point (43--95)

## Key Techniques
- Log-transform on target to fix skewed distribution
- Scaling for numeric features
- OneHotEncoding for categorical features
- Sklearn Pipeline to prevent data leakage
- olumn transformer
- Linear Regression → exp(y_pred) to recover price

## Prediction Model
<img width="247" height="77" alt="image" src="https://github.com/user-attachments/assets/cffb8024-2f73-4362-a7d8-b8e45f302e7c" />

## Actual vs Predicted Prices
<img width="770" height="495" alt="image" src="https://github.com/user-attachments/assets/b3365566-e2e6-4677-8972-8cf999c309d8" />

