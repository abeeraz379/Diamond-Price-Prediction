# Diamond-price-prediction 
# By: ABeer Al-Zebda | Machine Learning Engineer
A supervised machine learning project that predicts diamond prices using advanced regression techniques. Trained on the classic Diamonds dataset (53,940 samples), the project compares Linear Regression vs XGBoost Regressor to deliver highly accurate price estimates.
![1_IT2AEMBYOt5qICLjpuwbyg](https://github.com/user-attachments/assets/5e174ba1-ef26-4199-b0e0-4620690094b5)


# Overview
Diamond pricing depends on physical dimensions, quality grades, and cut precision. This project builds a complete ML pipeline—from data preprocessing to production-ready predictions—comparing Linear Regression and XGBoost Regressor.

## Key Innovation: Diamond prices exhibit right-skewed distribution, and Linear Regression can predict impossible negative prices. Solution: Applied np.log() transformation on target during training, then np.exp() on predictions—guaranteeing positive prices only.

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
  
## Clean The Data 
- There was no Missing Values or Duplicate
- The data was Consistent 

## Explore the data
- Count Plot for Cut and Table Features
<img width="1367" height="507" alt="image" src="https://github.com/user-attachments/assets/29183051-ba32-47c6-8b6f-2460769219b2" />

- Scatter Plot for Carat vs price which indicates a positive realtionship
  
- scatter plot for Table vs price which indicates that almost all of the table feature was between 50 and 70 ith diffrent prices
<img width="1243" height="486" alt="image" src="https://github.com/user-attachments/assets/3ec2e123-b9aa-4772-8a64-4ad56ac7c2d1" />

## Feature Scaling and One Hot Encoder 
- Feature Scaling for numeric features using StandardScaler and pipeline
- One Hot Encoder for categorical Features using OneHotEncoder and pipeline
- Column Transformer 
  
## LinearRegression Model
<img width="247" height="77" alt="image" src="https://github.com/user-attachments/assets/cffb8024-2f73-4362-a7d8-b8e45f302e7c" />

## Evaluate the model
The model evaluation results :
- MAE  = 0.111
- RMSE = 0.158
- R^2  = 0.975
  
## Actual vs Predicted Prices Using LinearRegression
<img width="855" height="506" alt="image" src="https://github.com/user-attachments/assets/8bd35226-3871-439b-9705-c39ee39f2e9f" />

# xgboost Model
<img width="787" height="302" alt="image" src="https://github.com/user-attachments/assets/03439df9-2bc1-4dc8-9945-c0fad1835ed5" />

## Evaluate the model
The model evaluation results :
- MAE  = 0.061
- RMSE = 0.084
- R^2  = 0.993

## Actual vs Predicted Prices Using xgboost
<img width="807" height="503" alt="image" src="https://github.com/user-attachments/assets/7bfd27fa-95eb-4090-9bc0-4901f2667e23" />


## Predict on new data 
I add a new data to test the model and it shows a realistic and logical results for price with a closer value for the real data

## Conclusion
The project successfully built a production-ready diamond pricing pipeline, with **XGBoost outperforming Linear Regression** (R² 0.993 vs 0.975). 

**Key Technical Achievement**: Solved negative price prediction using **log-transform + exp inverse**—eliminating impossible outputs while handling skewed distribution perfectly.

**XGBoost Excellence**: Achieved **MAE 0.061** and **RMSE 0.084**, demonstrating superior pattern recognition across 53k+ samples.

**Business Impact**: Accurate pricing model enables optimal pricing strategies, inventory valuation, and competitive market positioning.

**Future Enhancements**:
- **Feature Engineering**: Volume (`x*y*z`), aspect ratios, symmetry scores
- **Ensemble**: Stacking XGBoost + LightGBM
- **Hyperparameter Optimization**: Bayesian optimization for peak performance

This pipeline transforms raw gemstone data into actionable business intelligence!
