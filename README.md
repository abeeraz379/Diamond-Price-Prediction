# Diamond-price-prediction 
# By: ABeer Al-Zebda | Machine Learning Engineer
A supervised machine learning project that predicts diamond prices using regression techniques. The model is trained on the classic Diamonds dataset (53,940 samples) and leverages both numerical and categorical features to produce accurate price estimates.
![1_IT2AEMBYOt5qICLjpuwbyg](https://github.com/user-attachments/assets/5e174ba1-ef26-4199-b0e0-4620690094b5)


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
  
## Prediction Model
<img width="247" height="77" alt="image" src="https://github.com/user-attachments/assets/cffb8024-2f73-4362-a7d8-b8e45f302e7c" />

## Evaluate the model
The model evaluation results :
- MAE  = 0.111
- RMSE = 0.158
- R^2  = 0.975
  
## Actual vs Predicted Prices
<img width="855" height="506" alt="image" src="https://github.com/user-attachments/assets/8bd35226-3871-439b-9705-c39ee39f2e9f" />

## Predict on new data 
I add a new data to test the model and it shows a realistic and logical results for price with a closer value for the real data

## Conclusion
Through analyzing the model's coefficients, we can conclude that each feature has an effect on the target variable (price), but to varying degrees. Some features influence the price positively, while others have a negative impact. The predicted prices are close to the actual prices, which indicates that the model is performing well and has learned the underlying patterns in the data. The biggest challenge I faced during model building was the appearance of negative predicted prices, which is physically impossible for diamond pricing. After researching the issue, I found that applying np.log() on the target variable during training and then using np.exp() to convert predictions back to real prices was an effective solution. This approach eliminated all negative predictions and resulted in significantly better results. Finally, while the model performs reasonably well, there is still room for improvement. Feature engineering could be a strong next step — creating new meaningful features from the existing ones such as volume (x * y * z) or carat-to-depth ratio may help the model capture more complex relationships and improve prediction accuracy.
