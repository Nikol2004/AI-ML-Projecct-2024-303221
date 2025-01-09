# AI and ML Project 2024-2025

# Project 4 - AEROPOLIS

#### Group ID: 15

### Group Members:

* Nikol Tushaj (303221)
* Rajla Çulli (297601)
* Giulio Carbone (290721)

## INTRODUCTION

In the futuristic city of Aeropolis, autonomous delivery drones are essential to ensure fast and efficient delivery of goods across the sprawling metropolis. Each drone's performance is evaluated based on how much cargo it can deliver per flight. However, many factors influence its performance, from weather conditions to the type of terrain it navigates. To optimize drone performance, data scientists are tasjed with predicting the cargo capacity per flight based on various environmental and operational factors.

### LIBRARIES
The main python libraries we used for the project are:

* **`pandas`**: to manipulate the dataset;
* **`matplotlib.pyplot`**: to plot graphs;
* **`numpy`**: to perform mathematical operations;
* **`seaborn`**: to plot graphs;
* **`time`**: to track execution time;
* **`math`**: to perform mathematical operations;
* **`warnings`**: to get rid of unnecessary warnings;
* **`sklearn`**: to build the models and evaluate their performance;


## The Steps taken: 

* EDA
* Cleaning the Dataset
* Feature Selection
* Splitting into training and test data
* Model Building
* Feature Importance






## 1. EDA

### Data Overview

* The dataset contains 1,000,000 rows and 20 columns.
* The target **`Cargo_Capacity_kg`**, a numerical value predicting the drone's cargo capacity.
* Features include environmental factors (e.g., **`Air_Temperature_Celsius`**, **`Weather_Status`**), and categorical data (e.g., **` Package_Type`**, **`Market_Region`**).

### Data Integrity

* The dataset has 10% missing values across all columns.
* Key metrics were calculated using **`.info()`**, **`.nunique()`**, and **`.duplicated()`**, confirming no duplicates but highlighting missing data for further handling.

### Descriptive Statistics

### - Categorical Values

* Balanced distributions for categories such as **`Weather_Status`**, **`Market_Region`**, and **`Package_Tpye`**.

* Represented visually through pie charts for clear interpretation.

<img width="658" alt="1  Categorical Values Distribution" src="https://github.com/user-attachments/assets/dc93d0fe-54f1-408d-b27b-fb49ac91d748"/>

### - Numerical Values

* **`Cargo_Capacity_kg`** has a mean of 4.65 kg with a standard deviation of 1.69 kg, indicating moderate variability.
* Some negative values identified as anomalies.

* Histogram plots reveal bell-shaped distributions for some variables, while others, like **`Cleaning_Liquid_Usage_liters`**, show skewness.

<img width="656" alt="2  Numerical Values Distribution" src="https://github.com/user-attachments/assets/acd12211-ba8c-48ed-bb42-92a1b2b7e0aa"/>

### Handling missing values of the dependent variable

In a regression problem, the target variable (**`Cargo_Capacity_kg`**) is essential for training and evaluation. Rows with missing target values are removed because:

1. <b>Model Training</b>: The model cannot learn without target values, as it needs them to calculate errors and adjust weights.
2. <b>Evaluation</b>: Missing target values make it impossible to compare predictions and measure model performance.
3. <b>Data Integrity</b>: Retaining these rows adds noise and unnecessary complexity without contributing to the model.

#### Dropping rows where the dependent value is missing

**`Cargo_Capacity_kg missing values in the original dataset: 100233`**

**`Cargo_Capacity_kg missing values after dropping the rows: 0`**

**`We removed the 100233 rows and now there are 899767`**







## 2. Cleaning the dataset

### Encoding Categorical Values

* The dataset contained several categorical columns such as **`Weather_Status`**, **`Package_Type`**, and **`Vertical_Landing`**.

* These columns were mapped to numerical values to facilitate model training.
	For example:
	* **`Weather_Status`**: {'Cloudy': 0, 'Sunny': 1, 'Rainy': 2}
	* **`Package_Type`**: {'Maize': 0, 'Cotton': 1, 'Barley': 2}

 * **`NaN`** values were preserved during encoding for later imputation.
 * This transformation ensured that categorical variables were effectively handled, maintaining the dataset's integrity for modeling.

### Mapping and Validation

* Custom mappings were applied to each categorical column, and the changes were validated by displaying unique values before and after the mapping process.
* Example:
	* Before Mapping: ['Cloudy', 'Sunny', 'Rainy', nan]
   	* After Mapping: [0.0, 1.0, 2.0, nan]
 
* This validation process confirmed the accuracy and consistency of the transformations.

### Dataset Overview after encoding

* After processing, all columns were converted to numeric types, including previously categorical columns.
* ### Finding the correlation between the independent values and the target value

We compute the pairwise correlation between all the columns using **`.corr()`** and then visualize it in a heatmap using **`seaborn.heatmap`**.

<img width="659" alt="3  Correlatio Matrix of Variables" src="https://github.com/user-attachments/assets/29768730-6217-4055-8c03-b0a7e82ea456" />

* The heatmap shows correlations between all variables. For example, **`Wind_Speed_kmph`** has a strong positive correlation (**`0.76`**) with the target **`Cargo_Capacity_kg`**. 

* On the other hand, features like **`Vertical_Max_Speed`** and **`Market_Region`** show near-zero or negative correlations.

* Additionally, the lack of strong correlations among most features suggests low multicollinearity, which is advantageous for regression models, as it helps to avoid redundancy and ensures better feature contributions to predictions.

### Dropping rows with missing values in important features

**Purpose**

To ensure data integrity, rows containing missing values for critical features are dropped. This step minimizes the risk of biases caused by incomplete data during model training.

### Dropping Irrelevant or Negatively Correlated Columns

This step eliminates columns that do not contribute significantly to predicting **`Cargo_Capacity_kg`**. By focusing on highly relevant features, we simplify the dataset, improve computational efficiency, enhance model interpretability. 

### Removing Skewness

From the descriptive statistics before, we saw that one of the numerical values had potential of being skewed, therefore we have to remove it.

<img width="657" alt="4  Distribution of Cleaning_Liquid_Usage_liters" src="https://github.com/user-attachments/assets/b828b857-c4ab-4dbb-bd6b-357304523d2f" />

* The original distribution of **`Cleaning_Liquid_Usage_liters`** is right-skewed, indicating that most values are clustered near zero. After the log transformation, the distribution becomes more symmetric, which can improve model performance by ensuring that the feature follows a normal distribution.

We transform that column by using the **`np.log1p`**, which applies a logarithmic transformation (**`log(1+x)`**) to each value.


<img width="654" alt="5  Log_Transformed Distribution of Cleaning_Liquid_Usage_liters" src="https://github.com/user-attachments/assets/e95bd482-65d2-40e1-8b93-1e273697a7d1" />

* The graph displays the distribution of the log-transformed **`Cleaning_Liquid_Usage_liters`** values. The transformation successfully reduces skewness, making the data more symmetrical and closer to a normal distribution.

* This adjustment improves the suitability of the data for machine learning models that assume normally distributed input features.

* The peak near the center indicates the most frequent log-transformed values, while the tails on either side show lower frequencies of extreme values, effectively reducing their impact on modeling.


### Outlier Identification

**What are outliers?**

Outliers are data points that significantly differ from the majority of the dataset, often lying far outside the expected range. They can distort statistical analyses and model performance if not addressed.

We generate boxplots for each column in the dataset to identify potential outliers visually.

<img width="655" alt="6  Boxplots with outliers" src="https://github.com/user-attachments/assets/64dacd98-a9b9-47d4-884c-69969aa3d2e3" />

Several features, such as **`Flight Hours`** and **`Cleaning_Liquid_Usage_liters`**, show significant outliers. Identifying these helps decide whether to remove or handle them, depending on their impact on model performance.

### Removing Outliers

The function **`remove_outliers_iqr`** applies the IQR method to filter out outliers.

* It calculates the first quartile (Q1), third quartile (Q3), ad the IQR (Q3 - Q1) for each column.

* Using the lower and upper bounds defined as **`Q1 - 1.5 * IQR`** ad **`Q3 + 1.5 * IQR`**, it removes rows with values outside this range for each column.

**`Original DataFrame shape: (235937, 14)`**
**`Cleaned DataFrame shape: (185910, 14)`**

<img width="654" alt="7  Boxplots without outliers" src="https://github.com/user-attachments/assets/4bb2bdf2-ba5f-46fc-8d17-852a689a8814" />

* The new boxplots show that most extreme values have been removed, and the distributions are now more compact.

* The absence of extreme values suggests the dataset is now better suited for analysis and modeling.


### Downsampling

**What is downsampling?** 

Downsampling is the process of reducing the size of a dataset by randomly selecting a subset of rows while maintaining the overall structure and distribution of the data. This is typically done to make the dataset smaller, more manageable, and computationally efficient for machine learning tasks.

**Why downsample the Dataset?**

In the more filtered dataset, the shape indicates a very large number of rows (235k). Processing such a large dataset can be computationally expensive and time-consuming, especially for iterative tasks like hyperparameter tuning or model evaluation. Downsampling reduces the dataset size to a more manageable level, allowing quicker analysis and experimentation. Additionally, downsampling preserves the distribution of the target variable, ensuring that insights gained during analysis or modeling remain representative of the original dataset.

### Distribution of the target value before sampling

<img width="656" alt="8  Distribution of Cargo_Capacity_kg before downsampling" src="https://github.com/user-attachments/assets/389abf3c-45b3-48c0-b2c4-25a4054499d0" />

* The distribution of **`Cargo_Capacity_kg`** appears symmetrical and bell-shaped, consistent with a normal distribution.

### Distribution of the target value after sampling

<img width="655" alt="9  Distribution of Cargo_Capacity_kg after downsampling" src="https://github.com/user-attachments/assets/2fdee265-fa23-4279-b8ca-0d6fe8a076e6" />

* After downsampling, the shape of the distribution is the same and remains unchanged, confirming that random sampling preserved the dataset's statistical properties.
— vetem kte shto posht dyshit







## 3. Feature Selection

### Splitting Discrete and Continuous Values

* The dataset is split into discrete and continuous variables. Numerical columns are identified using **`select_dtypes`** for data types **`float64`** and **`int64`**.
* A threshold of 10 unique values determines whether a column is classified as discrete or continuous.

### Normalizing Continuous Features

* Continuous variables are scaled using **`StandardScaler`** to normalize the data. Scaling adjusts the features to have a mean of 0 and a standard deviation of 1.
* This ensures that featurs are on the same scale, preventing dominance of any single feature during model training.

### K-Nearest Neighbors (KNN)

**What is KNN and how does it work?**

K-Nearest Neighbors (KNN) in regression models works by predicting the target value of a given data point based on the average (or sometimes weighted average) of the target values of its k-nearest neighbors in the feature space. The neighbors are identified based on a distance metric, such as Euclidian distance, which measures similarity between data points.

**Why KNN?**

We decided to use KNN imputation because it effectively estimates missing values by considering the similarity between rows. This method works well for both discrete and continuous variables, preserving the structure and relationships within the dataset. Additionally, KNN ensures that missing values are replaced with contextually relevant data derived from similar rows, leading to more accurate and meaningful imputations.

### Imputing KNN

* Missing values in continuous variables are imputed as-is, while discrete variables are rounded to maintain their categorical nature.
* This ensures the dataset is complete and ready for subsequent modeling.


<img width="656" alt="10  Distribution of values after using KNN" src="https://github.com/user-attachments/assets/bd07481e-305d-4e9a-ad0b-e33cf7659309" />

**Interpretation**

* The imputation process did not distort the underlying distributions. Instead, it complemented the existing structure by estimating plausible values for missing data. 

* This approach minimizes the risk of bias and ensures the data is ready for further preprocessing or modeling.








## 4. Splitting into training and test data

**Purpose**

The dataset is split into training and test sets using **`train_test_split`**. This ensures that the model is trained on one subset of data and evaluated on another to prevent overfitting and assess generalization.

The **`X`** variables represent features, while **`y`** is the target variable.

The **`test_size=0.2`** specifies that 20% of the data is reserved for testing, and **`random_state=42`** ensures reproducibility. 

* Training set size: 14,872
* Test set size: 3,719
* Both splits maintain 13 features, confirming the intergrity of the division.

### Distribution of Values in Training and Test sets

**Purpose**

Visualizing the distribution of features in the trianing and test sets using Kernel Density Estimation (KDE) plots ensures that both subsets are statistically similar and unbiased.

<img width="580" alt="11  Distribution of values in the training and tets data comparison" src="https://github.com/user-attachments/assets/eb82aade-9d98-4473-86fc-54babad95517" />

* The KDE plots indicate that the feature distributions in both sets align closely, suggesting an even split without significat sampling bias. 

* This ensures that the models trained on the training data will generalize well to the test data. 

* However, features like **`Vertical_Landing`** and **`Terrain_type`** show distinct peaks, reflecting categorical distributions, while others like **`Air_Temperature_Celsius`** are continuous and symmetric.





## 5. Model Building

### A Regression Problem

This task was approached as a regression problem because the target variable, **Cargo_Capacity_kg**, is a continuous numerical value.

Regression models are specifically designed to predict continuous outcomes by learning the relationships between the input features and the target variable. Unlike classification, which deals with discrete categories, regression enables the prediction of a wide range of possible values, making it suitable for estimating quantities such as weight, price, or, in this case, the cargo caapacity of autonomous delivery drones. 

This choice aligns with the dataset structure and the objective of providing accurate numerical predictions, essential for operational and logistical planning in drone delivery systems.

### Models to compare

1. **`Linear Regression`**

* A basic regression model that assumes a linear relationship between the input features and the target variable. It fits a straight line to minimize the residual sum of squares between observed and predicted values.

2. **`Random Forest Regressor`**

* An ensemble learning method that build multiple decision trees and average their predictions to improve accuracy and reduce overfitting. It is robust to outliers and captures complex relationships.

3. **`Gradient Boosting Regressor`**

* An iterative ensemble technique that builds trees sequentially, with each tree correcting errors of the previous ones. It focueses on minimizing the loss function, making it effective for complex datasets.

4. **`K-Nearest Neighbors Regressor`**

* A non-parametric model that predicts the target value of a data point by averaging the values of its k nearest neighbors in the feature space. 

5. **`Support Vector Regressor`**

* A regression model that uses the concept of support vectors and hyperplanes. It tries to fit the data within a margin of tolerance (epsilon) while minimizing errors outside this margin. It is effective for datasets with a high dimensional feature space.

### Testing the models

These models were trained and evaluated. Key evaluation metrics included:

**`MSE (Mean Squared Error)`**: Measures the average squared differences between actual and predicted values.

**`MAE (Mean Absolute Error)`**: Measures the average absolute differences.

**`R^2 (Coefficient of Determination)`**: Indicates the proportion of the variance explained by the model.

**`Runtime`**: Captures the time each mocel takes for training and predictions.

<img width="657" alt="16  SVR curve and error plot" src="https://github.com/user-attachments/assets/d464fc49-b82a-416f-ac58-e8b485de1c98" />
<img width="661" alt="15  K-Nearest Neighbors curve and error plot" src="https://github.com/user-attachments/assets/e844096b-c493-46e2-b494-6e392d68b336" />
<img width="660" alt="14  Gradient Boosting curve and error plot" src="https://github.com/user-attachments/assets/f14c8f80-7477-4679-bdf6-e341451cc62b" />
<img width="657" alt="13  Random Forest curve and error plot" src="https://github.com/user-attachments/assets/7e62763c-5c5f-441d-a032-e305a29439cd" />
<img width="660" alt="12  Linear Regression curve and error plot" src="https://github.com/user-attachments/assets/4e289bdb-8354-47b8-b590-8ab1cc61b379" />

1. **Learning Curves:**

* A steep training curve suggests overfitting if it is much better than validation.
* Gradient Boosting and Linear Regression show relatively balanced training/validation curves, indicating good generalization.
* KNN shows high overfitting, with a significant gap between training and validation R^2.

2. **Prediction Error Plots:**

* Models with points close to the diagonal red line perform better.
* Points scattered far from the line suggests worse predictions.

From the results of the metrics we concluded that: 
* **`Linear Regression`** performs best in terms of R^2 (0.9123) and runtime (0.046) seconds, making it efficient and effective.

* **`Gradient Boosting`** has slightly lower R^2 but is robust for capturing nonlinear relationships.

* **`SVR`** performs comparably but takes londer to run.

* **`Random Forest`** performs moderately but has higher computational costs.

* **`KNN`** shows the worst performance due to overfitting, with significantly lower R^2.

### Visualizing the linear relationship of the values with the target value

<img width="657" alt="17  Linear Relations" src="https://github.com/user-attachments/assets/7c4018b5-e848-4296-b45f-09c3726cacd5" />

The scatterplots show that most features lack a strong linear relationship with the target variable, as the points are scattered uniformly. Specifically:

1. *Air Temperature*, *Flight Hours*, *Cleaning Liquid Usage*, *Autopilot Quality*, and *Route Optimization*: The data points are distributed without any discernible pattern, suggesting no clear linear correlation with the target.

2. *Wind Speed (kmph)*: This feature shows a visible pattern where the target value increases with wind speed, indicating a potential linear or non-linear correlation.

This confirms that Linear Regression is not suitable for accurately predicting the target variable in this dataset and will be used only as a baseline model.

### Choosing the best models

**Gradient Boosting**

*Why was it chosen?*

Gradient Boosting is a powerful ensemble learning technique that build models sequentially, minimizing errors at each step. It was chosen because:

* It captures complex, non-linear relationships in the data.

* The model has shown robust performance with low MSE and high R^3 scores during cross-validation.

* It is less prone to overfitting than Random Forest in some scenarios due to its iterative training process.

**Support Vector Regressor (SVR)**

*Why was it chosen?*

SVR uses kernel function to model non-linear relationships effectively. It was chosen because:

* It is capable of finding a balance between bias and variance by defining margins for the predictions.

* The model performed well in terms of accuracy ad R^2 scores, proving its suitability for this dataset.

* It can handle outliers better than simpler regression techniques.

**Linear Regression (as a baseline)**

*Why was it chosen?*

Linear Regression was included as a baseline model for comparison purposes. It was chosen because:

* It is straightforward, interpretable, and computationally efficient.

* Despite its simplicity, it provides a benchmark to evaluate the performance of more complex models.

* The scatterplots demonstrated that most features lack linear relationships with the target, confirming its limited utility but making it ideal for baseline evaluation.

##### Why not the others?

* *Random Forest*: While Random Forest is a strong model, it requires more computational resources and showed slightly inferior performance compared to Gradient Boosting and SVR.

* *K-Nearest Neighbors(KNN)*: KNN performed poorly with the highest MSE and lowest R^2, indicating its inability to capture the complexities of the dataset. Additionally, its performace degrades with high-dimensional data.

### Hyperparameter Tuning for Gradient Boosting Rergession

**1. Hyperparameter Grid Definition**
The hyperparameter grid was defined to optimize the performance of the Gradient Boosting Regressor. The parameters included:

* **`n_estimators`**: This represents the number of boosting stages (or trees). A higher number allows the model to learn more complex patterns but can lead to overfitting if too large.

* **`learning_rate`**: Conrols the contribution of each tree to the final prediction. Smaller values reduce overfitting and require more estimators for good performance.

* **`max_depth`**: Limits the maximum depth of individual trees, controlling how complex each tree can be. Shallower trees help prevent overfitting.

* **`min_samples_split`**: The minimum umber of samples required to split an internal node. Larger values result in less complex trees.

* **`min_samples_leaf`**: The minimum number of samples required to be at a leaf node. Higher values make the trees more robust by reducing overfitting.

**2. Initializing RandomizedSearchCV**

The hyperparameter tuning was concluded using **`RandomizedSearchCV`** with the following: 

* **`estimator`**: Specifies the model, in this case, GradientBoostingRegressor.

* **`param_distributions`**: The hyperparameter grid defined earlier (param_grid) that included values to sample during the search.

* **`n_iter`**: The number of random combinations of hyperparameters to try (100 iterations here).

* **`scoring='r2'`**: The performance metric used is the R^2 score.

* **`cv=5`**: Performs 5-fold cross-validation for each parameter combination.

* **`random_state=42`**: Ensures reproducibility of the results.

* **`n_jobs=-1`**: Utilizes all available CPUs for computation.

* **`verbose=1`**: Controls the amount of output during the execution.

**3. Model Training and Selection**

The best hyperparameters were identified by fitting the model to the training data. These parameters are:
	
* **`n_estimators`**: 200
 	
* **`min_samples_split`**: 15

* **`min_samples_leaf`**: 6

* **`max_dept`**: 3

* **`learning_rate`**: 0.05

**4. Evaluation on Test Data**

The tuned Gradient Boosting model was evaluated on the test set with the following metrics:

* **`MSE`**: 0.0882
* **`MAE`**: 0.2381
* **`R^2 Score`**: 0.9111

The model's test set results validate its effectiveness and confirm that the chosen hyperparameters generalize well.

### Hyperparameter Tuning for SVR

**1. Parameter Grid Definition**

The **`param_grid_svr`** dictionary defines the hyperparameter space for tuning the SVR.

*The parameters chosen*

* **`C`**: Regularization parameter. A smaller value of C allows for a larger margin, but may increase bias, while a larger value of C reduces margin for a better fit but may risk overfitting. Values like 0.1, 1, 10, 100 are tested.

* **`epsilon`**: Defines the margin of tolerance where no penalty is given in the training loss function. Smaller values are more sensitive to deviations from actual values. Values like 0.001, 0.01, 0.1, 1 are tested.

* **`kernel`**: Specifies the kernel type to be used in the algorithm
    * *linear*: A linear kernel assumes linear relationships in the data
    * *rbf*: Radial Basis Function kernel is flexible and works well with non-linear data.
    * *poly*: Polynomial kernel can capture complex relationships.

**2. Hyperparameter Tuning Process**

This initializes a hyperparameter seach over the SVR model:

* **`estimator=SVR()`**: Defines the base model (SVR) to tune.

* **`param_distributions=param_grid_svr`**: Uses the grid of parameters defined earlier.

* **`n_iter=20`**: The search will test 20 random combinations of the parameter grid.

* **`scoring='r2'`**: The performance metric used is the R^2 score.

* **`cv=5`**: Performs 5-fold cross-validation for each parameter combination.

* **`random_state=42`**: Ensures reproducibility of the results.

* **`n_jobs=-1`**: Utilizes all available CPUs for computation.

* **`verbose=2`**: Provides detailed output during the search.

**3. Model fitting and results**

The model underwent 20 iterations with 5-fold cross-validation for each hyperparameter combination. The best parameters indicate: 

* **`Kernel`**: 'linear' was optimal, meaning the relationship between predictors and the target is sufficiently linear.

* **`C:`** 1, which provides a balance between margin size and fitting accuracy.

* **`Epsilon`**: 0.01, a small margin of tolerance, indicating the model is sensitive to deviations in predictions.

The cross-validation R^2 score of 0.913 shows the model generalizes well on unseen data, with high accuracy.

**4. Evaluation on Test Set**

The best SVR model was evaluated on the test set:


* MSE: **`0.0869`** indicates low average squared error.

* MAE: **`0.2355`** suggests predictions are off by about 0.2355 units on average.

* R^2: **`0.9124`** confirms the model explains 91.24% of the variance in the target variable.

### Evaluation for Linear Regression 

Linear Regression is a baseline model without hyperparameter tuning making it straightforward and computationally efficient to evaluate.

* The Linear Regression model achieved comparable performance to the other models, with an MSE **`0.0870`**, MAE of **`0.2357`**, and an R^2 of **`0.9123`**. 

* This suggests that even without advanced techniques, linear regression is a strong baseline for this dataset.

### Comparison of Models

<img width="655" alt="18  Model Performance Comparison" src="https://github.com/user-attachments/assets/e5851e89-0300-4df6-8242-9fefca8e401d" />

* SVR has the lowest MSE and MAE, confirming it makes the smallest prediction errors among the three models. It also has the highest R^2, indicating it explains the variance in the target variable better than the others.

* Linear Regression performs almost as well as SVR, with comparable R^2 and slightly higher error metrics. 

* Gradient Boosting, while effective, shows slightly higher MSE and MAE, suggesting it may not generalize as well as the other models for this dataset.

### Residual Plots 

Residual plots help evaluate how well a model's predictions align with the actual data. They plot the residuals (differences between predicted and actual values) against the predicted values. Residuals should ideally have no discernible pattern and be centered around zero.

This indicates that the model captures the data effectively and that errors are randomly distributed. Using residual plots is essential to check for non-linearity, and any systematic bias in the predictions.

<img width="658" alt="21  Residual Plot for Linear Regression" src="https://github.com/user-attachments/assets/ab3e4071-7a2a-4a5a-ace8-a1afd037a583" />
<img width="656" alt="20  Residual Plot for SVR" src="https://github.com/user-attachments/assets/725a6d4a-f010-4019-8167-6ae8cbeec16d" />
<img width="659" alt="19  Residual Plot for Gradient Boosting" src="https://github.com/user-attachments/assets/ca5f9a0e-b2db-48b4-99c9-e0e4c6c8ab3e" />

1. *Gradient Boosting*: The residuals are distributed relatively evenly around zero, with no visible pattern. This indicates that the model captures the data structure well, though some variance in prediction errors is noticeable.

2. *SVR*: The SVR residual plot also shows residuals scattered evenly around zero with minimal structure or pattern, suggesting that the model predictions align well with the actual data. The dispersion of residuals is slightly more concentrated compared to Gradient Boosting, indicating consistent prediction behavior.

3. *Linear Regression*: The residuals for linear regression are evenly distributed around zero, similar to the other two models. However, since linear regression assumes a linear relationship, any slight deviations might indicate that this assumption may not fully capture the complexity of the data.

Overall, all three models show acceptable residual distributions, but Gradient Boosting and SVR may handle complex data patterns slightly better than Linear Regression due to heir more advanceed architectures.

### Training and evaluating the models after Hyperparameter Tuning

**1. Model Initialization**

* Defined models include Gradient Boosting, Support Vector Regressor (SVR), and Linear Regressor.
* Gradient Boosting and SVR models are configured with the best hyperparameters obtained during tuning.

**2. Performance Metrics**

* For each model, performance is evaluated on the test dataset using key metrics.
* Each model is iteratively trained and evaluated:
	* Predictions are made on the test dataset.
   	* Metrics and runtime are calculated and logged.
* Final results are stored in a dictionary for comparison across models.

**3. Learning Curves for each**

<img width="657" alt="24  Learning Curve with Best Parameters LR" src="https://github.com/user-attachments/assets/915db652-32a6-4048-8fcd-3eb95eb90058" />
<img width="652" alt="23  Learning Curve with Best Parameters SVR" src="https://github.com/user-attachments/assets/2812faf9-6891-41a1-bd95-893b1e2ba7b7" />
<img width="656" alt="22  Learning Curve with Best Parameters GB" src="https://github.com/user-attachments/assets/2f5bf035-7e67-4e1d-b6cf-cbbc9c60e154" />

**`Learning Curves`**

1. Gradient Boosting

    * The training accuracy decreases slightly as the training size increases, suggesting the model generalizes well.
    * The validation accuracy remains consistent with a slight improvement as more data is used, indicating low overfitting.

2. SVR

    * Similar to Gradient Boosting, the training accuracy decreases as the training size increases.
    * Validation accuracy shows marginal improvement with more datam indicating stable performance and minimal overfitting.

3. Linear Regression

    * The training accuracy is slightly lower than that of Gradient Boosting and SVR, reflecting the simpler nature of the model.
    * Validation accuracy remains steady and comparable to the training accuracy, indicating good generalization.

**`Performance Metrics`**

* *Gradient Boosting*: Offers strong performance with an R^2 of 0.9111 and low errors but has a slightly longer runtime compared to Linear Regression.

* *SVR*: Achieves the highest R^2 and lowest error metrics but has a significantly higher runtime.

* *Linear Regression*: Has a marginally lower R^2 but compensates with the fastest runtime, making it suitable for time-sensitive tasks.

**Why so small differences?**

The minimal differences observed in the learning curves and evaluation metrics across models can be attributed to the homogeneity of the dataset. Specifically, equally distributed categorical variables simplify paterns in the data, reducing the need for models to adapt to challenging or unique scenarios.

As a result, hyperparameter tuning has little impact, as the models can already achieve optimal or near-optimal performance on the dataset.







## 6. Feature Importance

### Feature Importance for Gradient Boosting


In this step, we analyze the importance of features in the Gradient Boosting model. Feature importance helps us understand which variables contribute most significantly to the prediction.

* The feature importance scores are extracted and stored in a pandas DataFrame.
* The **`plt.barh`** function is used to plot the bar chart, ensuring clarity of feature ranking.
* Features are plotted against their respective importance scores, providing insights into which features have the most impact on the model's predictions.

<img width="656" alt="25  Feature Importance for GB" src="https://github.com/user-attachments/assets/7faecf00-749a-42cf-96f6-c0bf6bf595f5" />

The bar chart shows that Wind_Speed_kmph dominates feature importance with a score of 0.641, followed by Quantum_Battery and Flight_Duration_Minutes. All other features contribute less.

Reasons for this behavior:

1. Scaling and Preprocessing:

    * Features were scaled, and outliers were handled appropriately, ensuring that no feature had an unfair advantage due to raw magnitude.

    * Despite scaling, Gradient Boosting inherently prioritizes features that strongly reduce impurity in the decision trees, which may explain the overwhelming importance assigned to a few features.

2. Model Characteristics:

    * Gradient Boosting is a tree-based model, and it may focus heavily on specific features that provide the most split gain while downplaying others.

    * Such models can exaggerate the perceived importance of a single feature due to interactions or correlations, even if preprocessing was performed correctly.


### Feature Importance for Linear Regression

For the Linear Regression model, feature importance is derived using the coefficients assigned to each feature. These coefficients represent the weight or contribution of each feature to the target variable.

<img width="657" alt="26  Feature Importance for LR" src="https://github.com/user-attachments/assets/354e95a0-39aa-4ff4-9fdc-d5c6ad0b54b9" />

* The graph illustrates the relative importance of features in the Linear Regression model. Quantum_Battery and Wind_Speed_kmph have the highest importance, followed by Flight_Duration_Minutes. 

* Linear Regression assigns importance based on the magnitude of feature coefficients. Features with higher coefficients exert greater influence on the predictions.

* The results make sense for Linear Regression since the importance reflects how directly the features correlate with the target variable. Features with near-zero coefficients, such as Package_Type and Weather_Status, have minimal or no linear correlation with the target.

### Comparison for Linear Regression and Gradient Boosting Feature Importance 

The Gradient Boosting model relies on iterative decision tree splits, which account for feature interactions and nonlinear patterns. As a result, **`Wind_Speed_kmph`** dominates the importance in Gradient Boosting, likely due to its nonlinear influence on the target. Conversey, Linear Regression evaluates only linear relationships, leading to **`Quantum_Battery`** and **`Wind_Speed_kmph`** having significant coefficienct due to their correlations.

This highlights the difference in the underlying mechanics of the two models: Gradient Boosting captures complex dependencies by amplifying the importance of certain features while minimizing others, while Linear Regression prioritizes direct proportionality.








<h2 align="center">Conclusion</h2>

In this project, we successfully developed machine learning models to predict the cargo capacity of autonomous delivery drones operating in Aeropolis, a futuristic urban environment. Through a combination of Gradient Boosting, Linear Regression and other algorithms, we analyzed the key factors affecting drone performance and optimized the models for accurate cargo predictions.

The feature importance analysis revealed that factors such as Wind_Speed_kmph, Quantum_Battery, and Flight_Duration_Minutes had the highest influence on the cargo capacity. These findings emphasize the critical role of environmental conditions and battery efficiency in ensuring optimal drone operations. Other features, while less significant, provided additional context for fine-tuning performance metrics in complex scenarios.

When evaluating the models on the test data, we observed that Gradient Boosting consistently outperformed other models, achieving the best balance between training and validation accuracy, as well as minimizing overfitting. This indicated that it captured both linear and non-linear relationships in the data effectively. In contrast, the test results from Linear Regression and Support Vector Regression (SVR) highlighted their limititations in handling non-linearity and subtle data patterns. These insights validated the robustness of Gradient Boosting for predictive accuracy and emphasized the importance of selecting models suited to the nature of the data.

To achieve reliable predictions, we applied rigorous preprocessing techniques, including scaling, outlier removal and feature engineering, which ensured the models effectively captured the relationships within the dataset. Gradient Boosting stood out as the most robust model due to its ability to handle non-linear interactions and complex data patterns, making it ideal for a dynamic and data-rich environment like Aeropolis. Meanwhile, Linear Regression provided straightforward insights into feature impacts, reinforcing our understanding of key variables.

This project highlights the potential of machine learning in optimizing autonomous drone logistics, particularly in high-demand, tech-forward cities like Aeropolis. Future work could expand on this by integrating real-time drone data, exploring weather forecasting models, and developing adaptive systems to dynamically adjust drone operations based on environmental and operational changes. These advancements could further enhance delivery efficiency and ensure seamless logistics in smart cities.








