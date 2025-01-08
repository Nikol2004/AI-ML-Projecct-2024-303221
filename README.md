# AI&ML Project 2024-2025

# Project 4 - AEROPOLIS

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

[EDA](EDA)


## 1. EDA

### 1.1 Data Overview

### 1.2 Checking for Duplicates

### 1.3 Checking Data Integrity

#### 1.3.1 Missing Values per row

### 1.4 Descriptive Statistics using the original data

#### 1.4.1 Categorical Values













### 1) Understanding the dataset
- 1.1)Overview of the dataset: with *'aeropolis_df.head()'* we extract the first few rows of the dataset, which helps us to better understand the structure of the data, the columns, and to have a glance at the values;
Then, thanks to the **`shape`** attribute, we can see that the dataset is composed by 1,000,000 rows and 20 columns: this is useful to verify the size of the dataset before processing.
Moreover, *'aeropolis_df.info()'* allows us to output the summary of a DataFrame, with information like the number of rows or columns or columns' data types, and the count of non-null values for each column.  This result highlights the fact that each column has missing data and that the data types are either **`float64`**, used for floating-point numbers in 11 columns out of 20, or **`object`**, for strings or columns containing mixed data types in the remaining 9 columns. 
- 1.2)Checking for duplicates: The **`.nunique()`** method calculates the number of unique values in each column of the DataFrame, which helps us to understand the distribution and variability of the data in each feature. 
The result shows us that there are some attributes that have a low number of unique values, which may indicate that they are categorical values, whilst there are other attributes that have a high number of unique values, which may indicate that they are numerical values. 
With **'aeropolis_df.duplicated().sum()'** we ensure the dataset is free from redundancy, which could bias our analysis or training process.
- 1.3)Checking data integrity: To check data integrity in a more detailed way we create the function **`missing_values_table`** to return the total count of missing values for each column, and their respective percentage relative to the number of rows.
The table we compute with **'missing_values_table(aeropolis_df)'** shows that the dataset has significant missing values with circa 10% in each column. 
	- 1.3.1)We can also see that as the number of missing values increases, the count of rows decreases significantly, which means that the major part of the dataset will require minimal imputation, while 	the other part will need to be processed or removed.
- 1.4)Descriptive Statistics using the original data: 
	- 1.4.1)Categorical Values: The **`select_dtypes(include=['object'])`** method filters out only the columns with data type object, which are the categorical variables.
	The **`.describe()`** method generates summary statistics for the categorical columns, such as **`count`**, which tells the number of non-null values in each column, or **`top`**, which tells the most 	frequent category (mode).
	**Interpretation**
	* The results from the categorical data description show that the dataset is fairly balanced in terms of representation across different categories, such as **`Weather_Status`**, **`Package_Type`**, 	**`Market_Region`**, and others. 	
	* For example, **`Market_Region`** has three categories, with "Local" being the most frequent at 300,377 instances. Similarly, **`Quantum_Battery`** is binary with a balanced split, showing **`True`** 	as the top value at 450,103.

	* This balance suggests that the categorical variables in the dataset provide diverse representations, minimizing potential biases.

	We loop through all categorical columns to then create a pie chart for each column to visualize the distribution of values, which is fairly balanced: for example, Weather_Status or Market_Region are 	almost equally divided into their 3 categories, whilst Package_Type is evenly distributed across all its 6 categories. 
(Insert plot) (**)
	Then we loop through the categorical variables to get more precise percentages for each category and we see that the categories differ by at most 0.1% in their distribution (which means that the latter is highly balanced).

	For example, the **`Vertical_Landing`** feature shows almost equal proportions for Unknown, Unsupported, and Supported. 
	This balance suggests that no single category dominates, reducing the risk of model bias towards a particular class and ensuring fair representation during training.

	- 1.4.2)Numerical Values: In this case, we proceed as we did for the categorical columns, using the select_dtypes method the DataFrame to include only numerical columns, the describe() method to 		calculate summary statistics for numerical columns, including **`Mean`**: , which gets the average value, or **`25%, 50%, 75%`**: which give the percentiles that help us better understand the data 		distribution.
	**Interpretation**
	* The statistical summary provides insights into the numerical variables in the dataset. For the target variable **`Cargo_Capacity_kg`**, the mean is approximately 4.65 kg, with a standard deviation 	of 1.69 kg, indicating moderate variability. 

	* Some negative values suggest potential anomalies or preprocessing errors.
	The **`hist()`** function generates histograms for all numerical columns to visualize their distribution. (Insert plot)(**)
	**Interpretation**

	* **`Cargo_Capacity_kg`** and **`Water_Usage_liters`**: Have bell-shaped distributions with potential outliers at the edges.
	* **`Cleaning_Liquid_Usage_liters`**: Highly skewed to the left, indicating most values are concentrated near zero.
	* **`Autopilot_Qualty_Index`**, **`Vertical_Max_Speed`**, and **`Wind_Speed_kmph`**: Show relatively uniform distributions. 

		-1.4.2.1) Checking for low variability columns: We check for numerical columns with a standard deviation (variability) of less than 0.01, and see that all numerical columns exhibit enough 			variation to potentially contribute to the analysis.

- 1.5) Handling missing values of the dependent variable: We remove rows missing the target value (of the (**`Cargo_Capacity_kg`**) variable), using the **`dropna()`** method, because we are working on a regression problem, and therefore:
 	* The model cannot learn without target values, as it needs them to calculate errors and adjust weights.
	* Missing target values make it impossible to compare predictions and measure model performance.
	* Retaining these rows adds noise and unnecessary complexity without contributing to the model.

### 2) Cleaning the dataset

- 2.1) Encoding Categorical Values: Here we filter out the categorical columns and iterate through each column to display its name and the unique categories it contains. It provides an overview of all possible values in each categorical column, including any missing values. 

The result will help us with the mapping. 
We calculate the average **`Cargo_Capacity_kg`** for each category within a categorical column by grouping the dataset using **`groupby`** and applying the **`.mean()`** function to the target column. It identifies if categories have distinct effects on the target value, and based on our results, we have that, for example, **`Weather_Status`** has similar mean values across **`Cloudy`**, **`Sunny`**, and **`Rainy`**, indicating little influence of weather on the target variable. 

* Similarly, other columns like **`Package_Type`** and **`Vertical_Landing`** also show minimal differences across categories, suggesting a weak correlation. This justifies the decision to then map categories randomly, as their direct impact on the target appears negligible.

**Interpretation**

The data types confirm that all columns are now numerical, ready for further processing and modeling. The random samples show the encoded categorical columns alongside the numerical features, with some missing values still present.

We then define a dictionary of mappings, assigning a numerical value to each category in the categorical columns, using the **`map()`** function to replace the original categorical values with their corresponding numerical mappings for each column. 

We preserve the **`NaN`** values for later imputation, and check the data types, which confirm that all columns are numerical.

We also get some random samples, which show the encoded categorical columns alongside the numerical features, with some missing values still present.

- 2.2) Finding the correlation between the independent values and the target value: We compute the pairwise correlation between all the columns using **`.corr()`** and then visualize it in a heatmap using **`seaborn.heatmap`**.
(Insert plot)(**) 
**Interpretation**

The heatmap shows correlations between all variables. For example, **`Wind_Speed_kmph`** has a strong positive correlation (**`0.76`**) with the target variable **`Cargo_Capacity_kg`**. 

On the other hand, features like **`Vertical_Max_Speed`** and **`Market_Region`** show near-zero or negative correlations. We then extract the correlation values between all features and the target, sorting them in ascending order, and see that **`Wind_Speed_kmph`** has the highest positive correlation (0.76), followed by **`Quantum_Battery`** (0.44), whilst features like **`Delivery_Time_Minutes`** and **`Market_Region`** have weak or negative correlations, making them potetial candidates for removal to simplify the model.

- 2.3) Dropping rows with missing values in important features: The rows with missing values for the highly correlated features are removed, ensuring that critical data points aren't compromised when training the model. 

- 2.4) Dropping irrelevant or negatively correlated columns: We make sure that columns with weak or negative correlations to the target are dropped to reduce dimensionality and eliminate noise in the dataset.

- 2.5) Removing skewness: From the descriptive statistics before, we saw that one of the numerical values had the potential of being skewed, and therefore we have to remove it. 
(Insert plot) (**)
From the plot we see that the original distribution of **`Cleaning_Liquid_Usage_liters`** is right-skewed, indicating that most values are clustered near zero. After the log transformation using **`np.log1p`**, the distribution becomes more symmetric, which can improve model performance by ensuring that the feature follows a normal distribution.
(Insert plot) (**)
After applying the transformation, the updated histogram shows a much more balanced distribution. This adjustment ensures that the feature aligns better with the assumptions of most machine learning algorithms and helps improve model accuracy and reliability.


- 2.6) **Removing Outliers**
Outliers are data points that deviate significantly from the majority of the data. These extreme values can distort analysis and negatively affect model performance. Therefore, we needed to identify and handle outliers effectively.

(Insert boxplot) (**)

To identify the outliers, boxplots were used for all numerical columns. Features such as Flight_Hours and Cleaning_Liquid_Usage_liters showed significant outliers. These outliers were handled by either removing or adjusting them, depending on their influence on the overall dataset.

(Insert boxplot visualization of outliers removed) (**)

By addressing these extreme values, we reduced variability in the dataset and improved the generalization of our machine learning models. Handling outliers also helps ensure that the models are not biased by these irregular data points and can provide more reliable predictions.
   - 2.6.2) **Removing Outliers**
      Outliers are extreme values that deviate significantly from other observations in the dataset. They can distort statistical analyses and negatively impact model performance. To address this, the IQR (Interquartile Range) method is applied.

      The IQR method calculates the range between the first quartile (Q1) and the third quartile (Q3). Values that lie below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are considered outliers and are removed. This ensures that the remaining data points represent a more consistent and accurate distribution.

      By applying the IQR method across the dataset's numerical features, we eliminate rows with extreme values, reducing variability and improving the dataset's quality.

     After removing the outliers, the dataset becomes more robust and less prone to the effects of extreme data points. This step helps reduce noise and ensures that subsequent analyses or machine learning models are not biased by these extreme values.

   - 2.6.3) **Boxplots of the Filtered Data
      After removing outliers, a new set of bo**xplots is generated for the filtered dataset. These visualizations allow us to confirm that the dataset is now free of extreme values.
	  Each feature's boxplot is presented, showcasing the improved distribution of values after the removal of outliers. This process ensures that the dataset is prepared for further analysis or model training.

- 2.7) **Downsampling the Dataset** 
Downsampling is performed to reduce the dataset size while preserving its statistical properties. This process simplifies computational requirements and ensures the dataset remains manageable.
   -  2.7.1)Distribution of Cargo Capacity (Downsampling Step)
       **Description:**  The distribution of Cargo_Capacity_kg was analyzed before and after downsampling to ensure the shape remained *symmetrical* and *bell-shaped* This step ensures that downsampling does not alter the dataset's inherent structure.
       **Downsampling Approach:** A random sample of 10% of the rows was selected from the dataset to reduce computational complexity.
   -  2.7.2)Comparing Variance in the Original and Downsampled Dataset
       **Objective:** Compare the variance of each feature between the original and downsampled datasets to verify that the downsampling process preserved the statistical properties. The variance values in the original and downsampled datasets are nearly identical.
       *This demonstrates that downsampling was effective without distorting the dataset's characteristics.*


### 3) Feature selection:
In this section, the only thing we had to decide was the treshold to select features to use to train the model. We simply tried to find the best value empirically, conducting an experiment that is explained in the **Experimental Design** section. The optimal result was obtained with a treshold of 0.15. As a consequence, the features dropped were: 'Gender', 'Arrival Delay in Minutes', 'Departure Delay in Minutes' and 'Age'. 

- 3.1) **Splitting Discrete and Continuous Values**
To prepare the dataset for preprocessing, the numerical columns were divided into two categories: *discrete* and *continuous*. This distinction helps streamline subsequent processing steps, as discrete values often require encoding, while continuous values may need normalization.
   **Discrete Variables**: These are numerical columns with a small, finite number of unique values (e.g., Weather_Status, Package_Type).
   **Continuous Variables**: These are numerical columns with a wide range of unique values (e.g., Cargo_Capacity_kg, Flight_Hours).
By setting a threshold of 10 unique values, the dataset was split into the respective categories to ensure appropriate handling of each variable type.
-3.2) **Normalizing Continuous Features**
To ensure all continuous variables are on the same scale, normalization was applied using the StandardScaler. This process adjusts the data such that each variable has a mean of 0 and a standard deviation of 1. Scaling is crucial for algorithms sensitive to feature magnitudes, such as gradient-based models.

*Validation of Scaling* : After normalization, the mean and standard deviation of each continuous column were checked to verify successful scaling. Results confirmed that all means are approximately **0**, and standard deviations are approximately **1**, indicating the data is ready for further analysis.
- 3.3) KNN (K-Nearest Neighbors) for Imputation
*What is KNN and How Does it Work?*
(KNN) algorithm predicts the value of a data point by considering its closest neighbors in the feature space. It identifies the k closest data points based on a distance metric and uses their average (or weighted average) to make the prediction.

**Steps:**
  1. Identify the k-nearest neighbors to the query point.
  2. Compute the average target value of these neighbors.
  3. Use this average as the predicted value for the query point.
  4. This algorithm assumes that similar data points have similar values, making it  effective for estimating missing values based on patterns in the dataset.
 
- 3.3.1) Imputing Missing Values Using KNN
    The KNN imputation process fills missing values by finding similar rows (neighbors) based on feature similarity. Continuous variables are directly imputed, while categorical (discrete) variables are rounded to ensure their integrity.
    **Process:**
    For *continuous variables*: Impute values as they are.
    For *categorical variables*: Round imputed values to maintain their categorical nature.
    After applying KNN, the dataset was rechecked to ensure no missing values remained.
    **Interpretation:**
    After applying KNN, all missing values across the dataset were filled, as confirmed by the absence of null values in the dataset.
    The imputation process ensures the dataset is now complete and ready for subsequent analysis or modeling.
    The relationships between features were preserved, producing more reliable and meaningful imputations.



### 4) Splitting into training and test data:
	The dataset is split into training and testing sets using **`train_test_split`**. 

The **`X`** variables represent features, while **`y`** is the target variable.

The **`test_size=0.2`** specifies that 20% of the data is reserved for testing, and **`random_state=42`** ensures reproducibility. 

The training and testing shapes are printed to verify the split proportions.
* The **`train_test_split`** confirms the data is divided as intended, with 14,872 sampled in the training set and 3,719 in the test set.

- 4.1) **Distribution of values in the training and test data**: KDE (Kernel Density Estimation) plots are used to visualize the distributions of features in the training and test sets. We iterate through all features, plot overlapping KDEs for both sets, and adjust the layout to ensure clear visualization.
(Insert plot) (**)
**Interpretation**

* The KDE plots indicate that the feature distributions in both sets align closely, suggesting an even split without significant sampling bias. This ensures that the models trained on the training data will generalize well to the test data. However, features like **`Vertical_Landing`** and **`Terrain_type`** show distinct peaks, reflecting categorical distributions, while others like **`Air_Temperature_Celsius`** are continuous and symmetric.

### 5) Model Building
<h5 style="color: lightpink">A regression problem</h5>

This task was approached as a regression problem because the target variable, **Cargo_Capacity_kg**, is a continuous numerical value.

Regression models are specifically designed to predict continuous outcomes by learning the relationships between the input features and the target variable. Unlike classification, which deals with discrete categories, regression enables the prediction of a wide range of possible values, making it suitable for estimating quantities such as weight, price, or, in this case, the cargo caapacity of autonomous delivery drones. 

This choice aligns with the dataset structure and the objective of providing accurate numerical predictions, essential for operational and logistical planning in drone delivery systems.
- 5.1) **Testing Different Models**: We chose to test using the following regression models:
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
	-5.1.1) **Computing the metrics and comparing**: We define a dictionary **`models`** that holds several machine learning models for regression.
	Iterating through the models, training them on the dataset, and calculates the key metrics:

	**`MSE (Mean Squared Error)`**: Measures the average squared differences between actual and predicted values.

	**`MAE (Mean Absolute Error)`**: Measures the average absolute differences.

	**`R^2 (Coefficient of Determination)`**: Indicates the proportion of the variance explained by the model.

	**`Runtime`**: Captures the time each model takes for training and predictions.

	We then compute the Learning Curves and the Prediction Error Plots for each model.

	(Insert plots)(**)

	Afterwards, we aggregate the results from all models into a DataFrame. It sorts and displays them by R^2 in descending order, providing a direct comparison of their performances.
	**Interpretation**

	* **`Linear Regression`** performs best in terms of R^2 (0.9123) and runtime (0.046) seconds, making it efficient and effective.

	* **`Gradient Boosting`** has slightly lower R^2 but is robust for capturing nonlinear relationships.

	* **`SVR`** performs comparably but takes londer to run.

	* **`Random Forest`** performs moderately but has higher computational costs.

	* **`KNN`** shows the worst performance due to overfitting, with significantly lower R^2.
	- 5.1.2) Visualizing if the continuous values have a linear relationship with the target value: We create scatterplots to visualize the relationship between each continuous feature and the target 		variable (**`y_train`**). It loops through all continuous columns in the training dataset, plotting each feature against the target.
	(Insert plots)(**)

	**Interpretation**

	The scatterplots show that most features lack a strong linear relationship with the target variable, as the points are scattered uniformly. Specifically:

	1. *Air Temperature*, *Flight Hours*, *Cleaning Liquid Usage*, *Autopilot Quality*, and *Route Optimization*: The data points are distributed without any discernible pattern, suggesting no clear 		linear correlation with the target.

	2. *Wind Speed (kmph)*: This feature shows a visible pattern where the target value increases with wind speed, indicating a potential linear or non-linear correlation.

	This confirms that Linear Regression is not suitable for accurately predicting the target variable in this dataset and will be used only as a baseline model.

- 5.2) **Choosing the best models**: 

	##### 1. Gradient Boosting:
	**Why was it chosen?**

	Gradient Boosting is a powerful ensemble learning technique that build models sequentially, minimizing errors at each step. It was chosen because:

	* It captures complex, non-linear relationships in the data.

	* The model has shown robust performance with low MSE and high R^3 scores during cross-validation.

	* It is less prone to overfitting than Random Forest in some scenarios due to its iterative training process.

	##### 2. Support Vector Regressor (SVR)

	**Why was it chosen?**

	SVR uses kernel function to model non-linear relationships effectively. It was chosen because:

	* It is capable of finding a balance between bias and variance by defining margins for the predictions.

	* The model performed well in terms of accuracy ad R^2 scores, proving its suitability for this dataset.

	* It can handle outliers better than simpler regression techniques.

	##### 3. Linear Regression (as a baseline)

	**Why was it chosen?**

	Linear Regression was included as a baseline model for comparison purposes. It was chosen because:

	* It is straightforward, interpretable, and computationally efficient.

	* Despite its simplicity, it provides a benchmark to evaluate the performance of more complex models.

	* The scatterplots demonstrated that most features lack linear relationships with the target, confirming its limited utility but making it ideal for baseline evaluation.

	##### Why not the others?

	* *Random Forest*: While Random Forest is a strong model, it requires more computational resources and showed slightly inferior performance compared to Gradient Boosting and SVR.

	* *K-Nearest Neighbors(KNN)*: KNN performed poorly with the highest MSE and lowest R^2, indicating its inability to capture the complexities of the dataset. Additionally, its performace degrades with 	high-dimensional data.

-5.3) **Hyperparameter Tuning for Gradient Boosting Regression**:
	- 5.3.1) Define the Hyperparameter Grid:
	In the first step, a hyperparameter grid for the Gradient Boosting model is defined (**`param_grid`**). It includes possible values for parameters such as the number of estimators 				(**`n_estimators`**), learning_rate (**`learning_rate`**), maximum depth of the trees (**`max_depth`**), and others.

	**The parameters chosen**

	* **`n_estimators`**: This represents the number of boosting stages (or trees). A higher number allows the model to learn more complex patterns but can lead to overfitting if too large.

	* **`learning_rate`**: Conrols the contribution of each tree to the final prediction. Smaller values reduce overfitting and require more estimators for good performance.

	* **`max_depth`**: Limits the maximum depth of individual trees, controlling how complex each tree can be. Shallower trees help prevent overfitting.

	* **`min_samples_split`**: The minimum umber of samples required to split an internal node. Larger values result in less complex trees.

	* **`min_samples_leaf`**: The minimum number of samples required to be at a leaf node. Higher values make the trees more robust by reducing overfitting.
	
	- 5.3.2) Initialize GridSearchCV:
	We initialize a  **`RandomizedSearchCV`** object for hyperparameter tuning of the **`GradientBoostingRegressor`**. Here's what each parameter does:

	* **`estimator`**: Specifies the model, in this case, GradientBoostingRegressor.

	* **`param_distributions`**: The hyperparameter grid defined earlier (param_grid) that included values to sample during the search.

	* **`n_iter`**: The number of random combinations of hyperparameters to try (100 iterations here).

	* **`scoring='r2'`**: The performance metric used is the R^2 score.

	* **`cv=5`**: Performs 5-fold cross-validation for each parameter combination.

	* **`random_state=42`**: Ensures reproducibility of the results.

	* **`n_jobs=-1`**: Utilizes all available CPUs for computation.

	* **`verbose=1`**: Controls the amount of output during the execution.

	- 5.3.3) Fit the model:
	The best model and hyperparameters are then selected using **`random_search_gb.best_params_`** and evaluated on the test set.

	**Interpretation**

	1. **`n_estimators`**: 200

	This means the model uses 200 boosting iterations (or trees). Increasing the number of estimators can improve model performance but also increases computational time. A value of 200 strikes a balance 	between accuracy and efficiency.

	2. **`min_samples_split`**: 15

	Specifies the minimum number of samples required to split an internal node. By setting it to 15, the model avoids overfitting by ensuring splits are only performed sufficiently large groups of data.

	3. **`min_samples_leaf`**: 6

	This is the minimum number of samples required to be in a leaf node. Setting it to 6 reduces the likelihood of the model learing overly specific patterns (overfitting) and encourages more generalized 	splits.

	4. **`max_dept`**: 3

	Limits the maximum depth of each tree to 3 levels. This helps the model maintain simplicity and prevents overfitting by not learning overly complex patterns.

	5. **`learning_rate`**: 0.05

	Controls the contribution of each tree to the final prediction. A smaller learning rate (0.05) slows down the learning process, allowing the model to build more robust trees by minimizing the chance 	of overfitting.
	
	-5.3.4) Evaluate on the Test Set:
	**Interpretation**

	On the test set, the model performed with an MSE of **`0.0882`**, MAE of **`0.2381`**, ad an R^2 of **`0.9111`**.

	The model's test set results validate its effectiveness and confirm that the chosen hyperparameters generalize well.

-5.4) Hyperparameter Tuning for SVR
	-5.4.1 Define the Parameter Grid: 
	**The parameters chosen**

	* **`C`**: Regularization parameter. A smaller value of C allows for a larger margin, but may increase bias, while a larger value of C reduces margin for a better fit but may risk overfitting. Values 	like 0.1, 1, 10, 100 are tested.

	* **`epsilon`**: Defines the margin of tolerance where no penalty is given in the training loss function. Smaller values are more sensitive to deviations from actual values. Values like 0.001, 0.01, 	0.1, 1 are tested.

	* **`kernel`**: Specifies the kernel type to be used in the algorithm
		* *linear*: A linear kernel assumes linear relationships in the data
		* *rbf*: Radial Basis Function kernel is flexible and works well with non-linear data.
		* *poly*: Polynomial kernel can capture complex relationships.

	-5.4.2) Initialize RandomizedSearchCV:
	This initializes a hyperparameter seach over the SVR model:

	* **`estimator=SVR()`**: Defines the base model (SVR) to tune.

	* **`param_distributions=param_grid_svr`**: Uses the grid of parameters defined earlier.

	* **`n_iter=20`**: The search will test 20 random combinations of the parameter grid.

	* **`scoring='r2'`**: The performance metric used is the R^2 score.

	* **`cv=5`**: Performs 5-fold cross-validation for each parameter combination.

	* **`random_state=42`**: Ensures reproducibility of the results.

	* **`n_jobs=-1`**: Utilizes all available CPUs for computation.

	* **`verbose=2`**: Provides detailed output during the search.
	-5.4.3) Fit the Model:
	The **`random_search.fit(X_train, y_train)`** code trains the model by:

	* Running 20 iterations of hyperparameter combinations.

	* For each combination, 5-fold cross-validation is conducted on the training data to compute the R^2 score.

	* The best hyperparameters and corresponding cross-validated R^2 score are identified.

	The print statements output the best parameters and cross-validation results, such as:

	* Best Parameters for SVR

	* Best Cross-Validation R^2

	**Interpretation**

	The best parameters indicate:

	* **`Kernel`**: 'linear' was optimal, meaning the relationship between predictors and the target is sufficiently linear.

	* **`C:`** 1, which provides a balance between margin size and fitting accuracy.

	* **`Epsilon`**: 0.01, a small margin of tolerance, indicating the model is sensitive to deviations in predictions.

	The cross-validation R^2 score of 0.913 shows the model generalizes well on unseen data, with high accuracy.
	-5.4.4) Evaluate on the Test Set:
	**Interpretation**

	* MSE: **`0.0869`** indicates low average squared error.

	* MAE: **`0.2355`** suggests predictions are off by about 0.2355 units on average.

	* R^2: **`0.9124`** confirms the model explains 91.24% of the variance in the target variable.

	These metrics show that the SVR model is well-tuned and performs similarly on both training and test datasets, indicating minimal overfitting.

-5.5) Evaluation for Linear Regression

A simple Linear Regression model is trained and evaluated on the test set without hyperparameter tuning, as it does not have tunable hyperparameters. The performance metrics are calculates similarly using MSE, MAE, and R^2.

**Interpretation**

* The Linear Regression model achieved comparable performance to the other models, with an MSE of **`0.0870`**, MAE of **`0.2357`**, and an R^2 of **`0.9123`**. 

* This suggests that even without advanced techniques, linear regression is a strong baseline for this dataset.

	-5.5.1) Comparison of the Results:
	**Interpretation**

	* The results show that SVR performs slightly better than both Gradient Boosting and Linear Regression, with the lowest MSE, lowest MAE, and the highest R^2.

	* Linear Regression performs suprisingly well, almost matching SVR in R^2, but its MSE and MAE are sligtly higher. 

	* Gradient Boostig, while effective, has slighlty higher error metrics and a marginally lower R^2, indicating it may not capture the underlying data patterns as effectively as SVR.

	(Insert plot Model Performance Comparison)(**)
	**Interpretation**

	The bar chart visualizes the performance metrics for the three models.

	* SVR has the lowest MSE and MAE, confirming it makes the smallest prediction errors among the three models. It also has the highest R^2, indicating it explains the variance in the target variable 		better than the others.

	* Linear Regression performs almost as well as SVR, with comparable R^2 and slightly higher error metrics. 

	* Gradient Boosting, while effective, shows slightly higher MSE and MAE, suggesting it may not generalize as well as the other models for this dataset.
	
	-5.5.2) Residual Plots
	Residual plots help evaluate how well a model's predictions align with the actual data. They plot the residuals (differences between predicted and actual values) against the predicted values. 		Residuals should ideally have no discernible pattern and be centered around zero.

	This indicates that the model captures the data effectively and that errors are randomly distributed. Using residual plots is essential to check for non-linearity, and any systematic bias in the 		predictions.
	(Insert plots Residual Plot for GB, SVR and Linear Regression)
	
	**Interpretation**

	1. *Gradient Boosting*: The residuals are distributed relatively evenly around zero, with no visible pattern. This indicates that the model captures the data structure well, though some variance in 	prediction errors is noticeable.

	2. *SVR*: The SVR residual plot also shows residuals scattered evenly around zero with minimal structure or pattern, suggesting that the model predictions align well with the actual data. The 		dispersion of residuals is slightly more concentrated compared to Gradient Boosting, indicating consistent prediction behavior.

	3. *Linear Regression*: The residuals for linear regression are evenly distributed around zero, similar to the other two models. However, since linear regression assumes a linear relationship, any 		slight deviations might indicate that this assumption may not fully capture the complexity of the data.

	Overall, all three models show acceptable residual distributions, but Gradient Boosting and SVR may handle complex data patterns slightly better than Linear Regression due to heir more advanceed 		architectures.

-5.6) Training and evaluating the models after Hyperparameter Tuning

* This code defines the models (**`Gradient Boosting`**, **`SVR`**, and **`Linear Regression`**) with their best hyperparameters obtained during the tuning process.

* **`results_with_best_params`** is initialized as an empty dictionary to store the metrics for each model.

Then the function **`plot_learning_curve`** visualizes the relationship between training set size and model performance on training and validation data, and the resulting plot shows how the model's performance varies as the training set size increases, helping to detect underfitting or overfitting.

Each model is iteratively trained using the training data and predictions are made on the test set, and performance metrics and runtime are computed and stored in the results dictionary.

(Insert plots of Gradient Boosting, SVR and Linear Regression with Best parameters) (**)

**Interpretation**

**`Learning Curves`**

1. Gradient Boosting

	* The training accuracy decreases slightly as the training size increases, suggesting the model generalizes well.
	* The validation accuracy remains consistent with a slight improvement as more data is used, indicating low overfitting.

2. SVR

	* Similar to Gradient Boosting, the training accuracy decreases as the training size increases.
	* Validation accuracy shows marginal improvement with more data, indicating stable performance and minimal overfitting.

3. Linear Regression

	* The training accuracy is slightly lower than that of Gradient Boosting and SVR, reflecting the simpler nature of the model.
	* Validation accuracy remains steady and comparable to the training accuracy, indicating good generalization.


**`Performance Metrics`**

* *Gradient Boosting*: Offers strong performance with an R^2 of 0.9111 and low errors but has a slightly longer runtime compared to Linear Regression.

* *SVR*: Achieves the highest R^2 and lowest error metrics but has a significantly higher runtime.

* *Linear Regression*: Has a marginally lower R^2 but compensates with the fastest runtime, making it suitable for time-sensitive tasks.

**Why so small differences?**

The minimal differences observed in the learning curves and evaluation metrics across models can be attributed to the homogeneity of the dataset. Specifically, equally distributed categorical variables simplify patterns in the data, reducing the need for models to adapt to challenging or unique scenarios.

As a result, hyperparameter tuning has little impact, as the models can already achieve optimal or near-optimal performance on the dataset.


!!! CONTINUE FROM HERE!!!


### 6) Plotting Learning Curves:
Learning curves illustrate how a model's performance evolves as it's trained on varying amounts of data, revealing insights into overfitting, underfitting, and the impact of dataset size on model accuracy. Learning curves are crucial because they show how the model's performance varies with the size of the training set. They can reveal issues such as overfitting or underfitting. In addition, they can help us determine whether collecting more data would be useful. In this section, we are going to visualize and comment the learning curves for each model.
- 6.1) **Logistic Regression**: The learning curve for the logistic regression model is reported in the following plot: <br>  
<img src="images/Learning_Curve_Logistic_Regression.png" width="500" height="400"> <br>  
As we can see, the learning curve present a strange behavior: the training score is at its maximum with a lower amount of data, then it starts decreasing (maybe because the model becoming less complex). This idea is supported by the fact that the learning curve is initially increasing in the validation set. However, the two curves converges at the value 0.87, which is not a really bad result. In addition, since the performance on unseen data is increasing, the model is improving its ability to generalize.
- 6.2) **Decision Tree**: The learning curve for the decision tree model is reported in the following plot: <br>  
<img src="images/Learning_Curve_Decision_Tree.png" width="500" height="400"> <br>
In this plot, we can observe that the training score is always higher than the cross validation score: decision trees are indeed prone to overfitting. However, even if the training score is higher than the cross validation score, the two scores are really close with a large amount of data, indicating that the model is improving its ability to generalize.
- 6.3) **Random Forest**: The learning curve for the random forest model is reported in the following plot: <br>
<img src="images/Learning_Curve_Random_Forests.png" width="500" height="400"> <br>
In this plot, we can observe that the training score is almost constant: the model is slightly overfitting, even if the validation score is increasing and its value is higher than the validation score of the decision tree model. In addition, the two curves are starting to converge, indicating that the model is improving its ability to generalize.


### 7) Models Evaluation
- 7.1) **Classification metrics**: quantitative measures (such as accuracy, precision, recall (sensitivity), F1-score, ROC-AUC) we used to assess the performance of our classification models, providing insights into the model's ability to predict classes accurately, detect true positives, and minimize false predictions. In particular, with the methods *'classification_report'* and *'confusion_matrix'* we computed the following metrics:
  - **Precision**: Precision for a given class in multi-class classification is the fraction of instances correctly classified as belonging to a specific class out of all instances the model predicted to belong to that class. In general, precision is used to measure the model's ability to correctly identify the positive class and it is essential to minimize false positives.
  - **Sensitivity**: Sensitivity in multi-class classification is the fraction of instances in a class that the model correctly classified out of all instances in that class. In general, sensitivity is used to measure the model's ability to correctly identify the positive class and it is essential to minimize false negatives.
  - **F1-score**: The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
  - **Macro Average**: Macro-averaging calculates each class's performance metric (e.g., precision, recall) and then takes the arithmetic mean across all classes.
  - **Weighted Average**: not really useful in this case, since the dataset is balanced.
  - **Accuracy**: Accuracy is the fraction of instances the model correctly classified out of all instances.
- 7.2) **Confusion Matrices**: a tabular representation to visualize the performance of a classification algorithm, allowing a clear understanding of true positives, true negatives, false positives, and false negatives. This matrices are fundamental for evaluating a model's precision, recall, accuracy, and other classification metrics.
- 7.3) **ROC Curves**: (Receiver Operating Characteristic) are graphical representations that illustrate a classification model's performance across various thresholds. They plot the true positive rate (sensitivity) against the false positive rate (1-specificity) for different threshold values, providing a comprehensive overview of a model's ability to distinguish between classes: the area under the ROC curve (AUC-ROC) quantifies the model's overall performance, with a higher AUC indicating better discriminatory power.
- 7.4) **Models Comparison**:  We give final thoughts about the three classification models we chose, selecting the most suitable model based on the previous results we found and its predictive capabilities. All the results are reported in the **Results** section and commented in the **Conclusions** section.


## EXPERIMENTAL DESIGN

In this section, we are going to illustrate experiments conducted to demonstrate and validate the target contribution of the project. The experiments are divided into three main sections, each one with a specific objective, baseline(s) and evaluation metric(s). 

### 1) Handling Outliers

**Objective:**
This experiment aimed to assess the impact of noisy data on the machine learning model's performance. As evident from the Exploratory Data Analysis (EDA), the dataset contains outliers, particularly influencing the features 'Arrival Delay in Minutes' and 'Departure Delay in Minutes.' Conversely, many features exhibit a range between 1 and 5, rendering outliers less influential. The experiment sought to quantify the impact of outliers on various model performances and identify the optimal approach for handling them.

**Baseline(s):**
The baseline for comparison involves evaluating the model's performance without any outlier handling. This baseline serves as a reference point to determine whether the outlier handling approach enhances model performance.

**Evaluation Metric(s):**
Given the absence of a specific optimization goal (e.g., minimizing false positives or false negatives), multiple metrics were considered for assessing model performance. These metrics included accuracy, precision, recall, F1-score, and ROC-AUC score. However, prioritizing the minimization of false positives over false negatives, as predicting a customer as satisfied and later discovering dissatisfaction is deemed less impactful, precision was selected as the primary metric for model evaluation.

### Conclusion:
The most effective approach for handling outliers proved to be their removal. Model performance notably improved when outliers were excluded. However, the specific reason behind this improvement remains unclear, as the dataset (as said before) does not really present anomalous values and hence there is no proper reason for outlier removal. Consequently, a selective removal strategy was adopted, focusing only on outliers within the 'Distance' feature, where certain values significantly deviated from the norm ('Arrival Delay in Minutes' and 'Departure Delay in Minutes' were already removed). We firmly believe that this approach is better suited for a broad range of scenarios, and our objective was to construct a model capable of delivering high performance across diverse datasets.

### 2) Hyperparameter Tuning

**Objective:**
This experiment aimed to assess the impact of hyperparameter tuning on the machine learning model's performance and to find the optimal set of parameters.

**Baseline(s):**
The baseline for comparison involves evaluating the model's performance without any hyperparameter tuning. This baseline serves as a reference point to determine whether hyperparameter tuning enhances the model's performance.

**Evaluation Metric(s):**
Consistent with the previous experiment, multiple metrics were considered to evaluate the model's performance, with a particular emphasis on precision.

### Conclusion:
Detailed results, including the optimal hyperparameters for each model, are provided in the main notebook. The preferred approach to hyperparameter tuning involved utilizing the RandomizedSearchCV function to identify a promising set of hyperparameters. Subsequently, the GridSearchCV function was employed to pinpoint the best hyperparameters within a narrower range of values, resembling a local maximum.

### 3) Feature Selection

**Objective:**
This experiment aimed to assess the best approach for feature selection, identifying the optimal treshold.

**Baseline(s):**
The baseline for comparison involves evaluating the model's performance without any feature selection. This baseline serves as a reference point to determine whether feature selection enhances the model's performance.

**Evaluation Metric(s):**
Consistent with the previous experiment, multiple metrics were considered to evaluate the model's performance, with a particular emphasis on precision.

### Conclusion:

The optimal treshold for feature selection was found to be 0.15. This approach was selected because it yielded the best results in terms of precision, while also retaining a sufficient number of features to ensure a robust model. As a consequence, the features dropped were 'Age', 'Gender', 'Arrival Delay in minutes', 'Departure Delay in minutes', 'Track location rating' and 'Departure Arrival Time Rating'.

## RESULTS
