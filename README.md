
# Phishing URL detection from Web Characteristics using Machine Learning Techniques.





### Dataset acknowledgement:
```
Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. doi: https://doi.org/10.1016/j.cose.2023.103545
```
 
### Dataset Link:
 UC Irvine Machine Learning Repository:  
[Dataset Source Link](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

---

### Dataset Information:
- Labeled - YES  
- Classification Type: Binary Classification  
- Number of Classes: Phishing (0) and Legitimate (1)  
- Number of Samples: 235,795
- Number of Features: 56
---


## Data Preprocessing

1. Data Cleaning  
Drops irrelevant columns: FILENAME, URL, DOMAIN, TLD, TITLE to avoid training issues and focus on numerical features.

    Selects only numeric columns via 
    ```
    df.select_dtypes(include=['number']).copy().
    ```

    Removes duplicate rows with 
    ```
    df.drop_duplicates().​
    ```

2. Train-Test Split  
    Splits features (X: all but last column) and target (y: last column) using train_test_split(test_size=0.2, random_state=42, stratify=y).

    Keeping 20% for Testing and Training the rest of the 80%.

3. Feature Scaling  
    Applies StandardScaler fitted on training data only, then transforms both train and test sets.

    Converts scaled data back to DataFrames preserving column names and indices.​

4. Handling Class Imbalance  
    Compares oversampling methods: SMOTE, ADASYN, BorderlineSMOTE by checking class distributions (ytrain.value_counts()).

    Applies SMOTE (SMOTE(random_state=42).fit_resample()) to create balanced training sets (Xtrainresampled, ytrainresampled).​

5. Feature Selection  
    Uses SelectKBest with ANOVA F-test (f_classif, k=20) on resampled training data.

    Selects top features, applies to test set, and visualizes F-scores with a horizontal bar plot.

    Outputs selected feature names and shapes for train/test selected sets.​


    
## Feature Selection

Uses SelectKBest with ANOVA F-test (f_classif, k=20) on resampled training data. Selects top features, applies to test set, and visualizes F-scores with a horizontal bar plot. Outputs selected feature names and shapes for train/test selected sets.​

## Training

- __Decision Tree classifier:__  
 is trained using GridSearchCV to tune its hyperparameters. The model is evaluated with 3-fold cross-validation and optimized based on the F1-score, making it suitable for imbalanced datasets. The best-performing Decision Tree model is then selected and saved for later use.

- __Random Forest classifier:__  
is trained using GridSearchCV with a predefined parameter grid consisting of 100 trees and a maximum depth of 20. The model is evaluated using 3-fold cross-validation and optimized based on the F1-score to account for class imbalance. After training, the best Random Forest model identified by the grid search is saved for later use.

 - __Logistic Regression Model:__
 is trained using GridSearchCV with an L2 regularization penalty and a regularization strength (C) set to 1. The model is evaluated using 3-fold cross-validation and optimized based on the F1-score to handle class imbalance. A higher maximum number of iterations is specified to ensure convergence, and the best-performing Logistic Regression model is selected and saved for later use.

 - __K-Nearest Neighbors (KNN):__  
 The K-Nearest Neighbors (KNN) classifier is trained using GridSearchCV with a parameter grid that specifies 5 neighbors and distance-based weighting. The model is evaluated using 3-fold cross-validation and optimized based on the F1-score to better handle class imbalance. After training, the best-performing KNN model identified by the grid search is selected and saved for later use.

- __Gradient Boosting Classifier:__  
The Gradient Boosting Classifier is trained using GridSearchCV to evaluate different model configurations, including the number of boosting stages, learning rate, and tree depth. Specifically, the grid search tests 100 estimators with a learning rate of 0.1 and compares maximum tree depths of 3 and 5. The model is evaluated using 3-fold cross-validation and optimized based on the F1-score to address class imbalance. The best-performing Gradient Boosting model is then selected and saved for later use.

- __Support Vector Machine (SVM):__  
The Support Vector Machine (SVM) classifier is trained using GridSearchCV with a radial basis function (RBF) kernel. The grid search evaluates a regularization parameter (C) set to 1 and uses the default scaled gamma value. The model is assessed using 3-fold cross-validation and optimized based on the F1-score to better handle class imbalance. Probability estimation is enabled to allow probabilistic predictions, and the best-performing SVM model is selected and saved for later use.

- __XGBoost classifier:__  
The XGBoost classifier is trained using GridSearchCV to evaluate different boosting configurations, including the number of trees, tree depth, learning rate, and subsampling ratio. Specifically, the grid search tests 100 estimators with a maximum depth of 5, a learning rate of 0.1, and subsample values of 0.8 and 1. The model is evaluated using 3-fold cross-validation and optimized based on the F1-score to handle class imbalance. After training, the best-performing XGBoost model is selected and saved for later use.

 - __Stacking Classifier:__  
 A Stacking Classifier is built by combining multiple previously trained models as base learners, including a Decision Tree, K-Nearest Neighbors, and Support Vector Machine. These base models generate predictions that are passed to a Logistic Regression meta-learner, which learns how to best combine their outputs. The stacking model is trained using 3-fold cross-validation, with passthrough enabled to include the original input features alongside base model predictions. After training, the complete stacking ensemble is saved for later use.

 ## Evaluation:  
 Each trained model is evaluated on both the training and test datasets to assess performance and generalization. The code generates class predictions and, when available, predicted probabilities for computing the AUC-ROC score. For models that do not support probability estimates, the decision function is used instead.

For each model, training and test accuracy are reported, along with a detailed classification report showing precision, recall, and F1-score. The AUC-ROC score is also calculated to evaluate the model’s ability to distinguish between classes. Additionally, a confusion matrix is generated and visualized as a heatmap to provide insight into true and false predictions for each class.

## LIME Explainable Ai
LIME is used to explain individual predictions made by each trained model. A random sample from the test set is selected, and LIME generates local feature importance scores based on each model’s predicted probabilities. The explanations are visualized and stored in tabular form, allowing comparison of how different models interpret the same input.

---
## Results:
<img src="/images/ANOVA-F-TEST.png">
<img src="/images/DesTree_Confusion_matrix.png">
<img src="/">
