# Predictive Quality for Arc Welding - Time Series
## Seminar "Selected Topics in Data Science"

## 1. ABSTRACT
Gas Metal Arc Welding (GMAW) is a popular welding technique in the industry, also known as metal inert gas (MIG) welding. The quality assessment of welding samples by non-destructive techniques has several restrictions. Two prominent parameters in the welding process are current and voltage. Sensors capture these parameters in the form of a signal, which can be further digitized and analyzed with the supervised machine learning algorithm for evaluation of welding quality. This approach can be recommended in real-time welding quality assessment.

## 2. KEYWORDS
GMAW, MIG, Logistic Regression, Random Forest, Decision Tree, k-nearest neighbors, Supervised machine learning algorithm, scikit-learn, Hyperparameter tuning

## 3. CONCLUSION AND OUTLOOK
In this paper, I have implemented the four supervised machine learning algorithms Random Forest, Decision Tree, Logistic Regression, and 𝑘-Nearest Neighbors on sensor data from 32 welding procedures with  different types of error patterns. I observed that the Logistic Regression Algorithm has shown maximum accuracy for testing the new data set with nearly 0.99. I also applied hyper-parameter tuning with the GridSearchCV method for optimizing and verifying the performance of all machine learning algorithms.
All algorithms show the slight improvement in the Test accuracy. The validation accuracy is lower than the test accuracy after hyperparameter tuning. The possible causes would be over-fitting of the training data, a small validation set, data mismatch, and randomness in model evaluation.

In future work, I am also interested to explore any novel data-set related to GMAW with some additional electrical and mechanical parameters of welding process for better predictions of welding quality.

# 4. Implementation of Machine Learning Algorithms

## 4.1 Random Forest Classifier

The code implements a Random Forest Classifier using scikit-learn. It includes model training, validation, testing, evaluation, and hyperparameter tuning.

### Dependencies

| Component   | Description                 |
|-------------|-----------------------------|
| Dependencies| `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.metrics.accuracy_score`, `sklearn.metrics.confusion_matrix`, `sklearn.metrics.classification_report`, `sklearn.model_selection.cross_val_score` |

### Model Training and Testing

| Component                 | Description                                                                                                          |
|---------------------------|----------------------------------------------------------------------------------------------------------------------|
| `train_x_rf`, `val_x_rf`, `test_x_rf`  | Reshaped versions of `train_x`, `val_x`, and `test_x` to match the expected input shape for Random Forest classifier |
| `rf_classifier`            | `RandomForestClassifier` object                                                                                      |
| `rf_classifier.fit`        | Trains the Random Forest classifier on `train_x_rf` and `train_y`                                                    |
| `val_predictions`, `test_predictions` | Predicts labels for validation and test data using Random Forest classifier                                          |
| `val_accuracy`, `test_accuracy` | Calculates accuracy of predictions on validation and test data                                                |

### Evaluation Metrics

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `test_cm`                          | Confusion matrix for test data obtained using `confusion_matrix` function                                          |
| Plotting the Confusion Matrix         | Uses `seaborn`'s `heatmap` to plot the confusion matrix for test data                                                |
| `test_classification_report`       | Generates classification report for test data using `classification_report` function                               |

### Cross-validation

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Cross-validation                     | Performs 5-fold cross-validation on Random Forest classifier using `cross_val_score`                              |
| `average_cv_score`                  | Average cross-validation score obtained from `cv_scores`                                                       |

### Hyperparameter Tuning

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `rf_classifier`                       | `RandomForestClassifier` object                                                                                  |
| `param_grid`                           | Dictionary defining the hyperparameters grid for grid search                                                     |
| `grid_search`                          | `GridSearchCV` object for hyperparameter tuning                                                                    |
| `grid_search.fit`                      | Performs grid search to find the best parameters for Random Forest classifier                                    |
| `best_params`                          | Prints the best parameters found during grid search                                                              |

### Best Estimator

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `val_predictions`, `test_predictions`     | Predicts labels for validation and test data using the best estimator                                            |
| `val_accuracy`, `test_accuracy`           | Calculates accuracy of predictions using the best estimator                                                      |

## 4.2 Decision Tree Classification Algorithm

The code implements a decision tree classification algorithm using scikit-learn. It includes feature extraction, model training, validation, testing, evaluation, and hyperparameter tuning.

### Dependencies

| Component   | Description                                      |
|-------------|--------------------------------------------------|
| Dependencies| `seaborn`, `sklearn`, `numpy`, `matplotlib`       |
|             | `sklearn.neighbors.KNeighborsClassifier`         |
|             | `sklearn.metrics.accuracy_score`                 |
|             | `sklearn.metrics.classification_report`          |
|             | `sklearn.model_selection.cross_val_score`        |

### Feature Extraction Function

| Component                   | Description                                                 |
|-----------------------------|-------------------------------------------------------------|
| Feature Extraction Function | Calculates mean and standard deviation along rows of input `x` to create feature matrix |

### Extracting Features

| Component           | Description                             |
|---------------------|-----------------------------------------|
| Extracting Features | Extracts features from training, validation, and test datasets |

### Decision Tree Classifier

| Component                | Description                                         |
|--------------------------|-----------------------------------------------------|
| Decision Tree Classifier | Creates and trains decision tree classifier using `DecisionTreeClassifier` |

### Validation

| Component     | Description                                                 |
|---------------|-------------------------------------------------------------|
| Validation    | Predicts labels for validation dataset and calculates accuracy |

### Testing

| Component | Description                                        |
|-----------|----------------------------------------------------|
| Testing   | Predicts labels for test dataset and calculates accuracy |

### Classification Report

| Component            | Description                                                |
|----------------------|------------------------------------------------------------|
| Classification Report| Generates report based on test predictions and true labels |

### Confusion Matrix

| Component          | Description                                                    |
|--------------------|----------------------------------------------------------------|
| Confusion Matrix   | Computes confusion matrix based on test predictions and true labels |

### Confusion Matrix Visualization

| Component                  | Description                                                                      |
|----------------------------|----------------------------------------------------------------------------------|
| Confusion Matrix Visualization | Uses `sns.heatmap` to visualize confusion matrix                                  |

### Cross-validation

| Component          | Description                                                               |
|--------------------|---------------------------------------------------------------------------|
| Cross-validation   | Performs cross-validation using `cross_val_score` with 5-fold cross-validation |

### Hyperparameter Tuning

| Component           | Description                                                                   |
|---------------------|-------------------------------------------------------------------------------|
| Hyperparameter Tuning| Tunes hyperparameters using grid search (`GridSearchCV`) to find best parameters |

### Best Estimator

| Component       | Description                                                                |
|-----------------|----------------------------------------------------------------------------|
| Best Estimator  | Uses best estimator from grid search to predict labels for validation and test datasets |

### Accuracy Evaluation

| Component             | Description                                                                   |
|-----------------------|-------------------------------------------------------------------------------|
| Accuracy Evaluation   | Calculates and prints accuracy of predictions from best estimator              |

## 4.3 Logistic Regression Algorithm

The code implements a logistic regression algorithm using scikit-learn. It includes data preprocessing, model training, validation, testing, evaluation, and hyperparameter tuning.

### Dependencies

| Component   | Description                                                                                    |
|-------------|-----------------------------------------------------------------------------------------------|
| Dependencies| `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.linear_model.LogisticRegression`, `sklearn.metrics.classification_report`, `sklearn.metrics.confusion_matrix`, `sklearn.metrics.accuracy_score`, `sklearn.model_selection.cross_val_score` |

### Data Preprocessing

| Component   | Description                                                                                           |
|-------------|-------------------------------------------------------------------------------------------------------|
| `train_x_lr`, `val_x_lr`, `test_x_lr`  | Reshaped versions of `train_x`, `val_x`, and `test_x` to match the expected input shape for logistic regression  |

### Model Training and Testing

| Component       | Description                                                                   |
|-----------------|-------------------------------------------------------------------------------|
| `lr_classifier`   | `LogisticRegression` object with `max_iter=5000`                                   |
| `lr_classifier.fit` | Trains the logistic regression classifier on `train_x_lr` and `train_y`              |
| `val_predictions`, `test_predictions`   | Predicts labels for validation and test data using logistic regression classifier |
| `val_accuracy`, `test_accuracy`   | Calculates accuracy of predictions on validation and test data                   |

### Evaluation Metrics

| Component                        | Description                                                                        |
|----------------------------------|------------------------------------------------------------------------------------|
| `test_cm`                           | Confusion matrix for test data obtained using `confusion_matrix` function            |
| Plotting the Confusion Matrix     | Uses `seaborn`'s `heatmap` to plot the confusion matrix for test data                  |
| `test_classification_report`        | Generates classification report for test data using `classification_report` function |

### Cross-validation

| Component                        | Description                                                                        |
|----------------------------------|------------------------------------------------------------------------------------|
| Cross-validation                 | Performs 5-fold cross-validation on logistic regression classifier using `cross_val_score` |
| `average_cv_score`                  | Average cross-validation score obtained from `cv_scores`                            |

### Hyperparameter Tuning

| Component                        | Description                                                                        |
|----------------------------------|------------------------------------------------------------------------------------|
| `lr_classifier`                    | `LogisticRegression` object                                                                                      |
| `param_grid`                        | Dictionary defining the hyperparameters grid for grid search                                            |
| `grid_search`                       | `GridSearchCV` object for hyperparameter tuning                                                                    |
| `grid_search.fit`                      | Performs grid search to find the best parameters for logistic regression classifier                                    |
| `best_params`                          | Prints the best parameters found during grid search                                                              |

### Best Estimator

| Component                         | Description                                                                   |
|-----------------------------------|-------------------------------------------------------------------------------|
| `val_predictions_lr`, `test_predictions_lr`   | Predicts labels for validation and test data using the best estimator           |
| `val_accuracy_lr`, `test_accuracy_lr` | Calculates accuracy of predictions using the best estimator                     |

## 4.4 *k*-NN Classification Algorithm

The code implements a k-Nearest Neighbors (k-NN) classification algorithm using scikit-learn. It includes model training, validation, testing, evaluation, and hyperparameter tuning.

### Dependencies

| Component   | Description                                                                                   |
|-------------|-----------------------------------------------------------------------------------------------|
| Dependencies| `sklearn.neighbors.KNeighborsClassifier`, `sklearn.metrics.accuracy_score`, `sklearn.metrics.classification_report`, `sklearn.model_selection.cross_val_score` |

### Model Training and Testing

| Component                 | Description                                                                                                          |
|---------------------------|----------------------------------------------------------------------------------------------------------------------|
| `train_x_knn`, `val_x_knn`, `test_x_knn`  | Reshaped versions of `train_x`, `val_x`, and `test_x` to match the expected input shape for k-NN classifier          |
| `knn_classifier`            | `KNeighborsClassifier` object with `n_neighbors=3`                                                                        |
| `knn_classifier.fit`        | Trains the k-NN classifier on `train_x_knn` and `train_y`                                                                 |
| `val_predictions_knn`, `test_predictions_knn` | Predicts labels for validation and test data using k-NN classifier                                          |
| `val_accuracy_knn`, `test_accuracy_knn` | Calculates accuracy of predictions on validation and test data                                                |

### Evaluation Metrics

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `test_cm_knn`                          | Confusion matrix for test data obtained using `confusion_matrix` function                                          |
| Plotting the Confusion Matrix         | Uses `seaborn`'s `heatmap` to plot the confusion matrix for test data                                                |
| `test_classification_report_knn`       | Generates classification report for test data using `classification_report` function                               |

### Cross-validation

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Cross-validation                     | Performs 5-fold cross-validation on k-NN classifier using `cross_val_score`                                       |
| `average_cv_score_knn`                  | Average cross-validation score obtained from `cv_scores_knn`                                                       |

### Hyperparameter Tuning

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `knn_classifier`                       | `KNeighborsClassifier` object                                                                                      |
| `param_grid`                           | Dictionary defining the hyperparameters grid for grid search                                                     |
| `grid_search`                          | `GridSearchCV` object for hyperparameter tuning                                                                    |
| `grid_search.fit`                      | Performs grid search to find the best parameters for k-NN classifier                                             |
| `best_params`                          | Prints the best parameters found during grid search                                                              |

### Best Estimator

| Component                            | Description                                                                                                      |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `val_predictions`, `test_predictions`     | Predicts labels for validation and test data using the best estimator                                            |
| `val_accuracy`, `test_accuracy`           | Calculates accuracy of predictions using the best estimator                                                      |

---

## Leistungsbescheinigung

![Leistungsbescheinigung](https://github.com/dhananjayjm/Predictive-Quality-for-Arc-Welding---Time-Series/blob/main/Leistungsbescheinigung.png)

