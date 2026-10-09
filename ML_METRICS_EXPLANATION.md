# **Machine Learning Evaluation Metrics Explained**

## **Overview**

This document explains the key metrics used to evaluate machine learning models in our sepsis prediction project. Understanding these metrics is crucial for interpreting model performance, especially in imbalanced medical datasets.

---

## **Mathematical Formulas & Methods**

### **Confusion Matrix Components**

From the confusion matrix, we derive:
- **TP (True Positive)**: Correctly predicted sepsis
- **TN (True Negative)**: Correctly predicted no sepsis
- **FP (False Positive)**: Incorrectly predicted sepsis (false alarm)
- **FN (False Negative)**: Missed sepsis case (most dangerous!)

### **Core Metric Formulas**

**1. Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**2. Sensitivity (Recall / True Positive Rate)**
```
Sensitivity = TP / (TP + FN)
```
- Measures: Out of all actual sepsis cases, how many did we catch?

**3. Specificity (True Negative Rate)**
```
Specificity = TN / (TN + FP)
```
- Measures: Out of all actual non-sepsis cases, how many did we correctly identify?

**4. Precision (Positive Predictive Value)**
```
Precision = TP / (TP + FP)
```
- Measures: When we predict sepsis, how often are we correct?

**5. False Positive Rate**
```
FPR = FP / (FP + TN) = 1 - Specificity
```

**6. F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of Precision and Recall

**7. Brier Score**
```
Brier Score = (1/n) × Σ(pi - yi)²
```
Where:
- `pi` = predicted probability for instance i
- `yi` = actual outcome (0 or 1) for instance i
- `n` = number of instances
- Lower is better (0 = perfect calibration)

### **ROC Curve Calculation**

**ROC Curve** plots:
- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis**: True Positive Rate (TPR) = TP / (TP + FN) = Sensitivity

**AUROC Calculation:**
```
AUROC = ∫ TPR(FPR) dFPR
```
- Calculated using trapezoidal rule or integration
- Represents the probability that a randomly chosen positive instance ranks higher than a randomly chosen negative instance

### **Precision-Recall Curve Calculation**

**PR Curve** plots:
- **X-axis**: Recall = TP / (TP + FN) = Sensitivity
- **Y-axis**: Precision = TP / (TP + FP)

**AUPRC Calculation:**
```
AUPRC = ∫ Precision(Recall) dRecall
```
- Calculated using trapezoidal rule
- More informative for imbalanced datasets

### **Optimal Threshold Selection Methods**

**1. Youden's J Index**
```
J = Sensitivity + Specificity - 1
Optimal threshold = argmax(J)
```
- Maximizes the sum of sensitivity and specificity
- Balances both metrics equally

**2. F1-Maximization**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Optimal threshold = argmax(F1)
```
- Maximizes the harmonic mean of precision and recall
- Used in our baseline models

**3. Cost-Sensitive Threshold**
```
Cost = (Cost_FN × FN) + (Cost_FP × FP)
Optimal threshold = argmin(Cost)
```
- Minimizes total cost based on clinical consequences
- Can be customized for different clinical scenarios

### **Model-Specific Methods**

**1. Logistic Regression**
```
P(Y=1|X) = 1 / (1 + e^(-z))
where z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```
- Uses sigmoid function for probability estimation
- L2 regularization: `C` parameter controls regularization strength
- Loss function: Binary Cross-Entropy

**2. Random Forest**
```
Prediction = Mode or Mean of {Tree₁(X), Tree₂(X), ..., Treeₙ(X)}
```
- Ensemble of decision trees
- Each tree votes, majority wins (classification) or average (regression)
- Hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

**3. XGBoost**
```
Fₘ(x) = Fₘ₋₁(x) + αₘ × hₘ(x)
```
- Gradient boosting: sequentially adds weak learners
- Loss function: `logloss` for binary classification
- Hyperparameters: `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`

### **Preprocessing Methods**

**1. Median Imputation**
```
X_missing = median(X_observed)
```
- Replaces missing values with median of observed values
- Robust to outliers

**2. Standard Scaling (Z-score Normalization)**
```
X_scaled = (X - μ) / σ
```
Where:
- `μ` = mean of feature
- `σ` = standard deviation of feature
- Results in mean=0, std=1

**3. Feature Selection (SelectKBest)**
```
Score(feature) = Mutual Information(feature, target)
Select top K features with highest scores
```
- Uses mutual information to rank features
- Selects K features with highest information gain

### **Class Imbalance Handling**

**1. SMOTE (Synthetic Minority Oversampling Technique)**

**What it is:**
SMOTE is an oversampling technique that creates synthetic samples from the minority class (sepsis cases) to balance the dataset.

**Algorithm:**
```
For each minority class sample x:
  1. Find k nearest neighbors from minority class (typically k=5)
  2. Randomly select one neighbor x_neighbor
  3. Generate random number λ between 0 and 1
  4. Create synthetic sample: x_new = x + λ × (x_neighbor - x)
```

**Mathematical Formula:**
```
x_synthetic = x_i + random(0,1) × (x_j - x_i)
```
Where:
- `x_i` = original minority sample
- `x_j` = randomly selected k-nearest neighbor from minority class
- `random(0,1)` = random number between 0 and 1

**Why it works:**
- Creates new samples **along the line segment** between existing minority samples
- Generates samples in the **feature space** where minority class exists
- More sophisticated than simple duplication (which can cause overfitting)

**Advantages:**
- ✅ Increases minority class representation
- ✅ Creates diverse synthetic samples
- ✅ Better than random oversampling (duplication)
- ✅ Helps models learn better decision boundaries

**Disadvantages:**
- ⚠️ Can create noisy samples if k is too large
- ⚠️ May not work well with high-dimensional data
- ⚠️ Can amplify existing noise in minority class

**In our project:**
- Used specifically for **Logistic Regression**
- Applied only to training data (not validation/test)
- k_neighbors = min(5, number_of_positive_samples - 1)
- Applied after train-validation split to prevent data leakage

**Example:**
If we have 1000 negative cases and 20 positive cases:
- Without SMOTE: 20 positive, 1000 negative (2% positive)
- With SMOTE: 1000 positive (synthetic), 1000 negative (50% positive)
- Model can now learn better from balanced data

**Alternative approaches:**
- **Random Forest & XGBoost**: Use class weights instead (no SMOTE needed)
- **Undersampling**: Remove majority class samples (loses data)
- **Combined**: SMOTE + undersampling (SMOTEENN)

**2. Class Weights**
```
Weight_positive = n_total / (2 × n_positive)
Weight_negative = n_total / (2 × n_negative)
```
- Adjusts loss function to penalize misclassifying minority class more
- Used for Random Forest and XGBoost

### **Calibration Methods**

**1. Isotonic Calibration**
```
P_calibrated = IsotonicRegression().fit_transform(P_raw, Y_true)
```
- Non-parametric method that fits a stepwise constant function
- Ensures predicted probabilities match actual event rates
- Used in our baseline models

**2. Platt Scaling (Logistic Calibration)**
```
P_calibrated = 1 / (1 + exp(-(A × logit(P_raw) + B)))
```
- Parametric method using logistic regression
- Simpler but less flexible than isotonic

### **Cross-Validation**

**Stratified K-Fold**
```
Split data into K folds maintaining class distribution
For each fold:
  Train on K-1 folds, validate on 1 fold
Average performance across all folds
```
- Ensures each fold has similar class distribution
- Used for hyperparameter tuning

### **Patient-Level Splitting**

```
Unique_patients = unique(Patient_ID)
Train_patients, Test_patients = split(Unique_patients)
Train_data = all_records where Patient_ID in Train_patients
Test_data = all_records where Patient_ID in Test_patients
```
- Prevents data leakage
- Ensures no patient appears in both train and test sets
- Critical for medical data

---

## **1. ROC vs. AUC vs. AUROC - Understanding the Terminology**

### **Key Differences:**

**ROC (Receiver Operating Characteristic Curve)**
- This is the **actual curve/plot** itself
- A graphical representation showing the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
- X-axis: False Positive Rate (1 - Specificity)
- Y-axis: True Positive Rate (Sensitivity)
- **It's a visual tool** - you can see it plotted on a graph

**AUC (Area Under the Curve)**
- This is a **general term** meaning "area under any curve"
- Could refer to area under ROC curve, PR curve, or any other curve
- **Generic term** - needs context to know which curve

**AUROC (Area Under the ROC Curve)**
- This is **specifically** the area under the ROC curve
- A **single number** (0.0 to 1.0) that summarizes the ROC curve
- **Most specific term** - clearly indicates we're talking about the ROC curve area

### **In Practice:**
- **ROC** = The curve/plot (visual)
- **AUROC** = The area under that curve (single number/metric)
- **AUC** = Often used interchangeably with AUROC, but technically more general

### **Common Usage:**
- People often say "AUC" when they mean "AUROC" (area under ROC curve)
- In our project, we use **AUROC** to be precise
- When we say "AUC" in context of ROC curves, we mean AUROC

---

## **2. AUROC (Area Under the ROC Curve)**

### **What it is:**
- **Full name**: Area Under the Receiver Operating Characteristic Curve
- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - 0.5 = Random guessing (no better than flipping a coin)
  - 0.7-0.8 = Acceptable performance
  - 0.8-0.9 = Excellent performance
  - 0.9-1.0 = Outstanding performance
  - 1.0 = Perfect discrimination

### **What it measures:**
The model's ability to distinguish between positive (sepsis) and negative (no sepsis) cases across all possible classification thresholds.

### **ROC Curve:**
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity)
- The curve shows the trade-off between sensitivity and specificity at different thresholds

### **Why it's important:**
- **Primary metric** for model comparison
- **Threshold-independent**—evaluates performance across all possible thresholds
- **Robust** to class imbalance (unlike accuracy)
- **Clinical relevance**: Higher AUROC means better ability to separate high-risk from low-risk patients

### **Example:**
- AUROC of 0.63 means: "If I randomly pick one sepsis patient and one non-sepsis patient, the model will correctly rank the sepsis patient as higher risk 63% of the time"

---

## **2. AUPRC (Area Under the Precision-Recall Curve)**

### **What it is:**
- **Full name**: Area Under the Precision-Recall Curve
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better

### **What it measures:**
The trade-off between precision and recall (sensitivity) across different thresholds.

### **PR Curve:**
- **X-axis**: Recall (Sensitivity)
- **Y-axis**: Precision
- Shows how precision changes as we adjust the threshold to catch more cases

### **Why it's important:**
- **Better than AUROC for imbalanced datasets** (like ours with 2% positive cases)
- Focuses on the minority class (sepsis cases)
- More informative when positive cases are rare
- **Clinical relevance**: Shows how well the model performs on the cases we care about most

### **Example:**
- AUPRC of 0.04 means: "When trying to identify sepsis cases, the model achieves a precision-recall trade-off that results in an area of 0.04"
- Low AUPRC is common in highly imbalanced datasets

---

## **3. Accuracy**

### **What it is:**
- **Formula**: (True Positives + True Negatives) / Total Predictions
- **Range**: 0.0 to 1.0 (or 0% to 100%)
- **Interpretation**: Percentage of correct predictions

### **What it measures:**
Overall correctness of predictions.

### **Limitations:**
- **Misleading with imbalanced data**: A model that always predicts "no sepsis" would have 98% accuracy in our dataset
- Doesn't distinguish between types of errors
- Doesn't show performance on the minority class

### **When to use:**
- Balanced datasets
- When all types of errors are equally important
- As a secondary metric alongside others

### **Example:**
- Accuracy of 97.5% means: "The model correctly predicts 97.5% of all cases"
- But this doesn't tell us if it's catching sepsis cases!

---

## **4. Sensitivity (Recall / True Positive Rate)**

### **What it is:**
- **Formula**: True Positives / (True Positives + False Negatives)
- **Alternative names**: Recall, True Positive Rate, Hit Rate
- **Range**: 0.0 to 1.0 (or 0% to 100%)
- **Interpretation**: Higher is better

### **What it measures:**
The ability to correctly identify sepsis cases (how many sepsis patients we catch).

### **Clinical importance:**
- **Critical for sepsis prediction**: Missing a sepsis case can be fatal
- Measures "sensitivity" to detecting the condition
- High sensitivity = low false negative rate = fewer missed cases

### **Example:**
- Sensitivity of 42% means: "Out of 100 sepsis patients, the model correctly identifies 42 of them"
- This is concerning—we're missing 58% of sepsis cases!

---

## **5. Specificity (True Negative Rate)**

### **What it is:**
- **Formula**: True Negatives / (True Negatives + False Positives)
- **Alternative names**: True Negative Rate
- **Range**: 0.0 to 1.0 (or 0% to 100%)
- **Interpretation**: Higher is better

### **What it measures:**
The ability to correctly identify non-sepsis cases (how many healthy patients we correctly identify as healthy).

### **Clinical importance:**
- Reduces false alarms
- Prevents unnecessary interventions
- High specificity = low false positive rate = fewer false alarms

### **Example:**
- Specificity of 95.8% means: "Out of 100 non-sepsis patients, the model correctly identifies 95.8 of them as healthy"
- Only 4.2% false alarm rate

---

## **6. Precision (Positive Predictive Value)**

### **What it is:**
- **Formula**: True Positives / (True Positives + False Positives)
- **Alternative names**: Positive Predictive Value (PPV)
- **Range**: 0.0 to 1.0 (or 0% to 100%)
- **Interpretation**: Higher is better

### **What it measures:**
When the model predicts sepsis, how often is it correct?

### **Clinical importance:**
- Reduces unnecessary interventions
- Measures trustworthiness of positive predictions
- High precision = when we alert, we're usually right

### **Example:**
- Precision of 6.33% means: "Out of 100 patients the model flags as sepsis, only 6.33 actually have sepsis"
- This is very low—lots of false alarms!

---

## **7. F1-Score**

### **What it is:**
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better

### **What it measures:**
Harmonic mean of precision and recall—balances both metrics.

### **Why it's useful:**
- Single metric that considers both precision and recall
- Useful when you need to balance catching cases vs. avoiding false alarms
- Penalizes models that are good at one but poor at the other

### **Example:**
- F1-Score of 0.085 means: "The model achieves a balanced precision-recall performance of 0.085"
- Low F1 indicates either low precision, low recall, or both

---

## **8. Brier Score**

### **What it is:**
- **Formula**: Mean squared difference between predicted probabilities and actual outcomes
- **Range**: 0.0 to 1.0
- **Interpretation**: Lower is better (0 = perfect calibration)

### **What it measures:**
Accuracy of predicted probabilities (calibration quality).

### **Why it's important:**
- Measures if predicted probabilities are reliable
- A model with good AUROC but poor Brier Score makes unreliable probability estimates
- Critical for clinical decision-making based on risk scores

### **Example:**
- Brier Score of 0.24 means: "The average squared error between predicted probabilities and actual outcomes is 0.24"
- Lower is better—0.24 indicates moderate calibration

---

## **Confusion Matrix Overview**

To understand these metrics, it helps to visualize the confusion matrix:

```
                    Predicted
                 No Sepsis  Sepsis
Actual  No Sepsis    TN      FP
        Sepsis       FN      TP
```

Where:
- **TP (True Positive)**: Correctly predicted sepsis
- **TN (True Negative)**: Correctly predicted no sepsis
- **FP (False Positive)**: Incorrectly predicted sepsis (false alarm)
- **FN (False Negative)**: Missed sepsis case (most dangerous!)

---

## **Metric Relationships**

### **Sensitivity vs. Specificity:**
- **Trade-off**: Increasing sensitivity usually decreases specificity
- **Threshold adjustment**: Lower threshold = higher sensitivity, lower specificity
- **Clinical decision**: Choose threshold based on clinical priorities

### **Precision vs. Recall:**
- **Trade-off**: Increasing recall usually decreases precision
- **High recall**: Catch more cases but more false alarms
- **High precision**: Fewer false alarms but miss more cases

### **AUROC vs. AUPRC:**
- **AUROC**: Better for balanced datasets or when both classes matter equally
- **AUPRC**: Better for imbalanced datasets (like ours with 2% positive cases)
- **Use both**: AUROC for overall discrimination, AUPRC for minority class performance

---

## **Which Metrics Matter Most for Sepsis Prediction?**

### **Primary Metrics:**
1. **AUROC** - Overall discrimination ability
2. **Sensitivity** - Critical! We must catch sepsis cases
3. **AUPRC** - Performance on the rare but important class

### **Secondary Metrics:**
4. **Specificity** - Important to reduce false alarms
5. **Precision** - Important for clinical trust
6. **F1-Score** - Balanced view of precision and recall

### **Calibration Metrics:**
7. **Brier Score** - Ensures probabilities are reliable
8. **Calibration Curves** - Visual check of probability accuracy

---

## **Interpreting Our Baseline Model Results**

### **Logistic Regression:**
- **AUROC: 0.6301** - Best discrimination (63% better than random)
- **Sensitivity: 12.8%** - Very conservative, misses many cases
- **Specificity: 95.8%** - Excellent at avoiding false alarms
- **Interpretation**: Conservative model, good for reducing false alarms but misses many sepsis cases

### **Random Forest:**
- **AUROC: 0.5949** - Moderate discrimination
- **Sensitivity: 42.12%** - Best at catching sepsis cases
- **Specificity: 69.58%** - Moderate false alarm rate
- **Interpretation**: Balanced model, best at detecting sepsis but with more false alarms

### **XGBoost:**
- **AUROC: 0.5466** - Lower discrimination
- **Sensitivity: 40.85%** - Good at catching cases
- **Specificity: 67.57%** - Moderate false alarm rate
- **Interpretation**: Similar to Random Forest, good sensitivity

---

## **Key Takeaways**

1. **No single metric tells the whole story** - Use multiple metrics together
2. **Class imbalance matters** - Accuracy can be misleading
3. **Clinical context matters** - Sensitivity is critical for sepsis
4. **Trade-offs exist** - Can't maximize all metrics simultaneously
5. **Threshold selection matters** - Different thresholds optimize different metrics
6. **Calibration matters** - Reliable probabilities are crucial for clinical use

---

## **References**

- **AUROC**: Receiver Operating Characteristic (ROC) analysis
- **AUPRC**: Precision-Recall curves for imbalanced classification
- **Confusion Matrix**: Standard classification evaluation framework
- **Brier Score**: Probabilistic prediction evaluation

---

**For questions or clarifications, refer to the model evaluation results in `baseline_models_metrics.json`**

