# Missing Values Handling
MSc coursework project (Modelling of Big Data in Business and Finance) exploring methods for identifying and handling missing data in a business analytics context.

**Approach:**
  - Identified and categorised missing data patterns in the dataset
  - Compared four strategies for handling missing values:
  1. Listwise deletion (removing rows with missing values)
  2. Median imputation (column-level)
  3. Mean imputation with grouping (imputing based on the mean of a selected column, grouped by category)
  4. Median imputation with grouping (imputing based on the median of a selected column, grouped by category)
  - Evaluated the impact of each strategy on downstream model performance using **Logistic Regression** and **K-Nearest Neighbors**, each with **k-fold cross-validation**

**Tools:** Python (Pandas, NumPy, scikit-learn)

**Outcome:** Grouped median imputation combined with Logistic Regression (k-fold cross-validation) delivered the best performance, outperforming both simpler imputation strategies and the KNN model.
