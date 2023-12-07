import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_loader import load_data_from_csv
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt



def preprocess_data(data, columns_to_scale=None, date_columns=None):
    """
    Preprocess the data by converting date columns to Unix timestamps and
    scaling specified numeric columns using StandardScaler.

    Parameters:
    - data (DataFrame): The pandas DataFrame containing the data to be preprocessed.
    - columns_to_scale (list of str, optional): List of numeric column names to be scaled. If None, no scaling is applied.
    - date_columns (list of str, optional): List of date column names to be converted to Unix timestamps. If None, no conversion is applied.

    Returns:
    - DataFrame: The preprocessed pandas DataFrame.
    """
    if date_columns:
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            data[col] = data[col].astype('int64') // 10**9  # Convert to Unix timestamp
    if columns_to_scale:
        scaler = StandardScaler()
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale].astype(float))
    return data

# Load and preprocess data
raw_data = load_data_from_csv(r'C:\Users\PC\Downloads\OD_2019-07.csv')
processed_data = preprocess_data(raw_data, columns_to_scale=['duration_sec'], date_columns=['start_date', 'end_date'])

def get_model_pipeline(model, scaler=None):
    """
    Creates a machine learning pipeline consisting of an optional scaler and a model.

    Parameters:
    - model (estimator): The scikit-learn estimator to be used in the pipeline.
    - scaler (transformer, optional): An instance of a scaler if scaling is required in the pipeline. Default is None.

    Returns:
    - Pipeline: A scikit-learn Pipeline object.
    """
    steps = [('scaler', scaler)] if scaler else []
    steps.append(('model', model))
    pipeline = Pipeline(steps)
    return pipeline

def perform_cross_validation(pipeline, X, y, cv=5):
    """
    Performs cross-validation on the given pipeline and data.

    Parameters:
    - pipeline (Pipeline): The scikit-learn Pipeline object to be evaluated.
    - X (array-like): The input features for model training.
    - y (array-like): The target variable for model training.
    - cv (int, optional): The number of folds in cross-validation. Default is 5.

    Returns:
    - float: The mean score of the cross-validation.
    """
    cv_scores = cross_val_score(pipeline, X, y, cv=cv)
    return cv_scores.mean()


# Sampling a smaller subset of the data for quicker execution
sampled_data = processed_data.sample(frac=0.5, random_state=42)

# Splitting data into features and target
X = sampled_data.drop('is_member', axis=1)
y = sampled_data['is_member']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check data types of features
print(X.dtypes)

# Create a standard scaler instance
scaler = StandardScaler()

# Define a simpler model for faster execution
model = LogisticRegression()

# Get the pipeline
pipeline = get_model_pipeline(model, scaler)

# Fit the model using the training data
pipeline.fit(X_train, y_train)

# Perform cross-validation and print the mean score
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Model evaluation
y_pred = pipeline.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()