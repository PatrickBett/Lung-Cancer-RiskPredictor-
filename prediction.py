import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_palette('deep')

print("Loading the Lung Cancer dataset...")

# Download dataset
dataset_path = kagglehub.dataset_download("mysarahmadbhat/lung-cancer")
print(f"Dataset downloaded to: {dataset_path}")

# Find the CSV file in the downloaded directory
csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in the downloaded dataset")

# Load the dataset
file_path = os.path.join(dataset_path, csv_files[0])
data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("\n--- Dataset Information ---")
print(f"Shape: {data.shape}")
print("\n--- First 5 rows ---")
print(data.head())

# Check for missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Data Information
print("\n--- Data Types ---")
print(data.dtypes)

# Ensure consistent column naming - dealing with spaces in column names
data.columns = [col.strip().replace(" ", "_") for col in data.columns]

# Define the LabelEncoder before using it
le = LabelEncoder()

# Process all categorical columns - convert YES/NO to 2/1 if needed
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    # Skip the target variable for now
    if col != 'LUNG_CANCER':
        print(f"Converting column {col} from categorical to numeric...")
        if data[col].nunique() == 2:
            # Check if it's binary YES/NO
            if set(data[col].unique()) == {'YES', 'NO'}:
                data[col] = data[col].map({'YES': 2, 'NO': 1})
            # Check for other common yes/no formats
            elif set(data[col].unique()) == {'Yes', 'No'}:
                data[col] = data[col].map({'Yes': 2, 'No': 1})
            elif set(data[col].unique()) == {'yes', 'no'}:
                data[col] = data[col].map({'yes': 2, 'no': 1})
            else:
                data[col] = le.fit_transform(data[col])
        else:
            data[col] = le.fit_transform(data[col])

# Handle GENDER explicitly (as it appears as an object type)
if 'GENDER' in data.columns and data['GENDER'].dtype == 'object':
    gender_mapping = {'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0, 'Male': 1, 'Female': 0, 'm': 1, 'f': 0}
    data['GENDER'] = data['GENDER'].map(gender_mapping)
    print("Converted GENDER column from categorical to numeric.")

# Convert target variable
if 'LUNG_CANCER' in data.columns:
    # Get unique values to understand the format
    unique_values = data['LUNG_CANCER'].unique()
    print(f"Unique values in LUNG_CANCER: {unique_values}")
    
    # Convert target variable based on its values
    if set(unique_values) == {'YES', 'NO'}:
        data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    elif set(unique_values) == {'Yes', 'No'}:
        data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'Yes': 1, 'No': 0})
    elif set(unique_values) == {'yes', 'no'}:
        data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'yes': 1, 'no': 0})
    else:
        data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])
    
    print(f"Converted LUNG_CANCER to binary values: {data['LUNG_CANCER'].unique()}")

print("\n--- Data Types After Conversion ---")
print(data.dtypes)

# Statistical summary
print("\n--- Statistical Summary ---")
print(data.describe())

# Exploratory Data Analysis
print("\n--- Performing Exploratory Data Analysis ---")

# Count of target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='LUNG_CANCER', data=data)
plt.title('Distribution of Lung Cancer Cases')
plt.xlabel('Lung Cancer (0: No, 1: Yes)')
plt.tight_layout()
plt.savefig('lung_cancer_distribution.png')
plt.close()

# Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='GENDER', hue='LUNG_CANCER', data=data)
plt.title('Gender Distribution by Lung Cancer Status')
plt.xlabel('Gender (0: Female, 1: Male)')
plt.tight_layout()
plt.savefig('gender_distribution.png')
plt.close()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='AGE', hue='LUNG_CANCER', kde=True, bins=20)
plt.title('Age Distribution by Lung Cancer Status')
plt.xlabel('Age')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.close()

# Risk factors analysis
risk_factors = [col for col in data.columns if col not in ['LUNG_CANCER', 'GENDER', 'AGE']]

# Create a figure with subplots for each risk factor
n_cols = 3
n_rows = (len(risk_factors) + n_cols - 1) // n_cols
plt.figure(figsize=(15, n_rows * 4))

for i, factor in enumerate(risk_factors, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(x=factor, hue='LUNG_CANCER', data=data)
    plt.title(f'{factor} vs Lung Cancer')
    plt.xlabel(factor)
    plt.xticks([0, 1] if set(data[factor].unique()) == {0, 1} else [1, 2], ['No', 'Yes'])
    plt.legend(['No Cancer', 'Cancer'])

plt.tight_layout()
plt.savefig('risk_factors.png')
plt.close()

# Correlation Matrix
print("\n--- Generating Correlation Matrix ---")
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Identify top correlated features with lung cancer
cancer_corr = correlation_matrix['LUNG_CANCER'].drop('LUNG_CANCER')
print("\n--- Top Features Correlated with Lung Cancer ---")
print(cancer_corr.sort_values(ascending=False))

# Prepare data for modeling
print("\n--- Preparing Data for Modeling ---")

# Split features and target
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Model Training and Evaluation
print("\n--- Training and Evaluating Multiple Models ---")

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_mean,
        'fpr': fpr,
        'tpr': tpr
    }
    
    # Print results
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"5-Fold CV Mean Accuracy: {cv_mean:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for name, result in results.items():
    plt.plot(result['fpr'], result['tpr'], label=f'{name} (AUC = {result["roc_auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

# Compare model performances
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_mean']
performance_df = pd.DataFrame({name: [results[name][metric] for metric in metrics] 
                             for name in results.keys()}, index=metrics)

# Display the performance comparison
print("\n--- Model Performance Comparison ---")
print(performance_df)

# Plot performance comparison
plt.figure(figsize=(12, 8))
performance_df.T.plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Find the best model based on ROC AUC
best_model_name = performance_df.loc['roc_auc'].idxmax()
best_model = results[best_model_name]['model']
best_auc = results[best_model_name]['roc_auc']

print(f"\n--- Best Model: {best_model_name} with ROC AUC of {best_auc:.4f} ---")

# Feature importance analysis for the best model
print("\n--- Feature Importance Analysis ---")

feature_names = X.columns
if best_model_name == 'Logistic Regression':
    importances = best_model.coef_[0]
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, importances)
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.tight_layout()
    plt.savefig('feature_importance_logistic.png')
    plt.close()
    
elif best_model_name in ['Decision Tree', 'Random Forest']:
    importances = best_model.feature_importances_
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, importances)
    plt.title(f'Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{best_model_name.lower().replace(" ", "_")}.png')
    plt.close()

# Save the best model for future use
import pickle
with open('best_lung_cancer_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'le': le
    }, f)

print("\n--- Model saved as 'best_lung_cancer_model.pkl' ---")

# Create a function for making predictions with the best model
def predict_lung_cancer_risk(data_dict):
    """
    Make prediction using the best model
    
    Parameters:
    data_dict: Dictionary containing patient information
    
    Returns:
    prediction: Binary prediction (0: No Cancer, 1: Cancer)
    probability: Probability of having lung cancer
    """
    # Load the model
    with open('best_lung_cancer_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Process the input data - convert categorical variables
    processed_dict = data_dict.copy()
    
    # Convert gender if it's a string
    if 'GENDER' in processed_dict and isinstance(processed_dict['GENDER'], str):
        gender_mapping = {'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0, 'Male': 1, 'Female': 0, 'm': 1, 'f': 0}
        processed_dict['GENDER'] = gender_mapping.get(processed_dict['GENDER'], processed_dict['GENDER'])
    
    # Handle spaces in column names
    rename_dict = {}
    for key in list(processed_dict.keys()):
        new_key = key.strip().replace(" ", "_")
        if new_key != key:
            rename_dict[key] = new_key
    
    # Rename keys with spaces
    for old_key, new_key in rename_dict.items():
        processed_dict[new_key] = processed_dict.pop(old_key)
    
    # Create a DataFrame for the input data
    input_df = pd.DataFrame([processed_dict])
    
    # Ensure the input has all required features
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Keep only the features used during training
    input_df = input_df[feature_names]
    
    # Print the final input data for debugging
    print("Processed input data:")
    print(input_df)
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

# Example of how to use the prediction function
example_patient = {
    'GENDER': 'M',  # M or F
    'AGE': 60,
    'SMOKING': 2,   # 1: No, 2: Yes
    'YELLOW_FINGERS': 2,
    'ANXIETY': 1,
    'PEER_PRESSURE': 1,
    'CHRONIC DISEASE': 2,  # Note the space in the column name
    'FATIGUE': 2,
    'ALLERGY': 1,
    'WHEEZING': 2,
    'ALCOHOL CONSUMING': 1,  # Note the space in the column name
    'COUGHING': 2,
    'SHORTNESS OF BREATH': 2,  # Note the space in the column name
    'SWALLOWING DIFFICULTY': 1,  # Note the space in the column name
    'CHEST PAIN': 2  # Note the space in the column name
}

print("\n--- Example Prediction ---")
print("Patient data:", example_patient)

try:
    prediction, probability = predict_lung_cancer_risk(example_patient)
    print(f"Prediction: {'Lung Cancer' if prediction == 1 else 'No Lung Cancer'}")
    print(f"Probability of Lung Cancer: {probability:.2%}")
except Exception as e:
    print(f"Error in prediction: {e}")

print("\n--- Lung Cancer Risk Predictor Project Completed ---")