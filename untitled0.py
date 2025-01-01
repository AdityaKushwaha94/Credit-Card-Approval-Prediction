
##main code here:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
application_data = pd.read_csv('application_record.csv')
credit_data = pd.read_csv('credit_record.csv')

# Merge datasets
merged_data = application_data.merge(credit_data, on='ID')

# Define a mapping for STATUS values to numeric scores
status_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'C': 0, 'X': 5}

def preprocess_status(status_sequence):
    """Convert STATUS sequence to numeric-friendly format."""
    numeric_sequence = []
    for char in status_sequence:
        if char in status_map:
            numeric_sequence.append(status_map[char])
        else:
            numeric_sequence.append(5)  # Default to high risk for unexpected values
    return sum(numeric_sequence)  # Aggregate the risk score

# Apply preprocessing to the STATUS column
credit_data['Processed_STATUS'] = credit_data['STATUS'].apply(
    lambda x: ''.join(char if char in status_map else 'X' for char in x)
)
credit_summary = credit_data.groupby('ID')['Processed_STATUS'].apply(preprocess_status).reset_index(name='Risk_Score')

# Merge risk score back into the main dataset
merged_data = merged_data.merge(credit_summary, on='ID', how='left')

# Preprocess application data
merged_data['Age'] = (-merged_data['DAYS_BIRTH']) // 365  # Convert days to years
merged_data['Years_Employed'] = (-merged_data['DAYS_EMPLOYED']) // 365  # Convert days to years

# Encode categorical variables
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

for col in categorical_cols:
    merged_data[col] = merged_data[col].fillna('Unknown')

# One-hot encoding for categorical data
merged_data = pd.get_dummies(merged_data, columns=categorical_cols, drop_first=True)

# Handle missing values
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())

non_numeric_cols = merged_data.select_dtypes(exclude=[np.number]).columns
merged_data[non_numeric_cols] = merged_data[non_numeric_cols].fillna('Unknown')

# Select features and target for the model
features = [
    'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'Age',
    'Years_Employed', 'Risk_Score'
] + [col for col in merged_data.columns if col.startswith(tuple(categorical_cols))]

merged_data['Approval_Status'] = np.where(merged_data['Risk_Score'] < 10, 1, 0)

X = merged_data[features]
y = merged_data['Approval_Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100
}

# Use xgboost.cv for hyperparameter tuning (alternatively to RandomizedSearchCV)
dtrain = xgb.DMatrix(X_train, label=y_train)

# Perform cross-validation using xgboost
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    nfold=3,
    early_stopping_rounds=10,
    metrics='logloss',
    as_pandas=True
)

best_iteration = cv_results['test-logloss-mean'].idxmin()  # Get the best iteration based on logloss
print(f'Best iteration: {best_iteration}')

# Train the final model using the best iteration
final_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=best_iteration)

# Predict using the trained model
dtest = xgb.DMatrix(X_test)
y_pred = final_model.predict(dtest)

# Convert prediction probabilities to binary class labels
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred_binary))


    ###
    ###
    ### for sample manual input testing .
    ###
    ###

sample_input = {
    'AMT_INCOME_TOTAL': [300000],
    'CNT_CHILDREN': [2],
    'CNT_FAM_MEMBERS': [4],
    'Age': [30],
    'Years_Employed': [5],
    'Risk_Score': [8],
    # Add one-hot encoded values for categorical columns, initializing all to 0
    'CODE_GENDER_M': [1],
    'FLAG_OWN_CAR_Y': [1],
    'FLAG_OWN_REALTY_Y': [1],
    'NAME_INCOME_TYPE_Working': [1],
    'NAME_EDUCATION_TYPE_Secondary': [1],
    'NAME_FAMILY_STATUS_Married': [1],
    'NAME_HOUSING_TYPE_Employee': [1],
    'OCCUPATION_TYPE_Security Staff': [1]
    # ... (add other one-hot encoded features with value 0)
}

# Convert the sample input into a DataFrame
sample_df = pd.DataFrame(sample_input)

# Get missing columns and add them to sample_df
missing_cols = set(features) - set(sample_df.columns)
for col in missing_cols:
    sample_df[col] = 0  # Fill missing columns with 0

# Now you can safely select the features
sample_df = sample_df[features]

# Convert to DMatrix format for prediction
dtest_sample = xgb.DMatrix(sample_df)

# Predict using the trained model
sample_prediction = final_model.predict(dtest_sample)

# Convert prediction probabilities to binary class labels (0 or 1)
sample_prediction_binary = (sample_prediction > 0.5).astype(int)

# Output the prediction
print(f"Predicted Approval Status: {sample_prediction_binary[0]}")




    ###
    ###
    ### for user input testing .
    ###
    ###

# Function to get user input for the required features
def get_user_input():
    print("Please provide the following details:")

    # Input for numerical features
    AMT_INCOME_TOTAL = float(input("Enter total income (AMT_INCOME_TOTAL): "))
    CNT_CHILDREN = int(input("Enter number of children (CNT_CHILDREN): "))
    CNT_FAM_MEMBERS = int(input("Enter number of family members (CNT_FAM_MEMBERS): "))
    Age = int(input("Enter age (Age): "))
    Years_Employed = int(input("Enter years of employment (Years_Employed): "))
    Risk_Score = int(input("Enter risk score (Risk_Score): "))

    # Input for categorical features (use 1 for presence and 0 for absence)
    CODE_GENDER_M = int(input("Is gender Male? (1 for Male, 0 for Female): "))
    FLAG_OWN_CAR_Y = int(input("Do you own a car? (1 for Yes, 0 for No): "))
    FLAG_OWN_REALTY_Y = int(input("Do you own realty? (1 for Yes, 0 for No): "))
    NAME_INCOME_TYPE_Working = int(input("Is income type 'Working'? (1 for Yes, 0 for No): "))
    NAME_EDUCATION_TYPE_Secondary = int(input("Is education type 'Secondary'? (1 for Yes, 0 for No): "))
    NAME_FAMILY_STATUS_Married = int(input("Is family status 'Married'? (1 for Yes, 0 for No): "))
    NAME_HOUSING_TYPE_Employee = int(input("Is housing type 'Employee'? (1 for Yes, 0 for No): "))
    OCCUPATION_TYPE_Security_Staff = int(input("Is occupation type 'Security Staff'? (1 for Yes, 0 for No): "))

    # Return input as a dictionary
    return {
        'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
        'CNT_CHILDREN': CNT_CHILDREN,
        'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS,
        'Age': Age,
        'Years_Employed': Years_Employed,
        'Risk_Score': Risk_Score,
        'CODE_GENDER_M': CODE_GENDER_M,
        'FLAG_OWN_CAR_Y': FLAG_OWN_CAR_Y,
        'FLAG_OWN_REALTY_Y': FLAG_OWN_REALTY_Y,
        'NAME_INCOME_TYPE_Working': NAME_INCOME_TYPE_Working,
        'NAME_EDUCATION_TYPE_Secondary': NAME_EDUCATION_TYPE_Secondary,
        'NAME_FAMILY_STATUS_Married': NAME_FAMILY_STATUS_Married,
        'NAME_HOUSING_TYPE_Employee': NAME_HOUSING_TYPE_Employee,
        'OCCUPATION_TYPE_Security_Staff': OCCUPATION_TYPE_Security_Staff
    }




# Function to predict approval status based on user input
def predict_approval_status(model, features):
    # Get user input
    user_input = get_user_input()

    # Convert user input into a DataFrame
    user_df = pd.DataFrame([user_input])

    # Ensure the input matches the order of the model's features
    user_df = user_df[features]

    # Convert to DMatrix format for prediction
    dtest_sample = xgb.DMatrix(user_df)

    # Predict using the trained model
    prediction = model.predict(dtest_sample)

    # Convert prediction probabilities to binary class labels (0 or 1)
    approval_status = (prediction > 0.5).astype(int)

    # Output the prediction
    if approval_status[0] == 1:
        print("Approval Status: Approved")
    else:
        print("Approval Status: Not Approved")

# Assuming final_model is the trained XGBoost model and features is the list of model features
# Call the function to predict approval status
predict_approval_status(final_model, features)

