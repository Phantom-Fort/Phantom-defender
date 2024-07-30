import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load datasets using raw string literals
datasets = {
    "unsw_nb15": r'C:\Users\Admin\Downloads\datasets\archive\NF-UNSW-NB15-v2.csv',
    "cicids2017": r'C:\Users\Admin\Downloads\datasets\archive\combine.csv'
}

# Function to load a dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    return df

# Load and preprocess each dataset
data_frames = {name: load_dataset(path) for name, path in datasets.items()}

# Print column names for debugging
for name, df in data_frames.items():
    print(f"Columns in {name}: {df.columns.tolist()}")

# Define columns for preprocessing
def get_features(df):
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    return numerical_features, categorical_features

# Get features from the first dataset for fitting
numerical_features, categorical_features = get_features(data_frames["unsw_nb15"])

# Print features for debugging
print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Initialize transformers
scaler = StandardScaler()
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Correct initialization

# Fit OneHotEncoder on categorical features of the first dataset
encoder.fit(data_frames["unsw_nb15"][categorical_features])

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
            ('scaler', scaler)
        ]), numerical_features),
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'  # Keeps other columns if present
)

# Preprocess datasets
def preprocess_dataset(df, preprocessor):
    # Print columns before processing
    print(f"Columns before preprocessing: {df.columns.tolist()}")
    
    # Apply transformations
    df_processed = preprocessor.fit_transform(df)
    
    # Get feature names from the fitted OneHotEncoder
    feature_names = (
        numerical_features +
        encoder.get_feature_names_out(categorical_features).tolist()
    )
    
    # Print feature names for debugging
    print(f"Feature names after preprocessing: {feature_names}")
    
    # Create DataFrame with new feature names
    return pd.DataFrame(df_processed, columns=feature_names)

# Process all datasets
processed_datasets = {}
for name, df in data_frames.items():
    try:
        processed_datasets[name] = preprocess_dataset(df, preprocessor)
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Ensure consistent columns in all datasets
common_columns = set.intersection(*(set(df.columns) for df in processed_datasets.values()))
processed_datasets = {name: df[common_columns] for name, df in processed_datasets.items()}

# Concatenate datasets
combined_dataset = pd.concat(processed_datasets.values(), ignore_index=True)

# Save the preprocessed dataset
combined_dataset.to_csv(r'C:\Users\Admin\Downloads\datasets\preprocessed_combined_dataset.csv', index=False)

print("Preprocessing complete and dataset saved.")
