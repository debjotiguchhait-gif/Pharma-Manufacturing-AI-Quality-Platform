import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------
# Load data
# --------------------------------------------

project_root = Path(__file__).resolve().parent.parent

data_file = (
    project_root
    / "data"
    / "pharma_quality_data.csv"
)

df = pd.read_csv(data_file)


# --------------------------------------------
# Define target and features
# --------------------------------------------

TARGET = "Quality_Score"

# Batch_ID is only an identifier
X = df.drop(
    columns=[TARGET, "Batch_ID"]
)

y = df[TARGET]


# --------------------------------------------
# Identify feature types
# --------------------------------------------

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


print("\nNUMERIC FEATURES")
print(numeric_features)

print("\nCATEGORICAL FEATURES")
print(categorical_features)


# --------------------------------------------
# Train-test split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN / TEST SPLIT")
print("-----------------------------")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))


# --------------------------------------------
# Numerical preprocessing
# --------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# --------------------------------------------
# Categorical preprocessing
# --------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# --------------------------------------------
# Combined preprocessing pipeline
# --------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# --------------------------------------------
# Test preprocessing
# FIT ONLY ON TRAINING DATA
# --------------------------------------------

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print("\nPREPROCESSING COMPLETE")
print("-----------------------------")

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed test shape:",
    X_test_processed.shape
)


# --------------------------------------------
# Verify missing values
# --------------------------------------------

print(
    "\nMissing values after preprocessing:",
    pd.isna(X_train_processed).sum()
)