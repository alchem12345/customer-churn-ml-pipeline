# this is where the pipeline for the preprocessing will be done like the one hot encoding or the standard scaler that will be used 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessor(numeric_features , categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num" , numeric_transformer, numeric_features),
            ("cat" , OneHotEncoder(handle_unknown='ignore') , categorical_features)
        ]
    )

# noiceee
