"""Shared, serializable preprocessing for the supplied historical car dataset."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BRANDS = {}
for brand, names in {
    "Maruti Suzuki": "ritz,sx4,ciaz,wagon r,swift,vitara brezza,s cross,alto 800,ertiga,dzire,alto k10,ignis,800,baleno,omni",
    "Toyota": "fortuner,innova,corolla altis,etios cross,etios g,etios liva,corolla,etios gd,camry,land cruiser",
    "Hyundai": "i20,grand i10,i10,eon,xcent,elantra,creta,verna",
    "Honda": "city,brio,amaze,jazz",
}.items():
    BRANDS.update({name: brand for name in names.split(",")})

RAW_FEATURES = ["Car_Name", "Year", "Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"]
NUMERIC = ["Present_Price", "Driven_kms", "Owner", "Car_Age", "Km_Per_Year", "Log_Km"]
CATEGORICAL = ["Car_Name", "Brand", "Fuel_Type", "Selling_type", "Transmission"]


def normalize_name(series):
    return series.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def validate_features(frame, reference_year=2018):
    missing = set(RAW_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame[RAW_FEATURES].copy()
    for col in ["Year", "Present_Price", "Driven_kms", "Owner"]:
        original = result[col]
        values = pd.to_numeric(original, errors="coerce")
        if (original.notna() & values.isna()).any():
            raise ValueError(f"{col} contains non-numeric values")
        if np.isinf(values).any() or (values.dropna() < 0).any():
            raise ValueError(f"{col} must contain finite nonnegative values")
        if col in ["Year", "Owner"] and (values.dropna() % 1 != 0).any():
            raise ValueError(f"{col} must contain whole numbers")
        result[col] = values
    if ((result.Year < 1886) | (result.Year > reference_year)).any():
        raise ValueError(f"Year must be between 1886 and {reference_year}; this is a historical model")
    if (result.Present_Price <= 0).any():
        raise ValueError("Present_Price must be positive")
    for col in ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]:
        result[col] = normalize_name(result[col]).replace("", pd.NA).astype(object)
        result[col] = result[col].where(pd.notna(result[col]), np.nan)
    return result


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Deterministic features; never accesses the target or learns from holdout data."""
    def __init__(self, reference_year=2018):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        validate_features(X, self.reference_year)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        frame = validate_features(X, self.reference_year)
        frame["Brand"] = frame.Car_Name.map(BRANDS).fillna("unknown")
        frame["Car_Age"] = self.reference_year - frame.Year
        frame["Km_Per_Year"] = frame.Driven_kms / frame.Car_Age.clip(lower=1)
        frame["Log_Km"] = np.log1p(frame.Driven_kms)
        return frame[NUMERIC + CATEGORICAL]


def build_pipeline(regressor):
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    return Pipeline([("features", FeatureEngineer()), ("preprocess", preprocessing), ("model", regressor)])
