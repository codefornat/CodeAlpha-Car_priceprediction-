"""Run with python train.py; all paths default to the project directory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from car_price import BRANDS, RAW_FEATURES, build_pipeline, normalize_name, validate_features

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "archive (3).zip")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data)
    if "Selling_Price" not in raw:
        raise ValueError("Dataset must include Selling_Price")
    cleaned = validate_features(raw)
    cleaned["Selling_Price"] = pd.to_numeric(raw.Selling_Price, errors="raise")
    if not np.isfinite(cleaned.Selling_Price).all() or (cleaned.Selling_Price <= 0).any():
        raise ValueError("Selling_Price must be finite and positive")
    duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()
    excluded = cleaned.loc[~cleaned.Car_Name.isin(BRANDS)]
    excluded.to_csv(out / "excluded_non_car_or_unmapped.csv", index_label="source_row")
    cars = cleaned.loc[cleaned.Car_Name.isin(BRANDS)].copy()
    cars.to_csv(out / "cars_cleaned.csv", index_label="source_row")
    X, y = cars[RAW_FEATURES], cars.Selling_Price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = [
        ("Median baseline", DummyRegressor(strategy="median"), {}),
        ("Ridge", Ridge(), {"model__alpha": [0.1, 1, 10, 100]}),
        ("Random forest", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=1),
         {"model__max_depth": [None, 6], "model__min_samples_leaf": [1, 2]}),
        ("Extra trees", ExtraTreesRegressor(n_estimators=250, random_state=42, n_jobs=1),
         {"model__min_samples_leaf": [1, 2, 4]}),
        ("Gradient boosting", GradientBoostingRegressor(random_state=42, loss="huber"),
         {"model__n_estimators": [100, 200], "model__max_depth": [2, 3], "model__learning_rate": [0.05]}),
    ]
    rows, searches = [], {}
    for name, estimator, grid in candidates:
        print(f"Cross-validating {name}...", flush=True)
        search = GridSearchCV(build_pipeline(estimator), grid, cv=cv,
                              scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
                              refit="mae", n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        i = search.best_index_
        rows.append({"model": name, "cv_mae": -search.best_score_,
                     "cv_mae_std": search.cv_results_["std_test_mae"][i],
                     "cv_rmse": -search.cv_results_["mean_test_rmse"][i],
                     "cv_r2": search.cv_results_["mean_test_r2"][i],
                     "parameters": json.dumps(search.best_params_)})
        searches[name] = search
    scores = pd.DataFrame(rows).sort_values("cv_mae")
    scores.to_csv(out / "cross_validation.csv", index=False)
    winner = scores.iloc[0]["model"]
    model = searches[winner].best_estimator_
    prediction = model.predict(X_test)
    baseline_prediction = searches["Median baseline"].predict(X_test)

    def metrics(truth, predicted):
        return {"mae": float(mean_absolute_error(truth, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
                "r2": float(r2_score(truth, predicted))}

    report = {"selected_model": winner, "best_parameters": searches[winner].best_params_,
              "holdout": metrics(y_test, prediction), "baseline_holdout": metrics(y_test, baseline_prediction),
              "training": metrics(y_train, model.predict(X_train)),
              "raw_rows": len(raw), "duplicates_removed": duplicates, "excluded_rows": len(excluded),
              "car_rows": len(cars), "training_rows": len(X_train), "test_rows": len(X_test),
              "raw_missing_values": raw.isna().sum().to_dict(),
              "reference_year": 2018, "random_seed": 42,
              "price_unit": "dataset units; commonly INR lakh for this dataset, not verified in supplied archive",
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    results = X_test.copy()
    results["actual"] = y_test
    results["predicted"] = prediction
    results["residual"] = y_test - prediction
    results.to_csv(out / "holdout_predictions.csv", index_label="source_row")
    pd.DataFrame({"source_row": cars.index, "split": np.where(cars.index.isin(X_train.index), "train", "test")}).to_csv(out / "split.csv", index=False)
    # The saved model is exactly the evaluated training-only model, not a later refit.
    joblib.dump(model, out / "car_price_model.joblib")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    y_train.hist(ax=axes[0, 0], bins=20, color="#287d8e")
    axes[0, 0].set(title="Training car selling prices", xlabel="Selling price (dataset units)", ylabel="Count")
    axes[0, 1].scatter(X_train.Present_Price, y_train, alpha=0.7)
    axes[0, 1].set(xlabel="Present price", ylabel="Selling price", title="Training price relationship")
    axes[1, 0].scatter(2018 - X_train.Year, y_train, alpha=0.7)
    axes[1, 0].set(xlabel="Age at assumed 2018 snapshot", ylabel="Selling price", title="Age and price (training data)")
    axes[1, 1].scatter(X_train.Driven_kms, y_train, alpha=0.7)
    axes[1, 1].set(xlabel="Odometer (km)", ylabel="Selling price", title="Usage and price (training data)")
    fig.tight_layout(); fig.savefig(out / "exploration.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].barh(scores.model, scores.cv_mae, xerr=scores.cv_mae_std, color="#287d8e")
    axes[0].set(title="5-fold model comparison", xlabel="Validation MAE ± fold SD")
    axes[1].scatter(y_test, prediction, alpha=0.8)
    low, high = min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())
    axes[1].plot([low, high], [low, high], "k--")
    axes[1].set(title=f"Holdout: {winner}", xlabel="Actual price", ylabel="Predicted price")
    axes[2].scatter(prediction, y_test - prediction, alpha=0.8)
    axes[2].axhline(0, color="black", linestyle="--")
    axes[2].set(title="Holdout residuals", xlabel="Predicted price", ylabel="Actual − predicted")
    fig.tight_layout(); fig.savefig(out / "evaluation.png", dpi=160); plt.close(fig)

    importance = permutation_importance(model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=20, random_state=42)
    imp = pd.DataFrame({"feature": RAW_FEATURES, "mae_increase": importance.importances_mean, "std": importance.importances_std}).sort_values("mae_increase")
    imp.to_csv(out / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.feature, imp.mae_increase, xerr=imp["std"], color="#287d8e")
    ax.set(title="Holdout permutation importance", xlabel="Increase in MAE when feature is shuffled")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=160); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
