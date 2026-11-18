"""Batch prediction using a trusted local model artifact."""
import argparse
from pathlib import Path
import warnings
import joblib
import pandas as pd
from car_price import BRANDS, normalize_name

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/car_price_model.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/new_predictions.csv")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if "Car_Name" not in data:
        raise ValueError("Missing required column: Car_Name")
    unknown = ~normalize_name(data.Car_Name).isin(BRANDS)
    if unknown.any():
        raise ValueError("Input contains unsupported car models or non-car vehicles; see BRANDS in car_price.py")
    model = joblib.load(args.model)  # Never load untrusted pickle/joblib files.
    data["Predicted_Selling_Price"] = model.predict(data)
    if (data.Predicted_Selling_Price < 0).any():
        warnings.warn("Some predictions are negative; inspect these unsupported estimates.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(data.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
