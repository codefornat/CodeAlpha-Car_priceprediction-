# Car price prediction with machine learning

A complete Python regression project for Task 3 using the supplied [dataset archive](archive%20%283%29.zip). Start with the executed [analysis notebook](Car_Price_Prediction.ipynb) or the [results report](REPORT.md).

## Run

Requires Python 3.11 or newer. From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py --input examples/cars.csv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If Windows' `python` alias does not work, use an installed interpreter. The environment used for this submission is:

```powershell
& 'C:/Users/Gallagher/AppData/Local/Python/pythoncore-3.14-64/python.exe' train.py
```

The training command accepts `--data path/to/data.csv` (or a ZIP containing one CSV) and `--output path/to/results`. Input must follow the supplied schema. The notebook uses existing outputs; set its `RUN_TRAINING` variable to `True` to regenerate them. To open it in Jupyter, optionally install `jupyterlab` into the environment and run `python -m jupyterlab`.

## Dataset and scope

The supplied CSV has 301 rows and nine columns, with no missing values. It includes two-wheelers. After normalizing text, removing two exact duplicates, and excluding 101 motorcycle/scooter records through an explicit car-model allowlist, 198 cars remain. Excluded records are saved for audit. Unknown models in alternative datasets require reviewing and extending the allowlist; they are not automatically assumed to be cars.

The archive does not include a source URL, license, collection date, or currency metadata. This resembles the commonly circulated CarDekho car dataset, whose prices are usually described in INR lakh, but this provenance and unit are **not verified**. Results therefore use **dataset price units**. Verify the original download page before redistribution or monetary interpretation.

| Column | Meaning and treatment |
| --- | --- |
| `Car_Name` | Model name; normalized, one-hot encoded, mapped to brand |
| `Year` | Manufacturing year; converted to age relative to assumed 2018 snapshot |
| `Selling_Price` | Regression target, never an input |
| `Present_Price` | Dataset's present/new-price reference; numeric predictor |
| `Driven_kms` | Odometer distance, not fuel efficiency |
| `Fuel_Type` | Categorical petrol, diesel, CNG |
| `Selling_type` | Dealer or individual |
| `Transmission` | Manual or automatic |
| `Owner` | Numeric ownership indicator/count as supplied; exact semantics unverified |

Horsepower, fuel economy, service history, accident condition, location, and measured brand goodwill are absent. Brand/model categories are a limited proxy for brand-related pricing differences; no invented horsepower or goodwill scores are added. The fixed 2018 reference year is an explicit assumption based on the newest manufacturing year, not a verified sale date. Per-year mileage is approximate, with age zero divided by one.

## Method

1. Validate schema and numerical ranges; normalize text and remove duplicates before splitting.
2. Reserve 20% (40 cars) as a holdout with seed 42; train on 158 cars.
3. Engineer brand, age, kilometres per year, and log odometer distance.
4. Fit median numeric imputation, most-frequent categorical imputation, numeric scaling, and one-hot encoding **inside each training fold**. Unseen categories are encoded safely.
5. Compare a median baseline, Ridge, random forest, extra trees, and gradient boosting using five-fold shuffled cross-validation. Tune a small hyperparameter grid and select by lowest mean MAE.
6. Evaluate the selected model and baseline on the holdout once, reporting MAE, RMSE and R². Save the exact training-only model used for evaluation.
7. Plot training-data exploration, validation comparison, holdout predictions/residuals, and holdout permutation importance. Importance is descriptive and is not used for retuning.

MAE is average absolute error in price units. RMSE penalizes large errors more strongly. R² compares squared prediction error with predicting the holdout mean and can be negative. CV standard deviations describe variability across folds, not confidence intervals. Selecting parameters using CV makes the winning CV score optimistic; the holdout is the final independent check.

## Files and generated results

- [car_price.py](car_price.py): reusable feature engineering and preprocessing pipeline.
- [train.py](train.py): data audit, comparison, tuning, evaluation and plots.
- [predict.py](predict.py): batch predictions from a saved model.
- [examples/cars.csv](examples/cars.csv): example historical inputs.
- [tests/test_workflow.py](tests/test_workflow.py): feature correctness, missing/unseen values, invalid-input checks.
- [outputs/metrics.json](outputs/metrics.json): metrics, row counts, source SHA-256 and package versions.
- [outputs/cross_validation.csv](outputs/cross_validation.csv): model scores and selected parameters.
- [outputs/holdout_predictions.csv](outputs/holdout_predictions.csv): predictions and residuals with original zero-based CSV row IDs.
- [outputs/split.csv](outputs/split.csv): reproducible train/test membership.
- [outputs/cars_cleaned.csv](outputs/cars_cleaned.csv) and [excluded records](outputs/excluded_non_car_or_unmapped.csv): data audit.
- [outputs/car_price_model.joblib](outputs/car_price_model.joblib): preprocessing plus fitted estimator; load only trusted artifacts and use compatible package versions.

## Real-world applications and limitations

Potential uses include dealer trade-in estimates, marketplace listing guidance, inventory pricing, and screening unusually priced listings. This small historical educational model is **not a current market valuation service**. Prediction inputs must refer to the same historical price context; years after 2018 are rejected. The CLI supports only the known car-model allowlist and cannot value motorcycles.

The random split measures interpolation within this dataset, not performance on future years, new regions or unseen vehicle families. Duplicate removal cannot rule out repeated listings for the same vehicle because there is no vehicle ID. Price-reference availability and timing must be checked before deploying: if `Present_Price` is unavailable at prediction time or was recorded after sale, the evaluation would not reflect deployable performance. No causal brand-goodwill claim can be drawn from feature importance.

For production, collect licensed recent listings with transaction timestamps, region, condition, horsepower and verified sale prices; split by time and vehicle identity; compare against regional baselines; validate subgroup errors and prediction intervals; and monitor drift. Keep human review for rare, luxury or out-of-range vehicles.
