"""
מודל רגרסיה לינארית לחיזוי אחוז ההצלחה (success_rate).

מטריקות שמחושבות על קבוצת המבחן:
    R2                - מקדם הקביעה
    Adjusted R2       - R2 מתוקנן למספר המשתנים המסבירים
    RMSE              - שורש שגיאת הריבועים הממוצעת
    MAE               - שגיאה מוחלטת ממוצעת
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocessing import TennisPreprocessor


@dataclass
class ModelResult:
    model: LinearRegression
    preprocessor: TennisPreprocessor
    metrics: dict
    coefficients: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    n_test: int
    n_features: int


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """n = מספר תצפיות, p = מספר משתנים מסבירים."""
    if n - p - 1 <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def train_regression(df_raw: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> ModelResult:
    pre = TennisPreprocessor()
    X, y = pre.fit_transform(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    n, p = len(y_test), X.shape[1]
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "R2": round(r2, 4),
        "Adjusted R2": round(adjusted_r2(r2, n, p), 4),
        "RMSE": round(rmse, 3),
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 3),
        "n_train": len(y_train),
        "n_test": n,
        "n_features": p,
    }

    coefs = (
        pd.DataFrame({"feature": pre.feature_columns_, "coefficient": model.coef_})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
    coefs.loc[len(coefs)] = ["intercept", model.intercept_]

    return ModelResult(model, pre, metrics, coefs, y_test, y_pred, n, p)


def predict_success_rate(result: ModelResult, new_data: pd.DataFrame) -> np.ndarray:
    X_new = result.preprocessor.transform(new_data)
    return np.clip(result.model.predict(X_new), 0, 100)
