"""
ניקוי והכנת נתונים למודל הרגרסיה.

שלבי הניקוי (לפי הבקשה):
    1. השלמת ערכים חסרים  - חציון למשתנים כמותיים, שכיח למשתנים קטגוריאליים
    2. הסרת ערכים קיצוניים - שיטת IQR (טווח בין-רבעוני) על המשתנים הכמותיים
    3. קידוד (encoding)     - One-Hot למשתנים קטגוריאליים נומינליים
    4. נרמול (normalization) - תקנון (StandardScaler) למשתנים הכמותיים

הטיפול בכל משתנה נגזר מסוגו:
    NUMERIC_FEATURES     - כמותיים  (כולל fitness_level שהוא אורדינלי 1-10)
    CATEGORICAL_FEATURES - קטגוריאליים נומינליים
    TARGET               - משתנה המטרה לחיזוי (אחוז הצלחה)
    LEAKAGE / ID         - נזרקים: מזהים, שם, ומשתנים שמהם נגזר ה-target
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# age_squared הוא משתנה מהונדס (feature engineering): השפעת הגיל על ההצלחה
# אינה לינארית (שיא באמצע), ולכן מוסיפים איבר ריבועי. המודל נשאר "לינארי
# בפרמטרים" - כלומר עדיין רגרסיה לינארית.
RAW_NUMERIC = ["age", "fitness_level", "years_playing", "weekly_training_hours", "height_cm"]
NUMERIC_FEATURES = RAW_NUMERIC + ["age_squared"]
CATEGORICAL_FEATURES = ["gender", "dominant_hand", "backhand_type"]
TARGET = "success_rate"
DROP_COLS = ["club", "player_id", "full_name", "matches_played", "matches_won"]  # מזהים + דליפת מטרה


@dataclass
class CleaningReport:
    n_input: int = 0
    n_after_outliers: int = 0
    missing_before: dict = field(default_factory=dict)
    imputation_values: dict = field(default_factory=dict)
    outliers_removed: dict = field(default_factory=dict)
    outlier_bounds: dict = field(default_factory=dict)
    encoded_columns: list = field(default_factory=list)
    feature_columns: list = field(default_factory=list)

    @property
    def total_outliers_removed(self) -> int:
        return self.n_after_outliers and (self.n_input - self.n_after_outliers)


class TennisPreprocessor:
    """
    מבצע fit על נתוני האימון (לומד חציונים/שכיחים/גבולות/סקיילר)
    ואז transform על כל דאטה חדשה - כך שהחיזוי משתמש באותם פרמטרים בדיוק.
    """

    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = iqr_factor
        self.medians_: dict[str, float] = {}
        self.modes_: dict[str, Any] = {}
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.scaler_: StandardScaler | None = None
        self.feature_columns_: list[str] = []
        self.report_ = CleaningReport()

    # ---------- שלבים בודדים ----------
    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["age_squared"] = df["age"] ** 2
        return df

    def _impute(self, df: pd.DataFrame, learn: bool) -> pd.DataFrame:
        df = df.copy()
        for col in RAW_NUMERIC:
            if col not in df:
                continue
            if learn:
                self.medians_[col] = float(df[col].median())
            df[col] = df[col].fillna(self.medians_[col])
        for col in CATEGORICAL_FEATURES:
            if col not in df:
                continue
            if learn:
                self.modes_[col] = df[col].mode(dropna=True).iloc[0]
            df[col] = df[col].fillna(self.modes_[col])
        return df

    def _fit_outlier_bounds(self, df: pd.DataFrame) -> None:
        for col in RAW_NUMERIC:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr)

    def _drop_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        removed = {}
        for col, (lo, hi) in self.bounds_.items():
            col_mask = df[col].between(lo, hi)
            removed[col] = int((~col_mask).sum())
            mask &= col_mask
        self.report_.outliers_removed = removed
        self.report_.outlier_bounds = {k: (round(v[0], 2), round(v[1], 2)) for k, v in self.bounds_.items()}
        return df[mask].copy()

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=float)

    # ---------- API ראשי ----------
    def fit_transform(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        self.report_ = CleaningReport(n_input=len(df_raw))
        self.report_.missing_before = df_raw.isna().sum().to_dict()

        df = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns])
        df = self._impute(df, learn=True)
        df = self._engineer(df)
        self.report_.imputation_values = {**self.medians_, **self.modes_}

        self._fit_outlier_bounds(df)
        df = self._drop_outliers(df)
        self.report_.n_after_outliers = len(df)

        y = df[TARGET].astype(float)
        df = df.drop(columns=[TARGET])
        df = self._encode(df)

        self.feature_columns_ = list(df.columns)
        self.report_.feature_columns = self.feature_columns_
        self.report_.encoded_columns = [c for c in df.columns if c not in NUMERIC_FEATURES]

        self.scaler_ = StandardScaler()
        df[NUMERIC_FEATURES] = self.scaler_.fit_transform(df[NUMERIC_FEATURES])
        return df, y

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """הכנה של נתונים חדשים לחיזוי (ללא הסרת outliers, ללא target)."""
        df = df_raw.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df_raw.columns])
        df = self._impute(df, learn=False)
        df = self._engineer(df)
        df = self._encode(df)
        for col in self.feature_columns_:
            if col not in df:
                df[col] = 0.0
        df = df[self.feature_columns_]
        df[NUMERIC_FEATURES] = self.scaler_.transform(df[NUMERIC_FEATURES])
        return df


def clean_for_display(df_raw: pd.DataFrame, pre: TennisPreprocessor) -> pd.DataFrame:
    """גרסה קריאה-לאדם של הנתונים אחרי השלמה והסרת קיצונים (לפני קידוד/נרמול)."""
    df = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns])
    df = pre._impute(df, learn=False)
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in pre.bounds_.items():
        mask &= df[col].between(lo, hi)
    return df[mask].copy()
