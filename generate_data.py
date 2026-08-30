"""
יצירת נתונים סינתטיים עבור ה-CRM של מועדון הטניס.

כל שורה = שחקן/ית. הנתונים נשמרים ל-data/players_raw.csv.
בכוונה מוזרקים ערכים חסרים וערכים קיצוניים (outliers) כדי ששלב
ניקוי הנתונים בדאשבורד יהיה משמעותי.

סוגי המשתנים:
    כמותיים (numeric)  : age, fitness_level (סדר/אורדינלי 1-10),
                         years_playing, weekly_training_hours, height_cm
    קטגוריאליים (nominal): gender, dominant_hand, backhand_type
    מזהים / לא למודל   : player_id, full_name, matches_played, matches_won
    משתנה מטרה (target) : success_rate  (אחוז ניצחונות)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_PATH = os.path.join(DATA_DIR, "players_raw.csv")

FIRST_NAMES_M = ["Yotam", "Daniel", "Noam", "Ariel", "Omer", "Ido", "Guy", "Ron", "Amit", "Eitan"]
FIRST_NAMES_F = ["Noa", "Shira", "Maya", "Tamar", "Yael", "Lior", "Adi", "Roni", "Gal", "Hila"]
LAST_NAMES = ["Cohen", "Levi", "Mizrahi", "Peretz", "Biton", "Avraham", "Friedman", "Katz", "Shapira", "Azoulay"]


def _age_curve(age: np.ndarray) -> np.ndarray:
    """תרומת הגיל להצלחה - שיא סביב גיל 25, ירידה לכיוונים."""
    return 22.0 * np.exp(-((age - 25.0) / 11.0) ** 2)


def generate(n: int = 650, seed: int | None = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed) if seed is not None else RNG

    gender = rng.choice(["male", "female"], size=n, p=[0.55, 0.45])
    age = np.clip(rng.normal(28, 12, n), 8, 70).round().astype(int)
    dominant_hand = rng.choice(["right", "left"], size=n, p=[0.87, 0.13])
    backhand_type = rng.choice(["two_handed", "one_handed"], size=n, p=[0.7, 0.3])
    fitness_level = np.clip(rng.normal(6, 2, n), 1, 10).round().astype(int)
    years_playing = np.clip(rng.gamma(2.2, 2.4, n), 0, 40).round(1)
    weekly_training_hours = np.clip(rng.gamma(2.0, 2.2, n), 0, 25).round(1)

    height_base = np.where(gender == "male", 176, 165)
    height_cm = np.clip(rng.normal(height_base, 8, n), 130, 210).round(1)
    # ילדים נמוכים יותr
    height_cm = np.where(age < 16, (height_cm - (16 - age) * 3).clip(120, None), height_cm).round(1)

    is_left = (dominant_hand == "left").astype(float)
    is_male = (gender == "male").astype(float)

    # מודל חבוי לאחוז ההצלחה (מה שהרגרסיה אמורה לשחזר)
    latent = (
        28.0
        + fitness_level * 2.6
        + is_left * 3.5
        + _age_curve(age)
        + years_playing * 0.7
        + weekly_training_hours * 1.3
        + is_male * 1.5
        + (backhand_type == "two_handed").astype(float) * 1.0
        + rng.normal(0, 5.0, n)
    )
    success_rate = np.clip(latent, 2, 99).round(1)

    matches_played = rng.integers(10, 120, n)
    matches_won = np.round(matches_played * success_rate / 100).astype(int)

    df = pd.DataFrame(
        {
            "player_id": [f"P{1000 + i}" for i in range(n)],
            "full_name": [
                f"{rng.choice(FIRST_NAMES_M if g == 'male' else FIRST_NAMES_F)} {rng.choice(LAST_NAMES)}"
                for g in gender
            ],
            "gender": gender,
            "age": age,
            "dominant_hand": dominant_hand,
            "backhand_type": backhand_type,
            "fitness_level": fitness_level,
            "years_playing": years_playing,
            "weekly_training_hours": weekly_training_hours,
            "height_cm": height_cm,
            "matches_played": matches_played,
            "matches_won": matches_won,
            "success_rate": success_rate,
        }
    )

    df = _inject_missing(df, rng)
    df = _inject_outliers(df, rng)
    return df


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator, frac: float = 0.08) -> pd.DataFrame:
    df = df.copy()
    for col in ["fitness_level", "weekly_training_hours", "height_cm", "dominant_hand", "backhand_type"]:
        idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, rng: np.random.Generator, k: int = 12) -> pd.DataFrame:
    df = df.copy()
    idx = rng.choice(df.index, size=k, replace=False)
    half = idx[: k // 2]
    other = idx[k // 2 :]
    df.loc[half, "age"] = rng.integers(95, 130, len(half))
    df.loc[half, "height_cm"] = rng.uniform(215, 245, len(half)).round(1)
    df.loc[other, "weekly_training_hours"] = rng.uniform(40, 70, len(other)).round(1)
    df.loc[other, "years_playing"] = rng.uniform(45, 60, len(other)).round(1)
    return df


def load_or_create(path: str = RAW_PATH, n: int = 650) -> pd.DataFrame:
    """טוען את ה-CSV אם קיים, אחרת מייצר ושומר."""
    if os.path.exists(path):
        return pd.read_csv(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate(n=n)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    data = generate()
    data.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(data)} rows to {RAW_PATH}")
    print(data.head())
    print("\nMissing values per column:")
    print(data.isna().sum())
