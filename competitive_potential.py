"""
חישוב "פוטנציאל תחרותי" לכל שחקן/ית.

הציון (0-100) מבוסס על ארבעה משתנים בלבד, כפי שהוגדר:
    - היד החזקה (dominant_hand)  - קטגוריאלי
    - רמת הכושר (fitness_level)  - כמותי/אורדינלי 1-10
    - מין (gender)               - קטגוריאלי
    - גיל (age)                  - כמותי

זהו ציון דטרמיניסטי (לא מודל סטטיסטי) - סכום משוקלל של ארבעה רכיבים.
המשקלים ניתנים לכיוונון מהדאשבורד.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "fitness": 40.0,   # משקל מרבי לרכיב הכושר
    "age": 30.0,       # משקל מרבי לרכיב הגיל
    "hand": 15.0,      # משקל מרבי לרכיב היד
    "gender": 15.0,    # משקל מרבי לרכיב המין
}


@dataclass
class PotentialConfig:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    age_peak: float = 25.0
    age_spread: float = 11.0
    left_hand_advantage: float = 1.0   # 1.0 = יתרון מלא לשמאלי, 0.6 לימני
    right_hand_advantage: float = 0.6


def _fitness_component(fitness: pd.Series, w: float) -> pd.Series:
    return (fitness.clip(1, 10) / 10.0) * w


def _age_component(age: pd.Series, w: float, peak: float, spread: float) -> pd.Series:
    return np.exp(-((age - peak) / spread) ** 2) * w


def _hand_component(hand: pd.Series, w: float, left_adv: float, right_adv: float) -> pd.Series:
    factor = hand.map({"left": left_adv, "right": right_adv}).fillna(right_adv)
    return factor * w


def _gender_component(gender: pd.Series, w: float) -> pd.Series:
    # אפקט קרוב-לניטרלי: שני המינים מקבלים כמעט את מלוא המשקל
    factor = gender.map({"male": 1.0, "female": 0.95}).fillna(0.97)
    return factor * w


def compute_competitive_potential(df: pd.DataFrame, config: PotentialConfig | None = None) -> pd.Series:
    """מחזיר Series של ציון פוטנציאל תחרותי (0-100) באותו אינדקס של df."""
    cfg = config or PotentialConfig()
    w = cfg.weights

    score = (
        _fitness_component(df["fitness_level"], w["fitness"])
        + _age_component(df["age"], w["age"], cfg.age_peak, cfg.age_spread)
        + _hand_component(df["dominant_hand"], w["hand"], cfg.left_hand_advantage, cfg.right_hand_advantage)
        + _gender_component(df["gender"], w["gender"])
    )
    total_w = sum(w.values())
    return (score / total_w * 100.0).clip(0, 100).round(1)


def potential_tier(score: float) -> str:
    if score >= 75:
        return "עילית"
    if score >= 60:
        return "גבוה"
    if score >= 45:
        return "בינוני"
    return "מתפתח"
