"""
TENNIS CRM - דאשבורד Streamlit

הרצה:
    cd Desktop/TENNIS_CRM
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from competitive_potential import (
    DEFAULT_WEIGHTS,
    PotentialConfig,
    compute_competitive_potential,
    potential_tier,
)
from auth import (
    authenticate,
    club_data_path,
    club_summaries,
    ensure_club_dataset,
    is_admin,
    list_member_clubs,
    load_all_players,
    load_clubs,
    register_club,
)
from model import predict_success_rate, train_regression
from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_NUMERIC,
    TennisPreprocessor,
    clean_for_display,
)
from theme import CATEGORICAL, DIVERGING, TIER_COLORS, chart, hero, inject_css

st.set_page_config(page_title="Tennis CRM", page_icon="🎾", layout="wide")
inject_css()

HAND_HE = {"right": "ימין", "left": "שמאל"}
GENDER_HE = {"male": "בן", "female": "בת"}
BACKHAND_HE = {"two_handed": "דו-ידני", "one_handed": "חד-ידני"}

PAGES = {
    "סקירת נתונים": "📊",
    "פוטנציאל תחרותי": "🎯",
    "ניקוי נתונים": "🧹",
    "מודל רגרסיה": "📈",
    "חיזוי אחוז הצלחה": "🔮",
}


# ----------------------------- טעינת נתונים -----------------------------
@st.cache_data
def get_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def get_all_players() -> pd.DataFrame:
    return load_all_players()


@st.cache_data
def get_model(df_raw: pd.DataFrame):
    return train_regression(df_raw)


# ----------------------------- מסך התחברות / רישום -----------------------------
def auth_screen() -> None:
    hero("🎾 Tennis CRM", "מערכת ניהול מועדוני טניס")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_login, tab_reg = st.tabs(["התחברות מועדון", "רישום מועדון חדש"])

        with tab_login:
            with st.form("login_form"):
                u = st.text_input("שם משתמש")
                p = st.text_input("סיסמה", type="password")
                submitted = st.form_submit_button("התחברות", width="stretch")
            if submitted:
                club = authenticate(u, p)
                if club:
                    st.session_state.club = club
                    st.rerun()
                else:
                    st.error("שם משתמש או סיסמה שגויים.")

        with tab_reg:
            with st.form("register_form"):
                name = st.text_input("שם המועדון")
                u = st.text_input("שם משתמש", key="reg_u")
                p = st.text_input("סיסמה (6 תווים לפחות)", type="password", key="reg_p")
                n = st.slider("מספר שחקנים לדאטה הסינתטית של המועדון", 200, 1500, 650, step=50)
                submitted = st.form_submit_button("צור מועדון", width="stretch")
            if submitted:
                try:
                    club = register_club(name, u, p, n=n)
                    st.session_state.club = club
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    st.caption(f"מועדונים רשומים במערכת: {len(load_clubs())}")


if "club" not in st.session_state:
    auth_screen()
    st.stop()

club = st.session_state.club
ADMIN = is_admin(club)


# ----------------------------- סרגל צד -----------------------------
st.sidebar.markdown("## 🎾 Tennis CRM" + (" · מנהל" if ADMIN else ""))
st.sidebar.markdown("**מנהל מערכת** — גישה לכל המועדונים" if ADMIN else f"**מועדון:** {club['name']}")
st.sidebar.caption(f"משתמש: {club['username']} · נוצר: {club['created_at']}")
if st.sidebar.button("🚪 התנתקות", width="stretch"):
    del st.session_state.club
    st.rerun()

st.sidebar.divider()

if ADMIN:
    members = list_member_clubs()
    labels = {"__ALL__": "🌐 כל המערכת (מצרפי)"}
    labels.update({c["club_id"]: c["name"] for c in members})
    source = st.sidebar.selectbox(
        "מקור נתונים", ["__ALL__"] + [c["club_id"] for c in members],
        format_func=lambda o: labels.get(o, o),
    )
    nav_pages = {"ניהול מועדונים": "🛠️", **PAGES}
else:
    source = None
    nav_pages = PAGES

page = st.sidebar.radio(
    "ניווט", list(nav_pages), format_func=lambda p: f"{nav_pages[p]}  {p}",
    label_visibility="collapsed",
)

st.sidebar.divider()

if ADMIN:
    if st.sidebar.button("🔄 רענן נתונים", width="stretch"):
        get_raw_data.clear()
        get_all_players.clear()
        st.rerun()
    if source == "__ALL__":
        raw = get_all_players().copy()
        st.sidebar.caption("איחוד כל קובצי המועדונים ב-`data/clubs/`")
    else:
        raw = get_raw_data(club_data_path(source)).copy()
        st.sidebar.caption(f"קובץ: `data/clubs/{source}/players_raw.csv`")
else:
    DATA_PATH = ensure_club_dataset(club["club_id"])
    if st.sidebar.button("🔄 ייצר דאטה סינתטית חדשה", width="stretch"):
        ensure_club_dataset(club["club_id"], reseed=True)
        get_raw_data.clear()
        st.rerun()
    st.sidebar.caption(f"קובץ הנתונים של המועדון:\n`data/clubs/{club['club_id']}/players_raw.csv`")
    raw = get_raw_data(DATA_PATH).copy()

if raw.empty:
    hero("Tennis CRM", "אין עדיין נתונים")
    st.warning("אין מועדונים רשומים במערכת. יש לרשום מועדון אחד לפחות כדי לראות נתונים.")
    st.stop()

raw["competitive_potential"] = compute_competitive_potential(
    raw.assign(
        fitness_level=raw["fitness_level"].fillna(raw["fitness_level"].median()),
        dominant_hand=raw["dominant_hand"].fillna("right"),
    ),
    PotentialConfig(),
)


# ===================== עמוד מנהל: ניהול מועדונים =====================
if page == "ניהול מועדונים":
    hero("ניהול מועדונים", "מבט מנהל על כל המועדונים במערכת")
    summ = club_summaries()
    if summ.empty:
        st.info("אין מועדונים רשומים עדיין.")
        st.stop()

    total_players = int(summ["שחקנים"].sum())
    weighted_sr = (summ["אחוז הצלחה ממוצע"] * summ["שחקנים"]).sum() / total_players
    c1, c2, c3 = st.columns(3)
    c1.metric("מועדונים", len(summ))
    c2.metric('סה"כ שחקנים', total_players)
    c3.metric("אחוז הצלחה ממוצע (משוקלל)", f"{weighted_sr:.1f}%")

    st.subheader("טבלת מועדונים")
    st.dataframe(summ, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("שחקנים לפי מועדון")
        fig = px.bar(summ, x="מועדון", y="שחקנים", text="שחקנים",
                     color_discrete_sequence=CATEGORICAL)
        chart(fig, height=340, showlegend=False)
    with c2:
        st.subheader("אחוז הצלחה ממוצע לפי מועדון")
        fig = px.bar(summ.sort_values("אחוז הצלחה ממוצע"), x="אחוז הצלחה ממוצע", y="מועדון",
                     orientation="h", color="אחוז הצלחה ממוצע", color_continuous_scale="Teal")
        chart(fig, height=340, showlegend=False)
    st.stop()


# ============================= עמוד 1: סקירה =============================
if page == "סקירת נתונים":
    hero("סקירת נתונים", "מבט-על על שחקני המועדון · משתנים כמותיים מול קטגוריאליים")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("שחקנים", len(raw))
    c2.metric("גיל ממוצע", f"{raw['age'].mean():.1f}")
    c3.metric("אחוז הצלחה ממוצע", f"{raw['success_rate'].mean():.1f}%")
    c4.metric("שמאליים", f"{(raw['dominant_hand'] == 'left').mean() * 100:.0f}%")

    st.subheader("טבלת נתונים גולמיים")
    disp = raw.copy()
    disp["gender"] = disp["gender"].map(GENDER_HE)
    disp["dominant_hand"] = disp["dominant_hand"].map(HAND_HE)
    disp["backhand_type"] = disp["backhand_type"].map(BACKHAND_HE)
    st.dataframe(disp, width="stretch", height=330)

    st.subheader("התפלגויות")
    col = st.selectbox("בחר משתנה", RAW_NUMERIC + ["success_rate", "competitive_potential"])
    fig = px.histogram(raw, x=col, color="gender", nbins=30, marginal="box",
                       color_discrete_sequence=CATEGORICAL, opacity=0.85)
    fig.for_each_trace(lambda t: t.update(name=GENDER_HE.get(t.name, t.name)))
    chart(fig, height=380)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("אחוז הצלחה מול גיל")
        fig = px.scatter(raw, x="age", y="success_rate", color="fitness_level",
                         color_continuous_scale="Teal", opacity=0.65,
                         labels={"age": "גיל", "success_rate": "אחוז הצלחה", "fitness_level": "כושר"})
        chart(fig)
    with c2:
        st.subheader("מטריצת מתאם (משתנים כמותיים)")
        corr = raw[RAW_NUMERIC + ["success_rate", "competitive_potential"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=DIVERGING, zmin=-1, zmax=1, aspect="auto")
        chart(fig)


# ===================== עמוד 2: פוטנציאל תחרותי =====================
elif page == "פוטנציאל תחרותי":
    hero("פוטנציאל תחרותי", "ציון 0-100 מ-4 משתנים: יד חזקה · רמת כושר · מין · גיל")

    st.sidebar.divider()
    st.sidebar.markdown("### ⚖️ משקלי הפוטנציאל")
    w = {
        "fitness": st.sidebar.slider("משקל כושר", 0.0, 60.0, DEFAULT_WEIGHTS["fitness"], 1.0),
        "age": st.sidebar.slider("משקל גיל", 0.0, 60.0, DEFAULT_WEIGHTS["age"], 1.0),
        "hand": st.sidebar.slider("משקל יד חזקה", 0.0, 40.0, DEFAULT_WEIGHTS["hand"], 1.0),
        "gender": st.sidebar.slider("משקל מין", 0.0, 40.0, DEFAULT_WEIGHTS["gender"], 1.0),
    }
    age_peak = st.sidebar.slider("גיל שיא", 18, 35, 25)
    cfg = PotentialConfig(weights=w, age_peak=float(age_peak))

    base = raw.copy()
    base["fitness_level"] = base["fitness_level"].fillna(base["fitness_level"].median())
    base["dominant_hand"] = base["dominant_hand"].fillna("right")
    base["competitive_potential"] = compute_competitive_potential(base, cfg)
    base["tier"] = base["competitive_potential"].map(potential_tier)

    c1, c2, c3 = st.columns(3)
    c1.metric("פוטנציאל ממוצע", f"{base['competitive_potential'].mean():.1f}")
    c2.metric("שחקני עילית", int((base["tier"] == "עילית").sum()))
    c3.metric("מתאם פוטנציאל↔הצלחה", f"{base['competitive_potential'].corr(base['success_rate']):.2f}")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("פוטנציאל מול אחוז הצלחה בפועל")
        fig = px.scatter(
            base, x="competitive_potential", y="success_rate", color="tier",
            hover_data=["full_name", "age", "fitness_level"], color_discrete_map=TIER_COLORS,
            labels={"competitive_potential": "פוטנציאל תחרותי", "success_rate": "אחוז הצלחה", "tier": "דרגה"},
        )
        chart(fig, height=420)
    with c2:
        st.subheader("התפלגות דרגות")
        fig = px.pie(base, names="tier", hole=0.55, color="tier", color_discrete_map=TIER_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        chart(fig, height=420, showlegend=False)

    st.subheader("דירוג שחקנים לפי פוטנציאל")
    view = base[["full_name", "gender", "age", "dominant_hand", "fitness_level",
                 "competitive_potential", "tier", "success_rate"]].copy()
    view["gender"] = view["gender"].map(GENDER_HE)
    view["dominant_hand"] = view["dominant_hand"].map(HAND_HE)
    view = view.sort_values("competitive_potential", ascending=False).reset_index(drop=True)
    view.columns = ["שם", "מין", "גיל", "יד חזקה", "כושר", "פוטנציאל", "דרגה", "אחוז הצלחה"]
    st.dataframe(view, width="stretch", height=380)


# ======================= עמוד 3: ניקוי נתונים =======================
elif page == "ניקוי נתונים":
    hero("ניקוי נתונים", "השלמת חוסרים · הסרת קיצונים · קידוד · נרמול")
    pre = TennisPreprocessor()
    X, y = pre.fit_transform(raw.drop(columns=["competitive_potential"]))
    rep = pre.report_

    st.subheader("1 · ערכים חסרים והשלמתם")
    miss = pd.DataFrame({"עמודה": list(rep.missing_before), "חסרים": list(rep.missing_before.values())})
    miss = miss[miss["חסרים"] > 0].copy()
    miss["שיטת השלמה"] = miss["עמודה"].map(
        lambda c: "חציון" if c in NUMERIC_FEATURES else ("שכיח" if c in CATEGORICAL_FEATURES else "-")
    )
    miss["ערך שהוזן"] = miss["עמודה"].map(lambda c: str(rep.imputation_values.get(c, "-")))
    c1, c2 = st.columns([1, 1])
    c1.dataframe(miss, width="stretch", hide_index=True)
    with c2:
        if not miss.empty:
            fig = px.bar(miss, x="עמודה", y="חסרים", color_discrete_sequence=CATEGORICAL, text="חסרים")
            chart(fig, height=300, showlegend=False)

    st.subheader("2 · הסרת ערכים קיצוניים (IQR, מקדם 1.5)")
    out = pd.DataFrame({
        "עמודה": list(rep.outliers_removed),
        "גבול תחתון": [rep.outlier_bounds[k][0] for k in rep.outliers_removed],
        "גבול עליון": [rep.outlier_bounds[k][1] for k in rep.outliers_removed],
        "שורות שהוסרו": list(rep.outliers_removed.values()),
    })
    st.dataframe(out, width="stretch", hide_index=True)
    st.info(f"לפני: {rep.n_input} שורות → אחרי: {rep.n_after_outliers} שורות "
            f"({rep.n_input - rep.n_after_outliers} הוסרו)")

    col = st.selectbox("הצג התפלגות לפני/אחרי", RAW_NUMERIC)
    cleaned = clean_for_display(raw.drop(columns=["competitive_potential"]), pre)
    cmp = pd.concat([
        raw[[col]].assign(שלב="גולמי (אחרי השלמה)"),
        cleaned[[col]].assign(שלב="אחרי הסרת קיצונים"),
    ])
    fig = px.box(cmp, x="שלב", y=col, color="שלב", color_discrete_sequence=CATEGORICAL, points="outliers")
    chart(fig, height=340, showlegend=False)

    st.subheader("3 · קידוד קטגוריאלי (One-Hot)")
    st.write(f"עמודות מקוריות: `{CATEGORICAL_FEATURES}`")
    st.write(f"עמודות אחרי קידוד: `{rep.encoded_columns}`")

    st.subheader("4 · נרמול כמותי (StandardScaler)")
    st.write(f"נורמלו: `{NUMERIC_FEATURES}` — ממוצע 0, סטיית תקן 1")
    st.dataframe(X[NUMERIC_FEATURES].describe().round(3), width="stretch")

    st.subheader("מטריצת המאפיינים הסופית (X)")
    st.dataframe(X.head(15), width="stretch")


# ======================= עמוד 4: מודל רגרסיה =======================
elif page == "מודל רגרסיה":
    hero("מודל רגרסיה לינארית", "חיזוי אחוז ההצלחה · הערכה על קבוצת מבחן")
    res = get_model(raw.drop(columns=["competitive_potential"]))
    m = res.metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", m["R2"])
    c2.metric("Adjusted R²", m["Adjusted R2"])
    c3.metric("RMSE", m["RMSE"])
    c4.metric("MAE", m["MAE"])
    st.caption(f"אימון: {m['n_train']} תצפיות · מבחן: {m['n_test']} תצפיות · "
               f"{m['n_features']} משתנים מסבירים (אחרי קידוד)")

    with st.expander("איך מחושבות המטריקות?"):
        st.latex(r"R^2_{adj} = 1 - (1 - R^2)\,\frac{n - 1}{n - p - 1}")
        st.write("n = מספר תצפיות המבחן, p = מספר המשתנים המסבירים. מתקנן כלפי מטה על משתנים שאינם תורמים.")
        st.latex(r"RMSE = \sqrt{\tfrac{1}{n}\textstyle\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ערכים אמיתיים מול חזויים")
        dfp = pd.DataFrame({"אמיתי": res.y_test.values, "חזוי": res.y_pred})
        fig = px.scatter(dfp, x="אמיתי", y="חזוי", color_discrete_sequence=CATEGORICAL, opacity=0.7)
        lo, hi = float(dfp.min().min()), float(dfp.max().max())
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dash", color="#9CA3AF"))
        chart(fig)
    with c2:
        st.subheader("שאריות (residuals)")
        resid = res.y_test.values - res.y_pred
        fig = px.scatter(x=res.y_pred, y=resid, color_discrete_sequence=CATEGORICAL, opacity=0.7,
                         labels={"x": "חזוי", "y": "שארית"})
        fig.add_hline(y=0, line_dash="dash", line_color="#9CA3AF")
        chart(fig)

    st.subheader("מקדמי הרגרסיה")
    coefs = res.coefficients[res.coefficients["feature"] != "intercept"].sort_values("coefficient")
    fig = px.bar(coefs, x="coefficient", y="feature", orientation="h",
                 color="coefficient", color_continuous_scale=DIVERGING,
                 labels={"coefficient": "מקדם", "feature": "משתנה"})
    chart(fig, height=380, showlegend=False)
    st.dataframe(res.coefficients.round(4), width="stretch", hide_index=True)
    st.caption("סקאלה מנורמלת: לכל משתנה כמותי, שינוי של סטיית תקן אחת משנה את אחוז ההצלחה בכמות המקדם. "
               "למשתנים מקודדים (One-Hot) המקדם הוא ההפרש מקטגוריית הבסיס.")


# ===================== עמוד 5: חיזוי =====================
elif page == "חיזוי אחוז הצלחה":
    hero("חיזוי אחוז הצלחה", "הזנת פרופיל שחקן → חיזוי מהמודל + ציון פוטנציאל")
    res = get_model(raw.drop(columns=["competitive_potential"]))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("מין", ["male", "female"], format_func=lambda x: GENDER_HE[x])
            age = st.slider("גיל", 8, 70, 24)
            hand = st.selectbox("יד חזקה", ["right", "left"], format_func=lambda x: HAND_HE[x])
        with c2:
            fitness = st.slider("רמת כושר (1-10)", 1, 10, 6)
            years = st.slider("שנות משחק", 0.0, 40.0, 6.0, 0.5)
        with c3:
            hours = st.slider("שעות אימון שבועיות", 0.0, 25.0, 5.0, 0.5)
            height = st.slider("גובה (ס\"מ)", 130, 210, 175)
            backhand = st.selectbox("סוג בקהנד", ["two_handed", "one_handed"],
                                    format_func=lambda x: BACKHAND_HE[x])

    new = pd.DataFrame([{
        "gender": gender, "age": age, "dominant_hand": hand, "backhand_type": backhand,
        "fitness_level": fitness, "years_playing": years,
        "weekly_training_hours": hours, "height_cm": height,
    }])
    pred = float(predict_success_rate(res, new)[0])
    pot = float(compute_competitive_potential(new, PotentialConfig()).iloc[0])

    c1, c2 = st.columns(2)
    c1.metric("אחוז הצלחה חזוי (רגרסיה)", f"{pred:.1f}%")
    c2.metric("פוטנציאל תחרותי", f"{pot:.1f} — {potential_tier(pot)}")
    st.caption(f"דיוק המודל: RMSE ≈ {res.metrics['RMSE']} נק' אחוז · Adjusted R² = {res.metrics['Adjusted R2']}. "
               "טווח האי-ודאות סביב החיזוי ≈ ± RMSE.")

    st.subheader("רגישות החיזוי לרמת הכושר")
    sweep = pd.concat([new.assign(fitness_level=f) for f in range(1, 11)], ignore_index=True)
    sweep["חיזוי"] = predict_success_rate(res, sweep)
    fig = px.line(sweep, x="fitness_level", y="חיזוי", markers=True,
                  color_discrete_sequence=CATEGORICAL, labels={"fitness_level": "רמת כושר"})
    fig.update_traces(line=dict(width=3))
    chart(fig, height=340)
