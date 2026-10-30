import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Batman Movie Predictor & Game",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Theme)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .main-title {
        color: #f6ad55;
        font-weight: 900;
        font-size: 2.5rem;
        text-align: center;
        letter-spacing: -0.5px;
        margin-top: 10px;
    }
    .sub-title {
        color: #a0aec0;
        font-weight: 500;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    .result-banner {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 2px solid #f6ad55;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(246, 173, 85, 0.15);
        margin-top: 15px;
    }
    .result-score {
        font-size: 3.8rem;
        font-weight: 900;
        color: #f6ad55;
        margin: 10px 0;
    }
    
    .game-card {
        background-color: #151923;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #11141d !important;
        border-right: 1px solid #2d3748;
    }
    .sidebar-header {
        color: #f6ad55;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    .footer-text {
        color: #718096;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 40px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# MODEL PARAMETERS & CONSTANTS
# ==========================================
# y = w1*x1 (Year) + w2*x2 (Runtime) + b
W1 = 0.0185      # Weight for Release Year
W2 = 0.0142      # Weight for Runtime (Minutes)
B = -31.8540     # Bias term

# Dataset of real Batman movies for the game
DATASET = [
    {"title": "Batman (1989)", "year": 1989, "runtime": 126, "real_rating": 7.5},
    {"title": "Batman Returns (1992)", "year": 1992, "runtime": 126, "real_rating": 7.1},
    {"title": "Batman & Robin (1997)", "year": 1997, "runtime": 125, "real_rating": 3.8},
    {"title": "Batman Begins (2005)", "year": 2005, "runtime": 140, "real_rating": 8.2},
    {"title": "The Dark Knight (2008)", "year": 2008, "runtime": 152, "real_rating": 9.0},
    {"title": "The Dark Knight Rises (2012)", "year": 2012, "runtime": 164, "real_rating": 8.4},
    {"title": "The Batman (2022)", "year": 2022, "runtime": 176, "real_rating": 7.8},
]

def predict_rating(year, runtime):
    """Calculates model prediction clamped between 1.0 and 10.0"""
    pred = (W1 * year) + (W2 * runtime) + B
    return round(float(np.clip(pred, 1.0, 10.0)), 2)

# ==========================================
# SIDEBAR: Context & 3-Level Loss
# ==========================================
with st.sidebar:
    st.title("🦇 פרויקט 2: רגרסיה בשני מאפיינים")

    st.markdown('<div class="sidebar-header">🎯 סקירת המודל</div>', unsafe_allow_html=True)
    st.write("""
    **יעד ($y$):** ציון IMDb  
    **מאפיין 1 ($x_1$):** שנת יציאה  
    **מאפיין 2 ($x_2$):** אורך הסרט (בדקות)
    """)

    st.divider()

    st.markdown('<div class="sidebar-header">📉 השוואת Loss בשלוש רמות</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline", "0.750")
    col2.metric("1 Feature", "0.718")
    col3.metric("2 Features", "0.584", delta="-18.6%")

    st.caption("**מסקנה:** הוספת אורך הסרט ($x_2$) שיפרה את דיוק המודל והורידה את ה-MSE באופן משמעותי.")

    st.divider()

    st.markdown('<div class="sidebar-header">📐 משוואת המודל</div>', unsafe_allow_html=True)
    st.latex(r"\hat{y} = 0.0185 \cdot x_1 + 0.0142 \cdot x_2 - 31.854")

    st.divider()

    st.markdown('<div class="sidebar-header">📝 תיעוד השינויים מפרויקט 1</div>', unsafe_allow_html=True)
    with st.expander("מה השתנה ולמה?"):
        st.markdown("""
        * **בדאטה:** הוספנו את עמודת אורך הסרט בדקות ($x_2$).
        * **במודל:** מעבר מ-$y = w_1 x_1 + b$ למודל עם שתי משקולות $y = w_1 x_1 + w_2 x_2 + b$.
        * **ב-Loss:** ירידה מ-0.718 ל-0.584 עקב הוספת מידע רלוונטי.
        * **בממשק:** הוספת מחשבון חיזוי דו-ממדי ומשחק אינטראקטיבי נגד המודל.
        """)

# ==========================================
# MAIN PAGE NAVIGATION
# ==========================================
st.markdown('<div class="main-title">🦇 Batman Movie Rating Predictor</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔮 חיזוי בלייב (Predictor)", "🎮 משחק נגד המודל (Game)"])

# ------------------------------------------
# TAB 1: Live Predictor
# ------------------------------------------
with tab1:
    st.markdown('<div class="sub-title">הזן שנת יציאה ואורך סרט לקבלת חיזוי מודל הרגרסיה</div>', unsafe_allow_html=True)
    
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        year_input = st.number_input("שנת יציאה (Release Year - $x_1$):", min_value=1930, max_value=2050, value=2024, step=1)
        runtime_input = st.number_input("אורך הסרט בדקות (Runtime - $x_2$):", min_value=40, max_value=300, value=140, step=5)
        
        predicted_val = predict_rating(year_input, runtime_input)
        
        st.markdown(
            f"""
            <div class="result-banner">
                <div style="font-size: 1.1rem; color: #e2e8f0; font-weight: 600;">ציון IMDb מנובא עבור סרט מ-{year_input} באורך {runtime_input} דקות</div>
                <div class="result-score">{predicted_val:.2f} <span style="font-size: 1.8rem; color: #a0aec0;">/ 10</span></div>
                <div style="font-size: 0.85rem; color: #a0aec0;">חישוב: $y = 0.0185 \\cdot {year_input} + 0.0142 \\cdot {runtime_input} - 31.854$</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------
# TAB 2: Game against Model (st.session_state)
# ------------------------------------------
with tab2:
    # Initialize Session State Variables
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "current_round" not in st.session_state:
        st.session_state.current_round = 0
    if "user_score" not in st.session_state:
        st.session_state.user_score = 0
    if "model_score" not in st.session_state:
        st.session_state.model_score = 0
    if "game_history" not in st.session_state:
        st.session_state.game_history = []

    # Reset/Start Game Function
    def start_new_game():
        st.session_state.game_started = True
        st.session_state.current_round = 0
        st.session_state.user_score = 0
        st.session_state.model_score = 0
        st.session_state.game_history = []

    # SCREEN 1: Welcome & Rules
    if not st.session_state.game_started:
        st.markdown("<h3 style='text-align: center; color: #f6ad55;'>🎮 נצח את המודל (Beat the Model)</h3>", unsafe_allow_html=True)
        st.write("""
        ### חוקי המשחק:
        1. המשחק כולל **5 סיבובים**, ובכל סיבוב תוצג דוגמה של סרט באטמן אמיתי מהדאטה-סט.
        2. המשימה שלך היא לנחש את ציון ה-IMDb האמיתי של הסרט על סמך השנה ואורך הסרט.
        3. המודל ייתן חיזוי משלו במקביל!
        4. **שיטת הניקוד:** מי שהערך שלו קרוב יותר לציון האמיתי יקבל **100 נקודות** בסיבוב. במקרה של שוויון בדיוק, שניכם תזכו בנקודות!
        """)
        
        st.button("🚀 התחל משחק חדש", on_click=start_new_game, use_container_width=True, type="primary")

    # SCREEN 2: End Game Summary
    elif st.session_state.current_round >= 5:
        st.balloons()
        st.markdown("<h3 style='text-align: center; color: #f6ad55;'>🏆 המשחק הסתיים!</h3>", unsafe_allow_html=True)
        
        col_u, col_m = st.columns(2)
        col_u.metric("הניקוד שלך", f"{st.session_state.user_score} נק'")
        col_m.metric("הניקוד של המודל", f"{st.session_state.model_score} נק'")
        
        if st.session_state.user_score > st.session_state.model_score:
            st.success("🎉 כל הכבוד! ניצחת את המודל!")
        elif st.session_state.user_score < st.session_state.model_score:
            st.error("🤖 המודל ניצח הפעם! נסה שוב.")
        else:
            st.info("🤝 תיקו! ביצועים מרשימים לשני הצדדים.")

        st.markdown("### 📋 סיכום הסיבובים:")
        history_df = pd.DataFrame(st.session_state.game_history)
        st.dataframe(history_df, use_container_width=True)

        st.button("🔄 שחק שוב", on_click=start_new_game, use_container_width=True)

    # SCREEN 3: Active Gameplay
    else:
        movie = DATASET[st.session_state.current_round]
        
        st.markdown(f"### סיבוב {st.session_state.current_round + 1} מתוך 5")
        
        # Display Progress Bar
        st.progress((st.session_state.current_round) / 5)

        # Current Metrics Banner
        score_c1, score_c2 = st.columns(2)
        score_c1.metric("ניקוד משתמש", st.session_state.user_score)
        score_c2.metric("ניקוד מודל", st.session_state.model_score)

        st.markdown(f"""
        <div class="game-card">
            <h2>🎬 הסרט: {movie['title']}</h2>
            <p>📅 <b>שנת יציאה:</b> {movie['year']}</p>
            <p>⏱️ <b>אורך הסרט:</b> {movie['runtime']} דקות</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form(key=f"round_form_{st.session_state.current_round}"):
            user_guess = st.slider("מה לדעתך ציון ה-IMDb האמיתי של הסרט?", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
            submit_button = st.form_submit_button("הגש ניחוש")

        if submit_button:
            # Model prediction
            model_pred = predict_rating(movie['year'], movie['runtime'])
            real_val = movie['real_rating']

            # Calculate absolute errors
            user_error = abs(user_guess - real_val)
            model_error = abs(model_pred - real_val)

            # Determine winner of round
            if user_error < model_error:
                st.session_state.user_score += 100
                winner = "המשתמש 🙋‍♂️"
            elif model_error < user_error:
                st.session_state.model_score += 100
                winner = "המודל 🤖"
            else:
                st.session_state.user_score += 100
                st.session_state.model_score += 100
                winner = "תיקו 🤝"

            # Save history
            st.session_state.game_history.append({
                "סרט": movie['title'],
                "ציון אמיתי": real_val,
                "ניחוש משתמש": user_guess,
                "חיזוי מודל": model_pred,
                "מנצח הסיבוב": winner
            })

            st.info(f"**הציון האמיתי:** {real_val} | **חיזוי המודל:** {model_pred} | **הניחוש שלך:** {user_guess}")
            
            # Advance round
            st.session_state.current_round += 1
            st.button("לסיבוב הבא ➡️")

st.markdown('<div class="footer-text">דבאח דניאל • פרויקט 2 - למידת מכונה</div>', unsafe_allow_html=True)
