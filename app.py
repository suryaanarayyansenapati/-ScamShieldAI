
import streamlit as st
import joblib
import re
from pathlib import Path
from datetime import datetime
from io import StringIO

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "scamshield_model.pkl"

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #0B0B0B, #171017);
        color: #F5F2ED;
    }

    h1, h2, h3 {
        color: #F5F2ED;
    }

    .hero {
        padding: 25px;
        border: 1px solid #8B0D1A;
        border-radius: 16px;
        background: linear-gradient(120deg, #25080D, #111111);
        margin-bottom: 20px;
    }

    .hero p {
        color: #D8D1D1;
    }

    .result-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #454545;
        background: #151515;
        margin: 8px 0;
    }

    div.stButton > button {
        background-color: #8B0D1A;
        color: white;
        border: 1px solid #A51C2B;
        border-radius: 9px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #A51C2B;
        color: white;
        border-color: #F5F2ED;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 3. LOAD MACHINE-LEARNING MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, "Model file not found."

    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as error:
        return None, str(error)


model, model_error = load_model()

# ==========================================
# 4. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_report" not in st.session_state:
    st.session_state.latest_report = None

# ==========================================
# 5. SCAM WARNING-SIGN ANALYSIS
# ==========================================

def analyse_warning_signs(message):
    text = message.lower()
    warnings = []
    score = 0

    urgency_patterns = [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bact now\b",
        r"\bwithin \d+ hours?\b",
        r"\btoday\b",
        r"\bwill be blocked\b",
        r"\bwill be disconnected\b",
        r"\bsuspended\b",
    ]

    sensitive_patterns = [
        r"\botp\b",
        r"\bpassword\b",
        r"\bpin\b",
        r"\bverify your account\b",
        r"\bbank details\b",
        r"\bcard details\b",
        r"\baccount details\b",
    ]

    link_patterns = [
        r"https?://",
        r"\bwww\.",
        r"\bclick here\b",
        r"\bclick this link\b",
        r"\bverify.*link\b",
    ]

    prize_patterns = [
        r"\byou have won\b",
        r"\byou won\b",
        r"\blottery\b",
        r"\bfree prize\b",
        r"\bclaim your reward\b",
        r"\bfree gift\b",
        r"\bcongratulations\b",
    ]

    money_patterns = [
        r"\bprocessing fee\b",
        r"\bpay now\b",
        r"\bsend money\b",
        r"\bpayment required\b",
        r"\btransfer money\b",
        r"\bpay.*to receive\b",
    ]

    groups = [
        (
            urgency_patterns,
            25,
            "Urgent language, pressure, or threats",
        ),
        (
            sensitive_patterns,
            30,
            "Request for sensitive information or account verification",
        ),
        (
            link_patterns,
            25,
            "Link or instruction that may require careful verification",
        ),
        (
            prize_patterns,
            20,
            "Prize, reward, or free-gift claim",
        ),
        (
            money_patterns,
            25,
            "Request for payment or money transfer",
        ),
    ]

    for patterns, points, description in groups:
        if any(re.search(pattern, text) for pattern in patterns):
            score += points
            warnings.append(description)

    return min(score, 100), warnings


# ==========================================
# 6. MACHINE-LEARNING PREDICTION
# ==========================================

def predict_message(message):
    if model is None:
        return "Unavailable", None, model_error

    try:
        prediction = str(model.predict([message])[0]).strip().lower()

        if prediction not in ("spam", "ham"):
            return (
                "Unavailable",
                None,
                f"Unexpected model label: {prediction}",
            )

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)

            if "spam" in classes:
                spam_index = classes.index("spam")
                probability = float(probabilities[spam_index]) * 100

        return prediction.upper(), probability, None

    except Exception as error:
        return "Unavailable", None, str(error)


# ==========================================
# 7. SAFETY ADVICE
# ==========================================

def get_safety_advice(level, warnings):
    advice = []

    if level == "High":
        advice.append(
            "Pause before responding. Verify the sender using an "
            "official website, app, or known phone number."
        )

    elif level == "Medium":
        advice.append(
            "Check the sender and verify the request independently "
            "before taking action."
        )

    else:
        advice.append(
            "No strong warning signs were found by the current checks. "
            "That does not guarantee the message is safe."
        )

    if any("sensitive information" in item for item in warnings):
        advice.append(
            "Never share an OTP, password, PIN, or banking credentials "
            "in response to an unsolicited message."
        )

    if any("Link or instruction" in item for item in warnings):
        advice.append(
            "Avoid opening unexpected links. Visit the official website "
            "directly instead."
        )

    if any("payment or money" in item for item in warnings):
        advice.append(
            "Do not pay an unexpected fee until the request has been "
            "independently verified."
        )

    advice.append(
        "ScamShield AI is an experimental educational tool, not a "
        "replacement for independent verification."
    )

    return advice


# ==========================================
# 8. HEADER
# ==========================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ ScamShield AI</h1>
        <p>
            An experimental machine-learning and rule-based tool
            for identifying potential scam messages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🛡️ ScamShield AI")
    st.caption("Version 1.0 • Educational prototype")
    st.divider()

    if model is not None:
        st.success("Machine-learning model loaded.")
    else:
        st.error("Machine-learning model unavailable.")
        st.caption(str(model_error))

    st.markdown("### How it works")
    st.write("1. Enter a message.")
    st.write("2. Review the ML prediction.")
    st.write("3. Check warning signs and safety advice.")

    st.divider()
    st.caption(
        "Privacy note: this application analyses text entered "
        "in the current session. Do not enter real passwords, "
        "OTPs, or other sensitive information."
    )

# ==========================================
# 9. MESSAGE INPUT
# ==========================================

st.subheader("Analyse a message")

st.write(
    "Paste a suspicious SMS, email excerpt, or other text "
    "you want to examine."
)

with st.form("analysis_form"):
    message = st.text_area(
        "Message text",
        placeholder=(
            "Example: URGENT! Your account will be blocked. "
            "Verify your details immediately."
        ),
        height=160,
        max_chars=10000,
    )

    submitted = st.form_submit_button(
        "🔍 Analyse Message",
        use_container_width=True,
    )

if submitted:
    if not message.strip():
        st.warning("Please enter a message before analysing it.")

    else:
        clean_message = message.strip()

        rule_score, warnings = analyse_warning_signs(clean_message)
        prediction, probability, prediction_error = predict_message(
            clean_message
        )

        # The score is a heuristic, not a probability.
        risk_score = rule_score

        if prediction == "SPAM":
            risk_score = max(risk_score, 60)

        if risk_score >= 60:
            risk_level = "High"
        elif risk_score >= 25:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        advice = get_safety_advice(risk_level, warnings)

        result = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": clean_message,
            "prediction": prediction,
            "probability": probability,
            "score": risk_score,
            "level": risk_level,
            "warnings": warnings,
            "advice": advice,
            "prediction_error": prediction_error,
        }

        st.session_state.history.insert(0, result)
        st.session_state.latest_report = result

# ==========================================
# 10. DISPLAY LATEST RESULT
# ==========================================

result = st.session_state.latest_report

if result is not None:
    st.divider()
    st.subheader("Analysis results")

    if result["prediction"] == "SPAM":
        st.error("ML prediction: SPAM")
    elif result["prediction"] == "HAM":
        st.success("ML prediction: HAM — likely ordinary message")
    else:
        st.warning("ML prediction unavailable")

    if result["prediction_error"]:
        st.caption(
            "Prediction issue: " + result["prediction_error"]
        )

    if result["probability"] is not None:
        st.metric(
            "Model-estimated spam probability",
            f"{result['probability']:.1f}%",
        )
        st.caption(
            "This is the model's estimate, not a verified measure "
            "of real-world accuracy."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Heuristic risk score", f"{result['score']}/100")

    with col2:
        st.metric("Risk level", result["level"])

    st.progress(result["score"] / 100)

    st.caption(
        "The risk score combines warning-sign rules with a minimum "
        "score when the model predicts SPAM. It is not a probability."
    )

    st.markdown("### ⚠️ Warning signs")

    if result["warnings"]:
        for warning in result["warnings"]:
            st.write(f"- {warning}")
    else:
        st.write("No warning signs were detected by the current rules.")

    st.markdown("### 🛡️ Safety advice")

    for item in result["advice"]:
        st.write(f"- {item}")

    # ======================================
    # 11. DOWNLOAD REPORT
    # ======================================

    report = [
        "SCAMSHIELD AI - MESSAGE ANALYSIS REPORT",
        "=" * 42,
        f"Analysis time: {result['time']}",
        "",
        "MESSAGE:",
        result["message"],
        "",
        f"ML prediction: {result['prediction']}",
        (
            "Model-estimated spam probability: "
            + (
                f"{result['probability']:.1f}%"
                if result["probability"] is not None
                else "Not available"
            )
        ),
        f"Heuristic risk score: {result['score']}/100",
        f"Risk level: {result['level']}",
        "",
        "WARNING SIGNS:",
    ]

    if result["warnings"]:
        report.extend(f"- {item}" for item in result["warnings"])
    else:
        report.append("- No warning signs detected by current rules.")

    report.extend(["", "SAFETY ADVICE:"])
    report.extend(f"- {item}" for item in result["advice"])

    report.extend(
        [
            "",
            "DISCLAIMER:",
            "This is an experimental educational prototype. "
            "Its predictions may be incorrect.",
            "A low risk score or HAM prediction does not guarantee "
            "that a message is safe.",
        ]
    )

    st.download_button(
        "⬇️ Download analysis report",
        data="\n".join(report),
        file_name="scamshield_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ==========================================
# 12. ANALYSIS HISTORY
# ==========================================

st.divider()
st.subheader("📋 Analysis history")

if st.session_state.history:
    st.caption(
        f"{len(st.session_state.history)} message(s) analysed "
        "in this session."
    )

    for index, item in enumerate(st.session_state.history, start=1):
        with st.expander(
            f"{index}. {item['time']} — "
            f"{item['prediction']} — {item['level']} risk"
        ):
            st.write("**Message:**")
            st.write(item["message"])
            st.write(f"**Risk score:** {item['score']}/100")
            st.write("**Warning signs:**")

            if item["warnings"]:
                for warning in item["warnings"]:
                    st.write(f"- {warning}")
            else:
                st.write("No warning signs detected.")

    history_text = StringIO()
    history_text.write("SCAMSHIELD AI - SESSION HISTORY\n")
    history_text.write("=" * 42 + "\n\n")

    for item in st.session_state.history:
        history_text.write(f"Time: {item['time']}\n")
        history_text.write(f"Message: {item['message']}\n")
        history_text.write(f"Prediction: {item['prediction']}\n")
        history_text.write(f"Risk score: {item['score']}/100\n")
        history_text.write(f"Risk level: {item['level']}\n")
        history_text.write("-" * 42 + "\n")

    st.download_button(
        "⬇️ Download session history",
        data=history_text.getvalue(),
        file_name="scamshield_session_history.txt",
        mime="text/plain",
    )

    if st.button("🗑️ Clear session history"):
        st.session_state.history = []
        st.session_state.latest_report = None
        st.rerun()

else:
    st.info("Your analysed messages will appear here.")

# ==========================================

# ==========================================
 
# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<div style='text-align:center;'>"
    "<p>Built with ❤️ by</p>"
    "<h3 style='color:#C6283D;'>Suryanarayan Senapati</h3>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;'>Let's Connect</p>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.write("")

with col2:
    st.link_button(
        "LinkedIn",
        "https://www.linkedin.com/in/suryanarayan-senapati-43229a38b/",
        use_container_width=True,
    )
    st.link_button(
        "Instagram DM",
        "https://www.instagram.com/suryathroughlenses_/",
        use_container_width=True,
    )

with col3:
    st.write("")

st.markdown(
    "<p style='text-align:center;color:#888888;font-size:12px;'>"
    "ScamShield AI Version 1.0 | Python | Streamlit | Machine Learning"
    "</p>",
    unsafe_allow_html=True,
)

st.caption(
    "Educational prototype. Experimental predictions may be incorrect."
)
