"""
app.py
------
Streamlit frontend for the Smart Email Assistant.
5th-Semester Minor Project — ML & NLP

Run with:
    streamlit run app.py

NOTE: Run  python preload_models.py  ONCE first to pre-save all models to disk.
"""

import streamlit as st
from email_assistant import (
    load_spam_model,
    load_bart_model,
    process_email,
    SPAM_THRESHOLD,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Email Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — all colors are explicit so
#  they look correct in both light & dark mode
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font ── */
html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 60%, #3b82f6 100%);
    border-radius: 16px;
    padding: 30px 36px;
    color: #ffffff !important;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(37,99,235,0.25);
}
.header-banner h1 {
    font-size: 1.9rem; font-weight: 700;
    margin: 0 0 4px 0; color: #ffffff !important;
}
.header-banner p {
    font-size: 0.98rem; opacity: 0.9; margin: 0;
    color: #ffffff !important;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.74rem;
    margin-top: 10px; margin-right: 5px;
    font-weight: 600; color: #ffffff !important;
}

/* ── Generic card wrapper ── */
.card {
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    border: 1.5px solid;
}
/* Spam card — red tones */
.card-spam {
    background-color: #fff0f0;
    border-color: #f87171;
    color: #1a1a1a !important;
}
/* Safe card — green tones */
.card-safe {
    background-color: #f0fdf4;
    border-color: #4ade80;
    color: #1a1a1a !important;
}
/* Summary card — blue tones */
.card-summary {
    background-color: #eff6ff;
    border-color: #60a5fa;
    color: #1a1a1a !important;
}
/* Reminder card — yellow tones */
.card-reminder {
    background-color: #fefce8;
    border-color: #facc15;
    color: #1a1a1a !important;
}

/* ── Section heading inside card ── */
.card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e3a5f !important;
    margin: 0 0 12px 0;
}

/* ── Spam/Safe badge ── */
.badge-spam {
    display: inline-block;
    background: #dc2626;
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 5px 20px;
    border-radius: 999px;
}
.badge-safe {
    display: inline-block;
    background: #16a34a;
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 5px 20px;
    border-radius: 999px;
}

/* ── Probability label ── */
.prob-label {
    font-size: 0.84rem;
    font-weight: 600;
    color: #1e293b !important;
    margin-bottom: 4px;
}

/* ── Summary text ── */
.summary-text {
    font-size: 0.93rem;
    line-height: 1.75;
    color: #1e293b !important;
    margin-top: 6px;
}

/* ── Reminder item ── */
.rem-item {
    background: #ffffff;
    border: 1.5px solid #fbbf24;
    border-left: 5px solid #d97706;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    color: #1a1a1a !important;
}
.rem-item .rem-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #1e3a5f !important;
    margin-bottom: 4px;
}
.rem-item .rem-meta {
    font-size: 0.84rem;
    color: #374151 !important;
}
.rem-item .rem-meta span {
    color: #374151 !important;
}

/* ── Empty-state box ── */
.empty-state {
    border: 2px dashed #94a3b8;
    border-radius: 12px;
    padding: 50px 24px;
    text-align: center;
    color: #64748b !important;
}
.empty-state .empty-icon { font-size: 2.5rem; margin-bottom: 10px; }
.empty-state .empty-title {
    font-size: 1rem; font-weight: 600; color: #475569 !important;
}
.empty-state .empty-sub {
    font-size: 0.85rem; color: #94a3b8 !important; margin-top: 4px;
}

/* ── Info box ── */
.info-box {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #334155 !important;
    margin-top: 10px;
}
.info-box ul { margin: 6px 0 0 0; padding-left: 18px; }
.info-box li { margin-bottom: 3px; color: #334155 !important; }

/* ── Model status bar ── */
.model-status {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.84rem;
    font-weight: 600;
    color: #166534 !important;
    margin-bottom: 18px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 0.78rem;
    margin-top: 36px;
    padding-top: 14px;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS ONCE AT STARTUP (from disk cache)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_spam_models():
    return load_spam_model()

@st.cache_resource(show_spinner=False)
def get_bart_models():
    return load_bart_model()

# Load silently — instant if preload_models.py was already run
with st.spinner("⚙️ Loading models from cache…"):
    try:
        tfidf, log_reg = get_spam_models()
        tokenizer, bart_model, device = get_bart_models()
        models_ok = True
    except Exception as load_err:
        models_ok = False
        load_error_msg = str(load_err)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>📧 Smart Email Assistant</h1>
    <p>Using Machine Learning &amp; Natural Language Processing</p>
    <span class="badge">🤖 TF-IDF + Logistic Regression</span>
    <span class="badge">🧠 BART Summarization</span>
    <span class="badge">📅 Date &amp; Time Extraction (re)</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  THRESHOLD SLIDER  (defined here so it's
#  available to both the status bar and col_left)
# ─────────────────────────────────────────────
st.markdown("#### 🎚️ Spam Detection Threshold")

threshold_pct = st.slider(
    label="threshold_slider",
    min_value=10,
    max_value=99,
    value=80,
    step=1,
    format="%d%%",
    label_visibility="collapsed",
)
threshold = threshold_pct / 100.0   # convert to 0–1 for the model

if threshold_pct < 50:
    hint_color, hint_icon, hint_text = "#b91c1c", "🔴", "Very strict — many emails may be flagged as spam"
elif threshold_pct < 70:
    hint_color, hint_icon, hint_text = "#d97706", "🟠", "Strict — catches most spam with some false positives"
elif threshold_pct < 85:
    hint_color, hint_icon, hint_text = "#15803d", "🟢", "Balanced — recommended for most use cases"
else:
    hint_color, hint_icon, hint_text = "#1d4ed8", "🔵", "Lenient — only very obvious spam is blocked"

st.markdown(
    f'<p style="font-size:0.82rem; color:{hint_color}; margin-top:-8px; font-weight:600;">'
    f'{hint_icon} &nbsp;{hint_text}</p>',
    unsafe_allow_html=True,
)

# Model status strip
if models_ok:
    st.markdown(
        f'<div class="model-status">✅ All models loaded and ready &nbsp;|&nbsp; '
        f'Device: <code>{device}</code> &nbsp;|&nbsp; '
        f'Spam threshold: <strong>{threshold_pct}%</strong> (use slider to adjust)</div>',
        unsafe_allow_html=True,
    )
else:
    st.error(
        f"❌ Model loading failed: {load_error_msg}\n\n"
        "Please run  `python preload_models.py`  first, then restart the app."
    )
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────────
#  LAYOUT — two columns
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

# ══════════════════════════════════════════════
#  LEFT COLUMN — Input
# ══════════════════════════════════════════════
with col_left:
    st.markdown("### ✍️ Paste Email Content")

    # Sample email picker
    sample_emails = {
        "— Select a sample email to try —": "",
        "📄 Safe Email — Meeting Reminder": (
            "Dear Team,\n\n"
            "This is a reminder that our quarterly review meeting is scheduled "
            "on Wednesday at 2:00 PM in Conference Room B.\n\n"
            "Please complete your reports and submit them by Tuesday noon.\n"
            "The agenda will be shared by Monday morning.\n\n"
            "Also, the registration deadline for the internal training program "
            "is 15 December 2026. Please register before 5:00 PM.\n\n"
            "Best regards,\nProject Manager"
        ),
        "📄 Safe Email — Deployment Update": (
            "Hi Team,\n\n"
            "The development phase is finished, but QA reported a few minor bugs "
            "in the login module. Because of this, we are pushing the deployment "
            "date back from Friday to next Wednesday.\n\n"
            "Please ensure all final code commits are pushed to the main branch "
            "by Monday at noon. Let me know if you have any questions!\n\nManagement"
        ),
        "📄 Safe Email — Robotics Competition": (
            "Dear Students,\n\n"
            "The e-Yantra Robotics Competition 2026-27 registration is now open.\n\n"
            "Students are encouraged to participate in this robotics competition "
            "organised by IIT Bombay.\n\n"
            "Registration Deadline: 25 August 2026.\n\n"
            "Please complete your registration before 5:00 PM on the deadline date.\n\n"
            "A meeting is scheduled on Monday at 10:30 AM.\n\n"
            "Visit the official website for more information.\n\n"
            "Regards,\nFaculty Coordinator"
        ),
        "🚫 Spam Email — Account Alert": (
            "Subject: URGENT - Your account has been limited\n\n"
            "Dear valued member,\n\n"
            "We've temporarily limited some features on your account. "
            "To restore full access, verify your identity immediately:\n"
            "https://account-verify.info/wallet/secure\n\n"
            "Act now! Your account will be suspended in 24 hours.\n\n"
            "Ref: id-JJC7505813\n"
            "Unsubscribe: https://track.co/unsubscribe"
        ),
        "🚫 Spam Email — MAMA EARTH SPAM": (
                   """CHECKOUT PAUSED

                            Flat 50% OFF is still applied

                            Just for 4 hours

                            Your order isn't confirmed yet

                            Confirm here: https://u1.mnge.co/MAMERT/A0Epj1n
                            -Mamaearth"""
                ),
    }

    selected = st.selectbox("🗂️ Try a sample:", list(sample_emails.keys()))
    default_text = sample_emails[selected]

    email_input = st.text_area(
        label="email_input",
        value=default_text,
        height=290,
        placeholder=(
            "Paste the full email here — subject, body, dates, deadlines…"
        ),
        label_visibility="collapsed",
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        analyze_btn = st.button(
            "🔍 Analyze Email", type="primary", use_container_width=True
        )
    with btn_col2:
        clear_btn = st.button(
            "🗑️ Clear / Reset", type="secondary", use_container_width=True
        )

    if clear_btn:
        st.session_state.result = None
        st.rerun()

    # How-it-works info box
    st.markdown("""
    <div class="info-box">
        <strong>Pipeline (from RR2.ipynb):</strong>
        <ul>
            <li>TF-IDF vectorizer + Logistic Regression — threshold set by slider above</li>
            <li>Safe emails → BART (<code>facebook/bart-large-cnn</code>) summarization</li>
            <li>Dates, times &amp; reminders extracted with Python <code>re</code> library</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  RIGHT COLUMN — Results
# ══════════════════════════════════════════════
with col_right:
    st.markdown("### 📊 Analysis Results")

    # Run analysis when button clicked
    if analyze_btn:
        if not email_input or not email_input.strip():
            st.warning("⚠️ Please paste some email content before clicking Analyze.")
        else:
            with st.spinner("🧪 Analyzing email…"):
                try:
                    result = process_email(
                        raw_text=email_input,
                        tfidf=tfidf,
                        log_reg=log_reg,
                        tokenizer=tokenizer,
                        bart_model=bart_model,
                        device=device,
                        threshold=threshold,
                    )
                    st.session_state.result = result
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    st.session_state.result = None

    result = st.session_state.get("result", None)

    # ── Empty state ────────────────────────────────────────────
    if result is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📬</div>
            <div class="empty-title">No email analyzed yet</div>
            <div class="empty-sub">
                Paste an email on the left and click <strong>Analyze Email</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────
    else:
        is_spam = result["status"] == "SPAM"
        prob    = result["spam_probability"]

        # ── Block 1: Spam / Safe verdict ──
        if is_spam:
            st.markdown(f"""
            <div class="card card-spam">
                <div class="card-title">🛡️ Spam Detection Result</div>
                <div class="badge-spam">🚫 &nbsp;SPAM DETECTED</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card card-safe">
                <div class="card-title">🛡️ Spam Detection Result</div>
                <div class="badge-safe">✅ &nbsp;SAFE EMAIL (HAM)</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability bar ──
        used_threshold = result.get("threshold_used", threshold_pct)
        st.markdown(
            f'<div class="prob-label">'
            f'Spam Probability: <strong>{prob}%</strong>'
            f'&nbsp;&nbsp;(Threshold used: <strong>{used_threshold}%</strong>)'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Draw both the spam probability AND the threshold marker
        st.progress(min(prob / 100, 1.0))

        # ── Spam branch: stop here ──
        if is_spam:
            st.error(
                f"🚫 **Email blocked.** Spam probability ({prob}%) exceeds the "
                f"{used_threshold}% threshold.  \n"
                "Summarization and date/time extraction are **skipped** for spam emails."
            )

        # ── Safe branch: show summary + reminders ──
        else:
            st.success(
                f"✅ Email is **safe** (spam probability {prob}% < threshold {used_threshold}%). "
                "Running summarization and date/time extraction…"
            )

            # ── Block 2: Summary ──
            summary = result.get("summary") or ""
            is_short = result.get("is_short", False)

            st.markdown("""
            <div class="card card-summary">
                <div class="card-title">📝 Email Summary (BART)</div>
            """, unsafe_allow_html=True)

            if is_short:
                st.info(
                    "ℹ️ Email is very short (< 20 words). "
                    "BART bypassed — original text shown."
                )

            st.markdown(
                f'<div class="summary-text">{summary if summary else "<i>No summary generated.</i>"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Block 3: Date / Time / Reminders ──
            reminders = result.get("reminders", [])

            st.markdown("""
            <div class="card card-reminder">
                <div class="card-title">📅 Important Dates &amp; Reminders (re library)</div>
            """, unsafe_allow_html=True)

            if reminders:
                for i, rem in enumerate(reminders, 1):
                    date_str = ", ".join(rem["date"]) if rem["date"] else "Not mentioned"
                    day_str  = ", ".join(rem["day"])  if rem["day"]  else "Not mentioned"
                    time_str = ", ".join(rem["time"]) if rem["time"] else "Not mentioned"

                    st.markdown(f"""
                    <div class="rem-item">
                        <div class="rem-title">#{i} &nbsp; {rem['details']}</div>
                        <div class="rem-meta">
                            📆 <span><strong>Date:</strong> {date_str}</span>
                            &nbsp;|&nbsp;
                            📅 <span><strong>Day:</strong> {day_str}</span>
                            &nbsp;|&nbsp;
                            🕒 <span><strong>Time:</strong> {time_str}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="color:#6b7280; font-size:0.9rem;">'
                    "No important dates, deadlines, or reminders found in this email."
                    "</p>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR — Project Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📘 Project Info")
    st.markdown("""
**Smart Email Assistant**
*Using ML and NLP*

---
**5th Semester Minor Project**

---
**Tech Stack**

| Component | Technology |
|-----------|-----------|
| Spam Filter | TF-IDF + Logistic Regression |
| Threshold | 80% (strict) |
| Summarizer | BART (facebook/bart-large-cnn) |
| Date/Time | Python `re` module |
| Dataset | 400,000 emails |
| Frontend | Streamlit |

---
**Model Performance**

| Metric | Score |
|--------|-------|
| Accuracy | 94.57% |
| Precision | 94.35% |
| Recall | 94.92% |
| F1-Score | 94.63% |

---
**Pipeline Steps**
1. 📨 Receive raw email text
2. 🛡️ Spam detection (LR + TF-IDF)
3. ✍️ Summarization (BART)
4. 📅 Date/time extraction (`re`)
5. 📊 Display results
    """)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Smart Email Assistant — 5th Semester Minor Project &nbsp;|&nbsp;
    ML + NLP &nbsp;|&nbsp; TF-IDF · Logistic Regression · BART · Streamlit
</div>
""", unsafe_allow_html=True)
