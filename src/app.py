import streamlit as st
import time
import os
import base64
import numpy as np

# --- Database Initialization ---
from database import init_db
init_db()

from auth import login_user, register_user
from tensorflow.keras.models import load_model
from lime.lime_text import LimeTextExplainer

from config import MODEL_PATH, TOKENIZER_PATH, MODEL_V2_PATH, TOKENIZER_V2_PATH, MAX_LEN
from preprocess import load_tokenizer
from predict import predict_text, DISCLAIMER
from explain import predict_proba as predict_proba_for_lime, class_names
from shap_explain import get_shap_explanation

# -----------------------------------------------------------------------------
# Page Configuration & Global Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewsGuard AI — Advanced Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS based on Reference Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #0c1017;
        --bg-sidebar: #0a0e14;
        --card-bg: rgba(22, 29, 41, 0.65);
        --card-border: rgba(255, 255, 255, 0.08);
        --text-ivory: #f7f5f0;
        --text-gold: #c5a880;
        --text-blue: #7ba7cc;
        --text-muted: #a6a29a;
        --text-dark: #1e293b;
        --accent-blue: #2e6c80;
        --accent-hover: #3d88a0;
    }

    /* Overall App Background — base colour only.
       The real dashboard background image is injected dynamically
       via inject_dashboard_bg_css() so it only applies when authenticated. */
    .stApp {
        background-color: var(--bg-main);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-ivory);
    }

    /* ================================================================
       SIDEBAR — modern product nav panel
    /* ================================================================
       SIDEBAR REFINEMENTS & NAVIGATION
       ================================================================ */
    [data-testid="stSidebar"] {
        background: rgba(7, 11, 22, 0.92) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
    }

    /* Brand block */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 0 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1.2rem;
    }

    .brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #1a3870 0%, #2358c8 100%);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 3px 12px rgba(35, 88, 200, 0.30);
        flex-shrink: 0;
    }

    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-ivory);
        line-height: 1.15;
    }

    .brand-title span { color: var(--text-blue); }

    .brand-subtitle {
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* User profile chip */
    .sidebar-user {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0.75rem;
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 9px;
        margin-bottom: 1.4rem;
    }

    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(46,108,128,0.6), rgba(35,88,200,0.4));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        color: var(--text-ivory);
        border: 1px solid rgba(123, 167, 204, 0.25);
        flex-shrink: 0;
    }

    .user-meta-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        line-height: 1.2;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .user-meta-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-ivory);
    }

    /* Nav section label */
    .nav-section-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(166, 162, 154, 0.6);
        padding: 0 0.25rem;
        margin-bottom: 0.6rem;
    }

    /* Sidebar Navigation Menu Items Override */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        color: #94a3b8 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        padding: 0.48rem 0.75rem !important;
        border-radius: 8px !important;
        margin: 2px 0 !important;
        min-height: unset !important;
        letter-spacing: 0.01em !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #f1f5f9 !important;
        border-color: transparent !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Active Sidebar Item */
    [data-testid="stSidebar"] .sidebar-nav-active .stButton > button {
        background: rgba(56, 114, 224, 0.12) !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
        border-left: 3px solid #60a5fa !important;
        border-radius: 0 8px 8px 0 !important;
        padding-left: 0.65rem !important;
    }

    [data-testid="stSidebar"] .sidebar-nav-active .stButton > button:hover {
        background: rgba(56, 114, 224, 0.18) !important;
        color: #93c5fd !important;
    }

    /* Sidebar Logout Item */
    [data-testid="stSidebar"] .sidebar-logout-item .stButton > button {
        background: transparent !important;
        border: 1px solid rgba(239, 68, 68, 0.18) !important;
        color: #f87171 !important;
        font-size: 0.84rem !important;
        padding: 0.45rem 0.75rem !important;
        margin-top: 0.3rem !important;
    }

    [data-testid="stSidebar"] .sidebar-logout-item .stButton > button:hover {
        background: rgba(239, 68, 68, 0.10) !important;
        border-color: rgba(239, 68, 68, 0.35) !important;
        color: #fca5a5 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Divider */
    .nav-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.06);
        margin: 1.2rem 0;
    }

    /* ================================================================
       TYPOGRAPHY & HERO
       ================================================================ */
    .hero-tag {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: var(--text-gold);
        margin-bottom: 0.5rem;
    }

    .hero-heading {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.4rem;
        font-weight: 600;
        color: var(--text-ivory);
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }

    .hero-heading .user-name {
        color: var(--text-blue);
        font-style: normal;
    }

    .hero-subheading {
        font-size: 0.95rem;
        color: var(--text-muted);
        line-height: 1.6;
        margin-top: 0.5rem;
        max-width: 600px;
    }

    /* Dashboard slogan pill */
    .slogan-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(46, 108, 128, 0.15);
        border: 1px solid rgba(123, 167, 204, 0.2);
        border-radius: 100px;
        padding: 0.35rem 0.95rem;
        font-family: 'Outfit', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--text-blue);
        letter-spacing: 0.03em;
        margin-bottom: 0.8rem;
    }

    /* ================================================================
       WORKFLOW STEP CARDS
       ================================================================ */
    .workflow-card {
        background: rgba(18, 26, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem 1.2rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        min-height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .workflow-card:hover {
        border-color: rgba(123, 167, 204, 0.3);
        transform: translateY(-2px);
    }

    .workflow-step-num {
        font-family: 'Outfit', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 0.65rem;
    }

    .step-num-1 { color: #60a5fa; }
    .step-num-2 { color: #2ec4b6; }
    .step-num-3 { color: #ebb35a; }

    .workflow-step-text {
        font-size: 0.86rem;
        color: var(--text-muted);
        line-height: 1.55;
    }

    .workflow-step-text strong {
        color: var(--text-ivory);
    }

    .workflow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: rgba(123, 167, 204, 0.45);
        font-size: 1.35rem;
        padding-top: 2.8rem;
    }

    /* CTA Primary Button Wrap */
    .cta-primary-wrap .stButton > button {
        background: linear-gradient(135deg, #1e5068 0%, #2e6c80 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(123, 167, 204, 0.35) !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.8rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 18px rgba(46, 108, 128, 0.30) !important;
        transition: all 0.2s ease !important;
        justify-content: center !important;
        text-align: center !important;
    }

    .cta-primary-wrap .stButton > button:hover {
        background: linear-gradient(135deg, #285e70 0%, #3a829a 100%) !important;
        box-shadow: 0 6px 24px rgba(46, 108, 128, 0.45) !important;
        border-color: rgba(123, 167, 204, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ================================================================
       GLOBAL BUTTONS & FORM CONTROLS
       ================================================================ */
    .stButton>button {
        background: linear-gradient(135deg, #1e4d5c 0%, #2e6c80 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 0.58rem 1.3rem !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.22) !important;
        transition: all 0.22s ease !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #285e70 0%, #3a829a 100%) !important;
        border-color: rgba(123, 167, 204, 0.38) !important;
        box-shadow: 0 5px 18px rgba(46, 108, 128, 0.32) !important;
    }

    /* Text inputs & Textareas */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(14, 20, 32, 0.75) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: var(--text-blue) !important;
        box-shadow: 0 0 0 2px rgba(123, 167, 204, 0.22) !important;
    }

    /* Glass Container (login card + about) */
    .glass-container {
        background: rgba(18, 26, 42, 0.70);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 2rem;
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
    }

    /* Prediction Result Cards */
    .result-badge-card {
        padding: 1.5rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        border: 1px solid;
    }

    .badge-real    { background: rgba(33, 195, 84, 0.08);  border-color: rgba(33, 195, 84, 0.28); }
    .badge-fake    { background: rgba(239, 68, 68, 0.08);  border-color: rgba(239, 68, 68, 0.28); }
    .badge-uncertain { background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.28); }

    .status-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .text-real     { color: #4ade80; }
    .text-fake     { color: #f87171; }
    .text-uncertain { color: #fbbf24; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# App State
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"

# -----------------------------------------------------------------------------
# Background image helpers
# -----------------------------------------------------------------------------
_LOGIN_BG_PATH = os.path.join(os.path.dirname(__file__), "static", "login_bg.jpg")
_DASH_BG_PATH  = os.path.join(os.path.dirname(__file__), "static", "dashboard_bg_v2.png")


def _get_login_bg_b64() -> str:
    """Return a base64-encoded data URI for the login background image."""
    try:
        with open(_LOGIN_BG_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"
    except FileNotFoundError:
        return ""


def inject_login_bg_css():
    """Inject full-page background CSS for unauthenticated (login/register) pages.

    The background image is served as a base64 data URI so no static-file
    server is needed.  A semi-transparent dark overlay keeps the Streamlit
    login card readable.  This CSS is scoped to .login-page-bg and is NOT
    applied to the authenticated dashboard.
    """
    bg_uri = _get_login_bg_b64()
    if not bg_uri:
        # Fallback: rich dark gradient when the image file is missing
        bg_css = """
            background: linear-gradient(
                135deg,
                #060d1a 0%,
                #0a1628 40%,
                #0d1f38 70%,
                #071220 100%
            );
        """
    else:
        bg_css = f"""
            background-image:
                linear-gradient(
                    to bottom,
                    rgba(4, 8, 18, 0.72) 0%,
                    rgba(6, 12, 26, 0.78) 50%,
                    rgba(4, 8, 18, 0.82) 100%
                ),
                url("{bg_uri}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        """

    st.markdown(f"""
    <style>
        /* ---- Login / Register page full-page background ---- */
        [data-testid="stAppViewContainer"] {{
            {bg_css}
        }}
        /* Hide default Streamlit top decoration on auth pages */
        [data-testid="stDecoration"] {{
            display: none;
        }}
        /* Force main block to be transparent so bg shows through */
        [data-testid="stAppViewContainer"] > .main {{
            background: transparent !important;
        }}
        /* Ensure header bar is translucent */
        [data-testid="stHeader"] {{
            background: rgba(4, 8, 18, 0.55) !important;
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        /* Hide sidebar completely on login / register */
        [data-testid="stSidebar"] {{
            display: none !important;
        }}
        /* Collapse sidebar toggle button */
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}
    </style>
    """, unsafe_allow_html=True)



def _get_dashboard_bg_b64() -> str:
    """Return a base64-encoded data URI for the authenticated dashboard background."""
    try:
        with open(_DASH_BG_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    except FileNotFoundError:
        return ""


def inject_dashboard_bg_css():
    """Inject the abstract dark-navy background for the authenticated dashboard.

    Strategy
    --------
    * The image is served as a base64 data URI — no HTTP request, no CORS issue.
    * A semi-transparent dark overlay (≈40 % opacity) is composited *on top* of
      the image so that all cards, text, and sidebar remain crisply readable.
    * background-attachment: fixed keeps the image pinned while the page scrolls,
      avoiding any awkward tiling or reflow.
    * A graceful fallback gradient is used when the PNG file is missing.
    """
    bg_uri = _get_dashboard_bg_b64()
    if not bg_uri:
        bg_css = """
            background-color: #0c1017;
            background-image:
                radial-gradient(circle at 80% 20%, rgba(46, 108, 128, 0.12) 0%, transparent 50%),
                radial-gradient(circle at 15% 85%, rgba(10, 20, 50, 0.18) 0%, transparent 55%);
        """
    else:
        bg_css = f"""
            background-image:
                linear-gradient(
                    180deg,
                    rgba(6, 10, 20, 0.42) 0%,
                    rgba(8, 13, 26, 0.38) 50%,
                    rgba(6, 10, 20, 0.45) 100%
                ),
                url("{bg_uri}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        """

    st.markdown(f"""
    <style>
        /* ---- Authenticated dashboard full-page background ---- */
        .stApp {{
            {bg_css}
        }}
        /* Make the default stApp inner wrapper transparent so bg shows */
        [data-testid="stAppViewContainer"] > .main {{
            background: transparent !important;
        }}
        /* Translucent toolbar */
        [data-testid="stHeader"] {{
            background: rgba(6, 10, 20, 0.50) !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        /* Frosted-glass sidebar — slightly darker than the bg */
        [data-testid="stSidebar"] {{
            background: rgba(6, 10, 22, 0.82) !important;
            backdrop-filter: blur(28px) !important;
            -webkit-backdrop-filter: blur(28px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }}
        /* Hide stDecoration banner */
        [data-testid="stDecoration"] {{
            display: none;
        }}
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached resource loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_hybrid_model():
    target_path = MODEL_V2_PATH if os.path.exists(MODEL_V2_PATH) else MODEL_PATH
    return load_model(target_path)

@st.cache_resource
def load_tokenizer_cached():
    target_path = TOKENIZER_V2_PATH if os.path.exists(TOKENIZER_V2_PATH) else TOKENIZER_PATH
    return load_tokenizer(target_path)

# -----------------------------------------------------------------------------
# Authentication UI
# -----------------------------------------------------------------------------
def render_login():
    # Inject the full-page login background (image + dark overlay)
    inject_login_bg_css()
    col1, col2, col3 = st.columns([1, 1.8, 1])
    
    with col2:
        st.markdown(
            '<div class="glass-container" style="text-align: center; margin-top: 3rem;">\n'
            '<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛡️</div>\n'
            '<div class="hero-tag">NEWSGUARD AI</div>\n'
            '<h2 style="font-family: \'Playfair Display\', serif; color: var(--text-ivory); margin-top: 0;">Welcome to NewsGuard</h2>\n'
            '<p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 2rem;">\n'
            'Advanced Explainable AI for Fake News Detection\n'
            '</p>\n'
            '</div>',
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            st.write("#### Sign In")
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Sign In to Dashboard", use_container_width=True)
            
            if submit:
                ok, msg, user = login_user(username, password)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.user_info = user
                    st.session_state.current_page = "dashboard"
                    st.success("Login successful!")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error(msg)
                    
        st.write("")
        col_reg1, col_reg2 = st.columns([2, 1])
        with col_reg1:
            st.markdown("<p style='color: var(--text-muted); font-size: 0.9rem; padding-top: 6px;'>Don't have an account yet?</p>", unsafe_allow_html=True)
        with col_reg2:
            if st.button("Register Here", use_container_width=True):
                st.session_state.current_page = "register"
                st.rerun()

def render_register():
    # Inject the full-page login background (image + dark overlay)
    inject_login_bg_css()
    col1, col2, col3 = st.columns([1, 1.8, 1])
    
    with col2:
        st.markdown(
            '<div class="glass-container" style="text-align: center; margin-top: 2rem;">\n'
            '<div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🛡️</div>\n'
            '<div class="hero-tag">NEWSGUARD AI</div>\n'
            '<h2 style="font-family: \'Playfair Display\', serif; color: var(--text-ivory); margin-top: 0;">Create Your Account</h2>\n'
            '<p style="color: var(--text-muted); font-size: 0.9rem;">\n'
            'Join our research platform for intelligent news verification\n'
            '</p>\n'
            '</div>',
            unsafe_allow_html=True
        )
        
        with st.form("register_form"):
            name = st.text_input("Full Name", placeholder="e.g. Dr. Jane Smith")
            username = st.text_input("Username", placeholder="e.g. janesmith")
            email = st.text_input("Email Address", placeholder="e.g. jane@example.com")
            password = st.text_input("Password", type="password", placeholder="Minimum 8 characters")
            confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                ok, msg = register_user(name, username, email, password, confirm)
                if ok:
                    st.success(msg)
                    time.sleep(1.2)
                    st.session_state.current_page = "login"
                    st.rerun()
                else:
                    st.error(msg)
                    
        if st.button("← Back to Sign In", use_container_width=True):
            st.session_state.current_page = "login"
            st.rerun()

# -----------------------------------------------------------------------------
# Main Dashboard UI Components
# -----------------------------------------------------------------------------
def _render_unsupported(reason: str):
    st.markdown(
        f'<div class="result-badge-card badge-uncertain">\n'
        f'<div class="status-title text-uncertain">⚠️ Unsupported Input</div>\n'
        f'<p style="color: var(--text-ivory); margin-bottom: 0.5rem;">{reason}</p>\n'
        f'<p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">Please provide a complete news claim, headline, or article for deep learning analysis.</p>\n'
        f'</div>',
        unsafe_allow_html=True
    )

def _render_prediction_result(result: dict):
    label = result["label"]
    prob_fake = result["prob_fake"]
    prob_real = result["prob_real"]
    confidence = result["confidence"]
    
    if label == "FAKE":
        badge_class = "badge-fake"
        text_class = "text-fake"
        icon = "🚨"
        label_display = "FAKE NEWS DETECTED"
    elif label == "REAL":
        badge_class = "badge-real"
        text_class = "text-real"
        icon = "✅"
        label_display = "VERIFIED REAL NEWS"
    else:
        badge_class = "badge-uncertain"
        text_class = "text-uncertain"
        icon = "⚠️"
        label_display = "UNCERTAIN / AMBIGUOUS"
        
    st.markdown(
        f'<div class="result-badge-card {badge_class}">\n'
        f'<div class="status-title {text_class}">{icon} {label_display}</div>\n'
        f'<div style="font-size: 1.1rem; color: var(--text-ivory); margin-top: 0.4rem;">\n'
        f'Model Confidence: <strong style="color: #ffffff;">{confidence * 100:.2f}%</strong>\n'
        f'</div>\n'
        f'</div>',
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("REAL Probability", f"{prob_real * 100:.2f}%")
    with col2:
        st.metric("FAKE Probability", f"{prob_fake * 100:.2f}%")
        
    st.markdown(
        f'<div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.75rem; line-height: 1.4;">\n'
        f'ℹ️ <em>{DISCLAIMER}</em>\n'
        f'</div>',
        unsafe_allow_html=True
    )

def render_dashboard_view(user_display_name: str):
    """Dashboard home: hero + 3-step workflow cards + single CTA."""

    # ── Hero ────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="padding: 1.2rem 0 0.4rem 0;">\n'
        f'<div class="hero-tag">NEWSGUARD AI — VERIFICATION PLATFORM</div>\n'
        f'<div class="hero-heading">Welcome back,&nbsp;<span class="user-name">{user_display_name}</span></div>\n'
        f'<div class="slogan-pill">✦ Verify the story. Understand the result.</div>\n'
        f'<div class="hero-subheading">Analyze news articles with deep learning, evaluate confidence scores, and inspect interpretability explanations.</div>\n'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── 3-Step workflow columns ──────────────────────────────────────
    col1, arr1, col2, arr2, col3 = st.columns([1, 0.08, 1, 0.08, 1])

    with col1:
        st.markdown(
            '<div class="workflow-card">\n'
            '<div class="workflow-step-num step-num-1">01 — Detect</div>\n'
            '<div class="workflow-step-text">Analyze a news article and receive a <strong>REAL</strong> / <strong>FAKE</strong> / <strong>UNCERTAIN</strong> prediction.</div>\n'
            '</div>',
            unsafe_allow_html=True
        )

    with arr1:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="workflow-card">\n'
            '<div class="workflow-step-num step-num-2">02 — Understand</div>\n'
            '<div class="workflow-step-text">View <strong>REAL probability</strong>, <strong>FAKE probability</strong> and model confidence.</div>\n'
            '</div>',
            unsafe_allow_html=True
        )

    with arr2:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)

    with col3:
        st.markdown(
            '<div class="workflow-card">\n'
            '<div class="workflow-step-num step-num-3">03 — Explain</div>\n'
            '<div class="workflow-step-text">Understand which words influenced the prediction using <strong>LIME</strong>.</div>\n'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Single primary CTA ───────────────────────────────────────────
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([1, 1.4, 1])
    with cta_col:
        st.markdown('<div class="cta-primary-wrap">', unsafe_allow_html=True)
        if st.button("Analyze a News Article →", key="btn_cta_analyze", use_container_width=True):
            st.session_state.nav_target = "Detect News"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def render_detect_news_view():
    st.markdown(
        '<div style="margin-bottom: 1.5rem;">\n'
        '<div class="hero-tag">INFERENCE & EXPLAINABILITY</div>\n'
        '<div class="hero-heading" style="font-size: 2.2rem;">Detect News</div>\n'
        '<p style="color: var(--text-muted); font-size: 1rem; margin-top: 0.25rem;">\n'
        'Analyze a news article and understand the model\'s prediction with transparent XAI insights.\n'
        '</p>\n'
        '</div>',
        unsafe_allow_html=True
    )
    
    with st.spinner("Loading hybrid model & tokenizer..."):
        model = load_hybrid_model()
        tokenizer = load_tokenizer_cached()
        
    user_text = st.text_area(
        "Enter News Article or Headline",
        height=200,
        placeholder="Paste your news article, claim, or headline here..."
    )
    
    if st.button("Analyze News", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
            # Clear any previous result so stale output is not shown
            st.session_state.pop("detect_result", None)
            st.session_state.pop("detect_text", None)
        else:
            with st.spinner("Analyzing text with NewsGuard AI pipeline..."):
                result = predict_text(user_text, model=model, tokenizer=tokenizer)
            # Persist result + article text so the XAI section
            # survives checkbox re-runs (button is not re-pressed on checkbox click)
            st.session_state["detect_result"] = result
            st.session_state["detect_text"] = user_text

    # ── Results + XAI section ─────────────────────────────────────────────────
    # Rendered unconditionally whenever session state holds a valid result.
    # This survives Streamlit re-runs triggered by checkbox interactions.
    if "detect_result" in st.session_state and "detect_text" in st.session_state:
        result = st.session_state["detect_result"]
        saved_text = st.session_state["detect_text"]

        if result["status"] == "UNSUPPORTED":
            _render_unsupported(result["reason"])
        else:
            _render_prediction_result(result)

            # ── LIME ─────────────────────────────────────────────────────────
            st.write("")
            st.markdown(
                '<div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">\n'
                '<div class="hero-tag">EXPLAINABLE AI (LIME)</div>\n'
                '<h4 style="font-family: \'Playfair Display\', serif; color: var(--text-ivory); margin: 0;">\n'
                'Why did the model make this prediction?\n'
                '</h4>\n'
                '</div>',
                unsafe_allow_html=True
            )

            explain = st.checkbox("Generate word-level explanation (LIME)", value=True, key="chk_lime")

            if explain:
                with st.spinner("Calculating local feature attributions via LIME..."):
                    lime_explainer = LimeTextExplainer(class_names=class_names)
                    exp = lime_explainer.explain_instance(
                        text_instance=saved_text,
                        classifier_fn=lambda x: predict_proba_for_lime(x, model, tokenizer),
                        num_features=10
                    )

                st.caption("Top contributing words (Positive weight → FAKE evidence, Negative weight → REAL evidence):")
                features, weights = zip(*exp.as_list())
                st.table({
                    "Word / Token": features,
                    "Attribution Weight": [f"{w:+.4f}" for w in weights],
                    "Impact Direction": ["Evidence for FAKE" if w > 0 else "Evidence for REAL" for w in weights]
                })

            # ── SHAP ─────────────────────────────────────────────────────────
            st.write("")
            st.markdown(
                '<div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">\n'
                '<div class="hero-tag">EXPLAINABLE AI (SHAP)</div>\n'
                '<h4 style="font-family: \'Playfair Display\', serif; color: var(--text-ivory); margin: 0;">\n'
                'Shapley Additive Explanations (SHAP)\n'
                '</h4>\n'
                '</div>',
                unsafe_allow_html=True
            )

            explain_shap = st.checkbox("Generate SHAP explanation", value=False, key="chk_shap")

            if explain_shap:
                with st.spinner("Calculating SHAP feature attributions (this may take ~5–15 seconds)..."):
                    try:
                        shap_result = get_shap_explanation(saved_text, model, tokenizer, top_k=10)
                    except Exception as shap_exc:
                        st.error(f"SHAP calculation error: {shap_exc}")
                        shap_result = None

                if shap_result is None:
                    pass  # Error message already displayed above
                elif shap_result == "UNSUPPORTED":
                    st.warning("SHAP explanation is unavailable for unsupported inputs.")
                elif not shap_result:
                    st.info("No influential words detected for SHAP explanation.")
                else:
                    st.caption("Top SHAP features (Positive weight → FAKE evidence, Negative weight → REAL evidence):")
                    s_features, s_weights = zip(*shap_result)
                    st.table({
                        "Word / Token": s_features,
                        "SHAP Attribution": [f"{w:+.4f}" for w in s_weights],
                        "Impact Direction": ["Evidence for FAKE" if w > 0 else "Evidence for REAL" for w in s_weights]
                    })

def render_history_view():
    st.markdown(
        '<div style="margin-bottom: 1.5rem;">\n'
        '<div class="hero-tag">ACTIVITY ARCHIVE</div>\n'
        '<div class="hero-heading" style="font-size: 2.2rem;">Prediction History</div>\n'
        '<p style="color: var(--text-muted); font-size: 1rem; margin-top: 0.25rem;">\n'
        'User prediction logs and audit history will be available here in upcoming updates.\n'
        '</p>\n'
        '</div>',
        unsafe_allow_html=True
    )
    st.info("Prediction history logging is slated for integration in a future release.")

def render_about_view():
    st.markdown(
        '<div style="margin-bottom: 1.5rem;">\n'
        '<div class="hero-tag">SYSTEM OVERVIEW</div>\n'
        '<div class="hero-heading" style="font-size: 2.2rem;">About NewsGuard AI</div>\n'
        '<p style="color: var(--text-muted); font-size: 1rem; margin-top: 0.25rem;">\n'
        'Advanced Explainable Fake News Detection Architecture\n'
        '</p>\n'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<div class="glass-container">\n'
        '<h4 style="font-family: \'Outfit\', sans-serif; color: var(--text-gold); margin-top: 0;">Project Architecture</h4>\n'
        '<p style="color: var(--text-ivory); line-height: 1.6;">\n'
        '<strong>NewsGuard AI</strong> is an advanced capstone application combining deep learning NLP with Explainable AI (XAI) to classify news credibility with high accuracy and transparent decision reasoning.\n'
        '</p>\n'
        '<ul style="color: var(--text-muted); line-height: 1.8;">\n'
        '<li><strong>Conservative Input Validation:</strong> Blocks non-news text, conversational greetings, math expressions, and gibberish before neural inference.</li>\n'
        '<li><strong>Canonical Preprocessing:</strong> Lowercasing, URL/punctuation/digit filtering, stopword removal, and WordNet lemmatization.</li>\n'
        '<li><strong>Hybrid CNN + BiLSTM:</strong> Dual-branch neural architecture extracting both local n-gram semantic patterns (CNN) and bidirectional sequential dependencies (BiLSTM).</li>\n'
        '<li><strong>Clean Retrained v2 Model:</strong> Strict dataset hygiene with no cross-split duplicate leakage, fitted on training splits only.</li>\n'
        '<li><strong>Explainable AI (LIME + SHAP):</strong> Local Interpretable Model-Agnostic Explanations and Shapley Additive Explanations highlighting exact words driving the model\'s prediction.</li>\n'
        '<li><strong>Security Foundation:</strong> Secure SQLite storage with salted bcrypt password hashing (12 rounds).</li>\n'
        '</ul>\n'
        '</div>',
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def render_authenticated_app():
    # Apply the abstract dark-navy dashboard background (image + subtle overlay)
    inject_dashboard_bg_css()

    # If a programmatic navigation target was set, apply it before widgets render
    if "nav_target" in st.session_state:
        st.session_state.nav_page = st.session_state.pop("nav_target")
        
    user = st.session_state.user_info
    username = user.get("username", "User") if user else "User"
    user_name = user.get("name", username) if user else "User"

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:

        # Brand
        st.markdown(
            '<div class="sidebar-brand">\n'
            '<div class="brand-icon">🛡️</div>\n'
            '<div>\n'
            '<div class="brand-title">NewsGuard <span>AI</span></div>\n'
            '<div class="brand-subtitle">News Verification Platform</div>\n'
            '</div>\n'
            '</div>',
            unsafe_allow_html=True
        )

        # User chip
        display_initial = username[0].upper() if username else "U"
        st.markdown(
            f'<div class="sidebar-user">\n'
            f'<div class="user-avatar">{display_initial}</div>\n'
            f'<div>\n'
            f'<div class="user-meta-label">Signed in as</div>\n'
            f'<div class="user-meta-name">{username}</div>\n'
            f'</div>\n'
            f'</div>',
            unsafe_allow_html=True
        )

        # Ensure nav_page is valid
        nav_options = ["Dashboard", "Detect News", "History", "About"]
        if "nav_page" not in st.session_state or st.session_state.nav_page not in nav_options:
            st.session_state.nav_page = "Dashboard"
        current_page = st.session_state.nav_page

        # Navigation label
        st.markdown('<div class="nav-section-label">NAVIGATION</div>', unsafe_allow_html=True)

        # Nav items — clean bullet navigation list
        nav_items = [
            ("Dashboard",   "Dashboard"),
            ("Detect News", "Detect News"),
            ("History",     "History"),
            ("About",       "About"),
        ]

        for label, page_key in nav_items:
            is_active = (current_page == page_key)
            dot = "●" if is_active else "○"
            wrapper_class = "sidebar-nav-active" if is_active else "sidebar-nav-item"
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            if st.button(f"{dot}  {label}", key=f"nav_item_{page_key}", use_container_width=True):
                st.session_state.nav_page = page_key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Divider before logout
        st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

        # Logout
        st.markdown('<div class="sidebar-logout-item">', unsafe_allow_html=True)
        if st.button("↪  Logout", key="btn_sidebar_logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.current_page = "login"
            st.session_state.nav_page = "Dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Main content router ───────────────────────────────────────────
    if st.session_state.nav_page == "Dashboard":
        render_dashboard_view(username)
    elif st.session_state.nav_page == "Detect News":
        render_detect_news_view()
    elif st.session_state.nav_page == "History":
        render_history_view()
    elif st.session_state.nav_page == "About":
        render_about_view()

def main():
    if not st.session_state.authenticated:
        if st.session_state.current_page == "register":
            render_register()
        else:
            render_login()
    else:
        render_authenticated_app()

if __name__ == "__main__":
    main()
