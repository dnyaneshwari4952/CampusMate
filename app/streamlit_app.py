import os
import sys
from pathlib import Path
import streamlit as st

# Add project root directory to sys.path so we can import src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.chatbot import Chatbot
from src.config import CONFIDENCE_THRESHOLD, AMBIGUITY_MARGIN

# Define logo path
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"

# 1. Page Configuration & Custom CSS Styles
st.set_page_config(
    page_title="CampusMate - University FAQ & Intent Assistant",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode styling with linear gradients and modern typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global font overwrite */
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean transparent header */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Hide unnecessary Streamlit default chrome (menu, deploy button, decorations, footer) */
    #MainMenu, [data-testid="stMainMenu"] { display: none !important; }
    footer, [data-testid="stFooter"] { display: none !important; }
    .stDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }

    /* Keep Sidebar controls visible and styled nicely */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button {
        background-color: rgba(30, 41, 59, 0.85) !important;
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
        border-radius: 8px !important;
        color: #818cf8 !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: rgba(99, 102, 241, 0.3) !important;
        border-color: #818cf8 !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #818cf8 !important;
        stroke: #818cf8 !important;
        color: #818cf8 !important;
    }

    [data-testid="stSidebarCollapsedControl"] button:hover svg,
    [data-testid="stSidebarCollapseButton"] button:hover svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Background gradients */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Streamlit Chat input styling */
    .stChatInputContainer {
        border-radius: 14px;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        background-color: #0f172a !important;
    }
    
    /* Custom Sidebar design */
    div[data-testid="stSidebar"] {
        background-color: #080c14 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Glassmorphic cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    
    .metric-box b {
        color: #818cf8;
    }
    
    /* Title gradients */
    h1 {
        background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #e2e8f0;
        font-weight: 700 !important;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Cached Chatbot Loader
@st.cache_resource
def load_chatbot_engine() -> Chatbot:
    """
    Caches chatbot initialization across sessions so model files
    are loaded exactly once at startup.
    """
    return Chatbot()

# 3. Model Loading
try:
    chatbot = load_chatbot_engine()
except Exception as e:
    st.error(f"Failed to load CampusMate intent classifier: {e}")
    st.info("Please run the training pipeline first to build model parameters:")
    st.code("python -m src.train")
    st.stop()

# 4. Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I am **CampusMate**, your University FAQ & Intent Assistant. I can help answer your questions about courses, fees, admissions, hostel accommodations, library hours, placements, and campus amenities. What would you like to know?",
            "intent": "greeting",
            "confidence": 1.0
        }
    ]

# 5. Sidebar Layout: CampusMate Console
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.image("https://img.icons8.com/color/96/graduation-cap.png", width=64)
    st.title("CampusMate Console")
    st.caption("University FAQ & Intent Assistant")
    st.markdown('<div class="status-badge">● NLP Engine Active</div>', unsafe_allow_html=True)
    st.markdown("---")
    debug_mode = False

    # Collapsible FAQ Intents Directory
    st.subheader("📂 Supported Categories")
    classes = chatbot.metadata.get("classes", []) if chatbot.metadata else list(chatbot.classifier.classes_)
    with st.expander(f"View Intent Catalog ({len(classes)})"):
        for tag in sorted(classes):
            st.markdown(f"- `{tag}`")

    st.markdown("---")
    
    # Reset Conversation Button
    if st.button("🗑️ Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I am **CampusMate**, your University FAQ & Intent Assistant. I can help answer your questions about courses, fees, admissions, hostel accommodations, library hours, placements, and campus amenities. What would you like to know?",
                "intent": "greeting",
                "confidence": 1.0
            }
        ]
        chatbot.reset_context()
        st.rerun()

# 6. Main Header
st.title("CampusMate")
st.markdown("#### University FAQ & Intent Assistant")
st.caption("A supervised classical NLP intent classification engine with rule-based response mapping and out-of-domain protection.")

# 7. Render Messages from Session State
for msg in st.session_state.messages:
    avatar = str(LOGO_PATH) if msg["role"] == "assistant" and LOGO_PATH.exists() else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # Overlay debug data if active
        if debug_mode and "debug_info" in msg:
            st.json(msg["debug_info"])
            if msg["role"] == "assistant" and "intent" in msg and msg["intent"] != "fallback":
                st.caption(f"Matched Intent: **`{msg['intent']}`** | Confidence: **{msg['confidence']*100:.1f}%**")

# 8. User Interaction Chat Box
if user_query := st.chat_input("Ask a university question (e.g., What courses are offered? What are the library timings?)..."):
    
    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Fetch chatbot response packet
    response_packet = chatbot.respond(user_query)
    bot_reply = response_packet["response"]
    pred_intent = response_packet["intent"]
    confidence = response_packet["confidence"]
    debug_info = response_packet["debug_info"]

    # Append response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "intent": pred_intent,
        "confidence": confidence,
        "debug_info": debug_info
    })

    # Render bot response bubble
    bot_avatar = str(LOGO_PATH) if LOGO_PATH.exists() else None
    with st.chat_message("assistant", avatar=bot_avatar):
        st.markdown(bot_reply)
        
        # Display debug info / confidence badge if enabled
        if debug_mode:
            st.json(debug_info)
            if pred_intent != "fallback":
                st.caption(f"Matched Intent: **`{pred_intent}`** | Confidence: **{confidence*100:.1f}%**")
