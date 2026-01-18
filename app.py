import streamlit as st
import google.generativeai as genai
import os
import faiss
import numpy as np
from gtts import gTTS
from io import BytesIO
from sentence_transformers import SentenceTransformer
from streamlit_mic_recorder import mic_recorder
from PIL import Image

# -------------------------------------------------
# 1. PAGE CONFIG & PREMIUM NATURE THEME
# -------------------------------------------------
st.set_page_config(page_title="KrushiMitra AI", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    /* Subtle Nature-Modern Palette */
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#cbd5e1 0.5px, transparent 0.5px);
        background-size: 30px 30px;
    }
    
    p, span, label, .stMarkdown {
        color: #334155 !important;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        color: #15803d !important;
        font-weight: 800 !important;
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(241, 245, 249, 0.9);
        border-right: 1px solid #e2e8f0;
    }

    /* Elegant Chat Bubbles */
    .stChatMessage {
        background-color: #ffffff !important;
        border-radius: 18px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* Innovative Language Buttons */
    .lang-card {
        border: 1px solid #dcfce7;
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        transition: 0.3s;
        cursor: pointer;
    }
    .lang-card:hover {
        background-color: #f0fdf4;
        border-color: #22c55e;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# 2. CORE ENGINES (RAG & AI)
# -------------------------------------------------
DATA_DIR = "agri_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    with open(os.path.join(DATA_DIR, "guide.txt"), "w", encoding="utf-8") as f:
        f.write("Organic farming uses natural compost. Drip irrigation saves 50% water. Pests like aphids can be controlled with neem oil.")

# Model Setup
genai.configure(api_key="AIzaSyBPP-ECKjUEh0UHox9tsIzhUEfN57Eb8uo")
gemini_model = genai.GenerativeModel("gemini-3-flash-preview")

@st.cache_resource
def init_rag():
    docs = [open(os.path.join(DATA_DIR, f), "r", encoding="utf-8").read() for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeds = model.encode(docs)
    idx = faiss.IndexFlatL2(embeds.shape[1])
    idx.add(np.array(embeds).astype('float32'))
    return model, idx, docs

embed_model, faiss_idx, raw_docs = init_rag()

# Language Mapping
LANG_CONFIG = {
    "Telugu (తెలుగు)": {"code": "te", "name": "Telugu"},
    "Hindi (हिन्दी)": {"code": "hi", "name": "Hindi"},
    "Kannada (ಕನ್ನಡ)": {"code": "kn", "name": "Kannada"},
    "Tamil (தமிழ்)": {"code": "ta", "name": "Tamil"},
    "Malayalam (മലയാളം)": {"code": "ml", "name": "Malayalam"},
    "English": {"code": "en", "name": "English"}
}

# -------------------------------------------------
# 3. SAFE VOICE GENERATOR (Fixes gTTSError)
# -------------------------------------------------
def safe_speak(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp
    except Exception as e:
        st.warning("⚠️ Voice note could not be generated due to connection issues.")
        return None

# -------------------------------------------------
# 4. SIDEBAR & STATE
# -------------------------------------------------
if "messages" not in st.session_state: st.session_state.messages = []
if "pending_query" not in st.session_state: st.session_state.pending_query = None
if "pending_img" not in st.session_state: st.session_state.pending_img = None

with st.sidebar:
    st.markdown("## 🌱 KrushiMitra Pro")
    st.caption("Multilingual Agricultural Advisor")
    
    up_img = st.file_uploader("Upload leaf/soil photo", type=['jpg', 'jpeg', 'png'])
    if up_img: st.image(up_img, caption="Analysis Source")

    st.divider()
    if st.button("🗑️ Clear Consultation"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

# -------------------------------------------------
# 5. MAIN INTERFACE
# -------------------------------------------------
st.title("🌾 KrushiMitra")
st.markdown("##### Personalized farming advice in your native language.")

# Show previous history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- INPUT SECTION ---
if not st.session_state.pending_query:
    col_mic, col_txt = st.columns([0.07, 0.93])
    with col_mic:
        audio = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key="mic", just_once=True)
    
    user_input = st.chat_input("Ask about crops, soil, or pests...")

    if audio:
        with st.spinner("Transcribing..."):
            res = gemini_model.generate_content(["Transcribe this audio strictly. No fluff.", {"mime_type": "audio/wav", "data": audio['bytes']}])
            user_input = res.text

    if user_input:
        st.session_state.pending_query = user_input
        st.session_state.pending_img = up_img
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

# --- STEP 2: DYNAMIC LANGUAGE SELECTION ---
if st.session_state.pending_query:
    st.info("💡 Your question is ready. In which language would you like the answer?")
    
    cols = st.columns(3)
    for i, (label, cfg) in enumerate(LANG_CONFIG.items()):
        with cols[i % 3]:
            if st.button(label, use_container_width=True):
                # GENERATE MULTILINGUAL RESPONSE
                with st.spinner(f"Consulting experts in {cfg['name']}..."):
                    # RAG Context
                    vec = embed_model.encode([st.session_state.pending_query])
                    dist, idx = faiss_idx.search(np.array(vec).astype('float32'), 1)
                    context = raw_docs[idx[0][0]] if dist[0][0] < 1.6 else "General farming knowledge."

                    # AI Multimodal Prompt
                    prompt = [
                        f"You are KrushiMitra. Context: {context}\n"
                        f"Task: Answer the question '{st.session_state.pending_query}' strictly in {cfg['name']} language.\n"
                        "Be polite, provide steps, and if an image is provided, analyze it."
                    ]
                    if st.session_state.pending_img:
                        prompt.append(Image.open(st.session_state.pending_img))

                    response = gemini_model.generate_content(prompt)
                    bot_text = response.text

                    # Update Chat
                    st.session_state.messages.append({"role": "assistant", "content": bot_text})
                    
                    # Safe Audio Generation
                    audio_file = safe_speak(bot_text, cfg['code'])
                    
                    # Final display
                    with st.chat_message("assistant"):
                        st.markdown(bot_text)
                        if audio_file:
                            st.audio(audio_file, format="audio/mp3", autoplay=True)
                    
                    # Reset
                    st.session_state.pending_query = None
                    st.session_state.pending_img = None
                st.rerun()

st.markdown("---")

st.markdown('<div style="text-align:center; color:#64748b; font-size:0.8rem;">Modern Multilingual Agri-Support • 2026</div>', unsafe_allow_html=True)

