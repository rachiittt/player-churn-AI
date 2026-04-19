import streamlit as st
import joblib
import numpy as np
import time
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

st.set_page_config(
    page_title="AI-Based Player Churn Prediction System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    return joblib.load("model.pkl"), joblib.load("scaler.pkl")

model, scaler = load_models()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@300;400;600&display=swap');

:root {
  --bg: #0a0a0f;
  --surface: #12121a;
  --surface2: #1a1a24;
  --border: #2a2a35;
  --accent: #00f0ff;
  --accent-glow: rgba(0, 240, 255, 0.2);
  --green: #00ff88;
  --yellow: #ffb800;
  --red: #ff3366;
  --text: #e2e2ec;
  --muted: #888899;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 1px;
}

.gradient-text {
  background: linear-gradient(90deg, var(--accent), #aa00ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

.card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  transition: transform 0.2s ease, border-color 0.2s ease;
  height: 100%;
}
.card:hover {
  transform: translateY(-2px);
  border-color: var(--accent);
}
.card-title {
  font-size: 0.75rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 8px;
  font-weight: 600;
}
.card-value {
  font-family: 'Orbitron', sans-serif;
  font-size: 2.2rem;
  font-weight: 700;
}
.green { color: var(--green); text-shadow: 0 0 10px rgba(0,255,136,0.3); }
.yellow { color: var(--yellow); text-shadow: 0 0 10px rgba(255,184,0,0.3); }
.red { color: var(--red); text-shadow: 0 0 10px rgba(255,51,102,0.3); }
.blue { color: var(--accent); text-shadow: 0 0 10px var(--accent-glow); }

.section-panel {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin: 20px 0;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.section-tag {
  display: inline-block;
  font-family: 'Orbitron', sans-serif;
  font-size: 0.7rem;
  padding: 4px 12px;
  border-radius: 20px;
  margin-bottom: 16px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.tag-blue { background: rgba(0,240,255,0.1); border: 1px solid var(--accent); color: var(--accent); }
.tag-purple { background: rgba(170,0,255,0.1); border: 1px solid #aa00ff; color: #d477ff; }
.tag-green { background: rgba(0,255,136,0.1); border: 1px solid var(--green); color: var(--green); }
.tag-yellow { background: rgba(255,184,0,0.1); border: 1px solid var(--yellow); color: var(--yellow); }

.step {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.9rem;
  font-weight: 600;
}
.dot { width: 10px; height: 10px; border-radius: 50%; box-shadow: 0 0 8px currentColor; }
.dot-done  { background: var(--green); color: var(--green); }
.dot-active { background: var(--accent); color: var(--accent); animation: pulse 1.5s infinite; }
.dot-wait  { background: var(--border); box-shadow: none; }
@keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.3); opacity: 0.5; } 100% { transform: scale(1); opacity: 1; } }

.stButton button {
  background: linear-gradient(90deg, var(--accent), #0099ff) !important;
  color: #000 !important;
  font-weight: 700 !important;
  font-family: 'Orbitron', sans-serif !important;
  border: none !important;
  border-radius: 8px !important;
  transition: all 0.3s ease !important;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.stButton button:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px var(--accent-glow) !important;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

KNOWLEDGE_BASE = [
    "Daily login rewards and streak bonuses reduce churn for casual players.",
    "Personalized push notifications increase re-engagement by up to 30%.",
    "Seasonal events and limited-time content create urgency for lapsed players.",
    "Better onboarding flows reduce early churn in the first 7 days.",
    "Social features like guilds and leaderboards increase session frequency.",
    "For RPG players, new story content maintains long-term engagement.",
    "Overly grindy mechanics or pay-to-win systems cause rapid disengagement.",
    "Visible progression satisfies players seeking completion.",
    "Battle passes improve retention for purchasing players.",
    "Win-back email campaigns with a free item have high success rates.",
    "Faster load times and save-anywhere features improve session duration.",
    "Discord servers and community events build identity and reduce churn.",
    "Adaptive difficulty prevents players from quitting due to frustration.",
    "Ranked leaderboards and social comparisons drive competitive engagement.",
    "Cosmetic customization heavily engages players under 20.",
    "Mid-level players often need a nudge via booster items to continue.",
    "Action players churn if matchmaking is unfair.",
    "Strategy players value balance patches and new faction content.",
    "Sports game events tied to the real-world calendar drive returns."
]

@st.cache_resource
def setup_rag():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        docs = [Document(page_content=text) for text in KNOWLEDGE_BASE]
        return FAISS.from_documents(docs, embeddings)
    except Exception:
        return None

def get_llm():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or not ChatGoogleGenerativeAI:
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=api_key)
    except Exception:
        return None

class AgentState(TypedDict):
    player_data: dict
    churn_prob: float
    risk: str
    strategies: list
    summary: str
    analysis: str
    plan: str
    refs: list
    error: Optional[str]

def node_profile(state: AgentState) -> AgentState:
    p = state["player_data"]
    prob = state["churn_prob"]
    state["risk"] = "LOW" if prob < 0.40 else "MEDIUM" if prob < 0.70 else "HIGH"
    
    purchases = "makes in-game purchases" if p["purchases"] else "does not make purchases"
    state["summary"] = (
        f"A {p['age']}-year-old {p['gender']} player from {p['location']} playing {p['genre']} games "
        f"at {p['difficulty']} difficulty. They average {p['sessions']} sessions/week ({p['avg_session']} mins each), "
        f"with {p['playtime']} hours total playtime. Currently Level {p['level']} with {p['achievements']} achievements. "
        f"This player {purchases}."
    )
    return state

def node_rag(state: AgentState) -> AgentState:
    p = state["player_data"]
    query = f"{state['risk']} churn risk, {p['genre']} game, {p['difficulty']} difficulty, age {p['age']}, {p['location']} region, {p['sessions']} sessions/week"
    vectorstore = st.session_state.get("vectorstore")
    
    if vectorstore:
        results = vectorstore.similarity_search(query, k=4)
        strategies = [r.page_content for r in results]
    else:
        strategies = KNOWLEDGE_BASE[:4]

    state["strategies"] = strategies
    state["refs"] = [f"[{i+1}] {s[:85]}..." for i, s in enumerate(strategies)]
    return state

def fallback_analysis(state, p, pct):
    flags, good = [], []
    if p["sessions"] < 3: flags.append("Very low weekly sessions")
    if p["playtime"] < 10: flags.append("Low total playtime")
    if p["achievements"] < 10: flags.append("Low engagement with progression")
    if p["sessions"] >= 7: good.append("Strong weekly habit")
    if p["level"] > 60: good.append("Highly invested level")
    
    lines = [f"**Churn Probability: {pct}% — {state['risk']} RISK**\n"]
    if flags: lines.extend(["**Risk Factors:**"] + [f"• {f}" for f in flags])
    if good: lines.extend(["\n**Positive Signals:**"] + [f"• {s}" for s in good])
    return "\n".join(lines)

def node_analysis(state: AgentState) -> AgentState:
    pct = round(state["churn_prob"] * 100, 1)
    llm = get_llm()
    if llm:
        prompt = f"""You are an expert AI retention analyst. 
Player Profile: {state['summary']}
Risk Level: {state['risk']} ({pct}% probability of churn)

Provide a concise, professional analysis identifying key risk factors driving the potential churn, and any positive signals. Use markdown bullet points. Do not generate a retention plan yet."""
        try:
            state["analysis"] = llm.invoke(prompt).content
            return state
        except Exception:
            pass
            
    state["analysis"] = fallback_analysis(state, state["player_data"], pct)
    return state

def node_plan(state: AgentState) -> AgentState:
    llm = get_llm()
    strats = "\n".join([f"- {s}" for s in state['strategies']])
    if llm:
        prompt = f"""You are an expert AI retention analyst.
Player Risk: {state['risk']}
Based on this retrieved knowledge base:
{strats}

Generate a concise, actionable, step-by-step retention plan to engage this specific player. Priority depends on risk level. Format neatly in markdown."""
        try:
            state["plan"] = llm.invoke(prompt).content
            return state
        except Exception:
            pass

    lines = [f"**Recommended Actions:**"]
    for i, strat in enumerate(state['strategies'], 1):
        lines.append(f"{i}. {strat}")
    lines.append(f"\n**Priority Response Required:** {'24 hours' if state['risk'] == 'HIGH' else '3 days' if state['risk'] == 'MEDIUM' else 'Routine update'}")
    state["plan"] = "\n".join(lines)
    return state

@st.cache_resource
def get_graph():
    g = StateGraph(AgentState)
    g.add_node("profile", node_profile)
    g.add_node("rag", node_rag)
    g.add_node("analysis", node_analysis)
    g.add_node("plan", node_plan)
    g.set_entry_point("profile")
    g.add_edge("profile", "rag")
    g.add_edge("rag", "analysis")
    g.add_edge("analysis", "plan")
    g.add_edge("plan", END)
    return g.compile()

gender_map = {'Male': 0, 'Female': 1}
location_map = {'Other': 0, 'USA': 1, 'Europe': 2, 'Asia': 3}
genre_map = {'Strategy': 0, 'Sports': 1, 'Action': 2, 'RPG': 3, 'Simulation': 4}
diff_map = {'Medium': 0, 'Easy': 1, 'Hard': 2}

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = setup_rag()

with st.sidebar:
    st.markdown("<h2 class='gradient-text'>🎮 Player Profile</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    age = st.slider("Age", 15, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Region", ["USA", "Europe", "Asia", "Other"])
    
    genre = st.selectbox("Genre", ["Action", "RPG", "Strategy", "Sports", "Simulation"])
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    purchases = st.radio("In-Game Purchases?", ["No", "Yes"], horizontal=True)
    
    playtime = st.slider("Total Playtime (hrs)", 0.0, 100.0, 15.0, 0.5)
    sessions = st.slider("Sessions / Week", 0, 40, 5)
    avg_session = st.slider("Avg Session (min)", 10, 300, 60, 5)
    level = st.slider("Player Level", 1, 100, 20)
    achievements = st.slider("Achievements", 0, 200, 25)
    
    st.markdown("---")
    run_btn = st.button("Analyze Player Performance")

st.markdown("<h1 class='gradient-text'>AI-Based Player Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--muted); font-size:1.1rem;'>Agentic Retention Assistant Pipeline (LangGraph + RAG + LLM)</p>", unsafe_allow_html=True)
st.markdown("---")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.warning("GEMINI_API_KEY environment variable not found. LLM reasoning will use fallback logic. Set your key to unlock the full AI Agent.")

if run_btn:
    X = np.array([[age, gender_map[gender], location_map[location], genre_map[genre], playtime, 1 if purchases == "Yes" else 0, diff_map[difficulty], sessions, avg_session, level, achievements]])
    X_scaled = scaler.transform(X)
    churn_prob = float(model.predict_proba(X_scaled)[0][1])
    
    player_data = {
        "age": age, "gender": gender, "location": location, "genre": genre, "difficulty": difficulty,
        "purchases": 1 if purchases == "Yes" else 0, "playtime": playtime, "sessions": sessions,
        "avg_session": avg_session, "level": level, "achievements": achievements
    }

    st.markdown("### 🤖 Agentic Pipeline Status")
    labels = {"profile": "Building Neural Profile", "rag": "Querying FAISS Knowledge Base", "analysis": "Executing LLM Churn Analysis", "plan": "Synthesizing Retention Plan"}
    placeholders = {k: st.empty() for k in labels}

    for k, label in labels.items():
        placeholders[k].markdown(f'<div class="step"><div class="dot dot-wait"></div>{label}</div>', unsafe_allow_html=True)

    for k, label in labels.items():
        placeholders[k].markdown(f'<div class="step"><div class="dot dot-active"></div><span class="gradient-text">{label}...</span></div>', unsafe_allow_html=True)
        time.sleep(0.4)
        placeholders[k].markdown(f'<div class="step"><div class="dot dot-done"></div>{label} ✓</div>', unsafe_allow_html=True)

    graph = get_graph()
    init = AgentState(player_data=player_data, churn_prob=churn_prob, risk="", strategies=[], summary="", analysis="", plan="", refs=[], error=None)
    result = graph.invoke(init)

    pct = round(churn_prob * 100, 1)
    risk = result.get("risk", "MEDIUM")
    color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(risk, "blue")
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="card"><div class="card-title">Churn Probability</div><div class="card-value {color}">{pct}%</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="card-title">Risk Assessment</div><div class="card-value {color}">{risk}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="card-title">Player Level</div><div class="card-value blue">Lvl {level}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Operational Report")

    st.markdown(f'<div class="section-panel"><span class="section-tag tag-blue">SUMMARY</span><p style="margin:0;line-height:1.6">{result["summary"]}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-panel"><span class="section-tag tag-purple">LLM ANALYSIS</span><p style="margin:0;line-height:1.6">{result["analysis"].replace(chr(10), "<br>")}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-panel" style="border-color:var(--green)"><span class="section-tag tag-green">RECOMMENDED PLAN</span><p style="margin:0;line-height:1.6">{result["plan"].replace(chr(10), "<br>")}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-panel"><span class="section-tag tag-yellow">RAG REFERENCES</span><ul style="margin:0;color:var(--muted);font-size:0.85rem">{"".join([f"<li>{r}</li>" for r in result["refs"]])}</ul></div>', unsafe_allow_html=True)

else:
    st.info("Configure the player profile via the sidebar parameters and initiate the analysis.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><div class="card-title">Orchestration</div><div class="card-value" style="font-size:1.2rem">LangGraph</div><p style="color:var(--muted);font-size:0.85rem;margin-top:10px;">Deterministic state tracking across multi-step execution graphs.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><div class="card-title">Retrieval</div><div class="card-value" style="font-size:1.2rem">FAISS + MiniLM</div><p style="color:var(--muted);font-size:0.85rem;margin-top:10px;">Semantic vector search against the engagement knowledge base.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><div class="card-title">Reasoning</div><div class="card-value" style="font-size:1.2rem">Generative AI</div><p style="color:var(--muted);font-size:0.85rem;margin-top:10px;">Advanced contextual analysis powered by Large Language Models.</p></div>', unsafe_allow_html=True)
 
