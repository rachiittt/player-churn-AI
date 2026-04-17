import streamlit as st
import joblib
import numpy as np
import time
import os
 
# page config - has to be first streamlit call or it breaks
st.set_page_config(
    page_title="Player Churn Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# --- load the models we trained in milestone 1 ---
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
 
model, scaler = load_models()
 
# -----------------------------------------------
# try to import langgraph and RAG stuff
# if not installed just use fallback logic below
# -----------------------------------------------
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Optional
    LANGGRAPH_OK = True
except ImportError:
    LANGGRAPH_OK = False
    from typing import TypedDict, Optional
 
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    RAG_OK = True
except ImportError:
    RAG_OK = False
 
# basic styling - kept it simple, dark theme looks cool for gaming
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@300;400;500&display=swap');
 
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #1c2333;
  --border: #21262d;
  --accent: #58a6ff;
  --green: #3fb950;
  --yellow: #d29922;
  --red: #f85149;
  --text: #c9d1d9;
  --muted: #8b949e;
}
 
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif;
}
 
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
 
/* metric cards */
.card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 22px;
  margin-bottom: 12px;
}
.card-title {
  font-size: 0.7rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 6px;
  font-family: monospace;
}
.card-value {
  font-family: 'Rajdhani', sans-serif;
  font-size: 1.9rem;
  font-weight: 700;
}
.green { color: var(--green); }
.yellow { color: var(--yellow); }
.red { color: var(--red); }
.blue { color: var(--accent); }
 
/* section panels for the agent output */
.section-panel {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 14px 0;
}
.section-panel.green-border { border-left-color: var(--green); }
.section-panel.yellow-border { border-left-color: var(--yellow); }
.section-panel.red-border { border-left-color: var(--red); }
 
.section-tag {
  display: inline-block;
  font-family: monospace;
  font-size: 0.68rem;
  padding: 2px 8px;
  border-radius: 4px;
  margin-bottom: 10px;
  font-weight: 700;
  text-transform: uppercase;
}
.tag-blue { background: #1f3a5c; color: var(--accent); }
.tag-purple { background: #2d1b5e; color: #a78bfa; }
.tag-green { background: #1a3a25; color: var(--green); }
.tag-yellow { background: #3a2a10; color: var(--yellow); }
.tag-gray { background: #21262d; color: var(--muted); }
 
/* step tracker */
.step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.85rem;
}
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.dot-done  { background: var(--green); }
.dot-active { background: var(--accent); animation: blink 1s infinite; }
.dot-wait  { background: var(--border); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
 
.stButton button {
  background: var(--accent) !important;
  color: #0d1117 !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 6px !important;
}
hr { border-color: var(--border) !important; }
div[data-testid="stMetric"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 14px !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# -----------------------------------------------
# KNOWLEDGE BASE for RAG
# we manually wrote these based on gaming research
# could use a real database later but this works
# -----------------------------------------------
KNOWLEDGE_BASE = [
    "Daily login rewards and streak bonuses reduce churn for casual players by encouraging habitual return.",
    "Personalized push notifications like 'Your guild needs you!' increase re-engagement by up to 30%.",
    "Seasonal events and limited-time content create urgency that pulls lapsed players back.",
    "Better onboarding flows reduce early churn - the first 7 days are the highest risk window.",
    "Social features like guilds and leaderboards increase session frequency for competitive players.",
    "For RPG players, new story content and side quests maintain long-term engagement effectively.",
    "If the game feels too grindy or pay-to-win, players disengage fast regardless of other factors.",
    "Visible progression (level-up animations, unlocks) satisfies players who care about completing things.",
    "Battle passes and exclusive cosmetics improve retention for players who make purchases.",
    "Win-back email campaigns with a free item or bonus currency have high success rates.",
    "Faster load times and save-anywhere features improve session duration and return rate.",
    "Discord servers and community events build identity around the game and reduce churn long-term.",
    "Hard-difficulty players often quit due to frustration - adaptive difficulty reduces this.",
    "Players from Asia region respond well to ranked leaderboards and social comparison features.",
    "Younger players under 20 engage heavily with cosmetic customization - skins, avatars, emotes.",
    "Mid-level players who stop progressing need a nudge - a booster item or easier quest helps.",
    "Players with low achievements but high playtime may be bored of the progression system.",
    "Action players churn if matchmaking is unfair - improving matchmaking directly reduces churn.",
    "Strategy players value balance patches and new faction content to stay engaged.",
    "Sports game players churn between real-world seasons - tie events to real sports calendars.",
]
 
 
# build the FAISS index - only runs once and gets cached
@st.cache_resource
def setup_rag():
    if not RAG_OK:
        return None
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        docs = [Document(page_content=text) for text in KNOWLEDGE_BASE]
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        # just use keyword fallback if this fails
        return None
 
 
# -----------------------------------------------
# AGENT STATE - this is what LangGraph passes around
# -----------------------------------------------
class AgentState(TypedDict):
    player_data: dict
    churn_prob: float
    risk: str
    strategies: list
    summary: str
    analysis: str
    plan: str
    refs: list
    disclaimer: str
    error: Optional[str]
    done_steps: list
 
 
# -----------------------------------------------
# NODE 1 - figure out risk level and write summary
# -----------------------------------------------
def node_profile(state: AgentState) -> AgentState:
    try:
        p = state["player_data"]
        prob = state["churn_prob"]
 
        # simple thresholds - tuned based on our dataset distribution
        if prob < 0.40:
            risk = "LOW"
        elif prob < 0.70:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
 
        state["risk"] = risk
 
        # write a plain english summary of the player
        purchases_text = "makes in-game purchases" if p["purchases"] else "does not make purchases"
        summary = (
            f"This is a {p['age']}-year-old {p['gender']} player from {p['location']} "
            f"who mainly plays {p['genre']} games at {p['difficulty']} difficulty. "
            f"They average {p['sessions']} sessions per week, each around {p['avg_session']} minutes long. "
            f"Total playtime is {p['playtime']} hours, they're at Level {p['level']}, "
            f"have unlocked {p['achievements']} achievements, and {purchases_text}."
        )
        state["summary"] = summary
        state["done_steps"] = state.get("done_steps", []) + ["profile"]
 
    except Exception as e:
        state["error"] = f"profile node error: {e}"
 
    return state
 
 
# -----------------------------------------------
# NODE 2 - RAG retrieval
# get relevant strategies from our knowledge base
# -----------------------------------------------
def node_rag(state: AgentState) -> AgentState:
    try:
        p = state["player_data"]
        risk = state["risk"]
 
        # build a query from player features
        query = (
            f"{risk} churn risk, {p['genre']} game, {p['difficulty']} difficulty, "
            f"age {p['age']}, {p['location']} region, {p['sessions']} sessions per week, "
            f"level {p['level']}, {p['achievements']} achievements"
        )
 
        vectorstore = st.session_state.get("vectorstore")
 
        if vectorstore:
            # semantic search
            results = vectorstore.similarity_search(query, k=4)
            strategies = [r.page_content for r in results]
        else:
            # fallback - keyword matching
            # not as good as embeddings but works without the model
            p_genre = p["genre"].lower()
            p_loc = p["location"].lower()
            strategies = []
            for doc in KNOWLEDGE_BASE:
                dl = doc.lower()
                if p_genre in dl or p_loc in dl or risk.lower() in dl:
                    strategies.append(doc)
            # add some general ones if we didn't find enough
            if len(strategies) < 4:
                for doc in KNOWLEDGE_BASE:
                    if doc not in strategies:
                        strategies.append(doc)
                    if len(strategies) >= 4:
                        break
 
        state["strategies"] = strategies[:4]
        state["refs"] = [f"[{i+1}] {s[:85]}..." for i, s in enumerate(strategies[:4])]
        state["done_steps"] = state.get("done_steps", []) + ["rag"]
 
    except Exception as e:
        state["error"] = f"rag node error: {e}"
        # still give something even if RAG fails
        state["strategies"] = KNOWLEDGE_BASE[:4]
        state["refs"] = ["[Fallback] General strategies used due to retrieval error."]
 
    return state
 
 
# -----------------------------------------------
# NODE 3 - analyze the churn risk
# look at each feature and flag what's concerning
# -----------------------------------------------
def node_analysis(state: AgentState) -> AgentState:
    try:
        p = state["player_data"]
        prob = state["churn_prob"]
        pct = round(prob * 100, 1)
 
        risk_flags = []
        good_signs = []
 
        # check each feature - these thresholds came from EDA in our notebook
        if p["sessions"] < 3:
            risk_flags.append("very low weekly sessions (under 3)")
        elif p["sessions"] >= 7:
            good_signs.append("strong weekly session habit (7+)")
 
        if p["playtime"] < 10:
            risk_flags.append("low total playtime - hasn't invested much time yet")
        elif p["playtime"] > 40:
            good_signs.append("high total playtime - invested player")
 
        if p["achievements"] < 10:
            risk_flags.append("very few achievements - not engaging with progression")
        elif p["achievements"] > 50:
            good_signs.append("lots of achievements - motivated by completion")
 
        if p["avg_session"] < 20:
            risk_flags.append("short sessions - possibly losing interest quickly")
        elif p["avg_session"] > 90:
            good_signs.append("long average sessions - deeply engaged")
 
        if p["level"] < 10:
            risk_flags.append("low level player - still in early game (high churn window)")
        elif p["level"] > 60:
            good_signs.append("high level - has put significant effort into progressing")
 
        if not p["purchases"]:
            risk_flags.append("no in-game purchases - less financially invested")
        else:
            good_signs.append("makes purchases - financially invested in the game")
 
        # build the analysis text
        lines = [f"**Churn Probability: {pct}% — {state['risk']} RISK**\n"]
 
        if risk_flags:
            lines.append("**Risk Factors:**")
            for f in risk_flags:
                lines.append(f"• {f.capitalize()}")
 
        if good_signs:
            lines.append("\n**Positive Signals:**")
            for s in good_signs:
                lines.append(f"• {s.capitalize()}")
 
        if not risk_flags and not good_signs:
            lines.append("No strong risk or positive signals detected — player shows average engagement.")
 
        state["analysis"] = "\n".join(lines)
        state["done_steps"] = state.get("done_steps", []) + ["analysis"]
 
    except Exception as e:
        state["error"] = f"analysis node error: {e}"
 
    return state
 
 
# -----------------------------------------------
# NODE 4 - build the actual retention plan
# combines what we retrieved with some genre logic
# -----------------------------------------------
def node_plan(state: AgentState) -> AgentState:
    try:
        strategies = state.get("strategies", [])
        risk = state["risk"]
        p = state["player_data"]
 
        urgency = {
            "HIGH":   "🚨 Immediate Intervention Needed",
            "MEDIUM": "⚡ Proactive Steps Recommended",
            "LOW":    "✅ Reward & Sustain Engagement"
        }
 
        lines = [f"**{urgency[risk]}**\n", "**Recommended Actions:**"]
 
        for i, strat in enumerate(strategies, 1):
            lines.append(f"{i}. {strat}")
 
        # add a genre-specific tip - we thought these up ourselves
        genre_tips = {
            "Action":     "💡 Fast and fair matchmaking is the #1 retention factor for Action players.",
            "RPG":        "💡 Release new story content or side quests to hook RPG players back in.",
            "Strategy":   "💡 A balance patch or new faction gives Strategy players a reason to return.",
            "Sports":     "💡 Tie a limited event to a real sports calendar - works really well for sports games.",
            "Simulation": "💡 New sandbox content or DLC announcement drives simulation players back."
        }
        if p["genre"] in genre_tips:
            lines.append(f"\n{genre_tips[p['genre']]}")
 
        # final priority action based on risk level
        if risk == "HIGH":
            lines.append("\n🔴 **Priority:** Send a win-back offer within 24 hours (bonus currency + exclusive item).")
        elif risk == "MEDIUM":
            lines.append("\n🟡 **Priority:** Trigger push notification for next in-game event within 3 days.")
        else:
            lines.append("\n🟢 **Priority:** Enroll in loyalty rewards program and tease upcoming content.")
 
        state["plan"] = "\n".join(lines)
 
        # ethical disclaimer - important to include this
        state["disclaimer"] = (
            "⚠ **Disclaimer:** These recommendations are meant to improve genuine player experience, "
            "not to exploit or manipulate players. Avoid dark patterns like artificial scarcity or "
            "targeting vulnerable users. All notifications should have opt-out options. "
            "This system is for academic purposes — predictions are probabilistic, not guarantees."
        )
        state["done_steps"] = state.get("done_steps", []) + ["plan"]
 
    except Exception as e:
        state["error"] = f"plan node error: {e}"
 
    return state
 
 
# -----------------------------------------------
# NODE 5 - error handler
# runs if any of the above nodes threw an error
# -----------------------------------------------
def node_fallback(state: AgentState) -> AgentState:
    # fill in defaults so the app doesn't break
    if not state.get("analysis"):
        state["analysis"] = "Analysis could not be completed due to an error."
    if not state.get("plan"):
        state["plan"] = "Could not generate plan. Please try again."
    if not state.get("disclaimer"):
        state["disclaimer"] = "Result may be incomplete."
    state["done_steps"] = state.get("done_steps", []) + ["fallback"]
    return state
 
 
def check_error(state: AgentState) -> str:
    return "fallback" if state.get("error") else "ok"
 
 
# build the graph once and reuse it
@st.cache_resource
def get_graph():
    if not LANGGRAPH_OK:
        return None
    try:
        g = StateGraph(AgentState)
 
        g.add_node("profile",  node_profile)
        g.add_node("rag",      node_rag)
        g.add_node("analysis", node_analysis)
        g.add_node("plan",     node_plan)
        g.add_node("fallback", node_fallback)
 
        g.set_entry_point("profile")
        g.add_conditional_edges("profile",  check_error, {"fallback": "fallback", "ok": "rag"})
        g.add_conditional_edges("rag",       check_error, {"fallback": "fallback", "ok": "analysis"})
        g.add_conditional_edges("analysis",  check_error, {"fallback": "fallback", "ok": "plan"})
        g.add_edge("plan",     END)
        g.add_edge("fallback", END)
 
        return g.compile()
    except Exception:
        return None
 
 
def run_pipeline(player_data, churn_prob):
    """run the agent - falls back to plain python if langgraph isn't working"""
    init = AgentState(
        player_data=player_data,
        churn_prob=churn_prob,
        risk="",
        strategies=[],
        summary="",
        analysis="",
        plan="",
        refs=[],
        disclaimer="",
        error=None,
        done_steps=[]
    )
 
    graph = get_graph()
    if graph:
        try:
            return graph.invoke(init)
        except Exception as e:
            init["error"] = str(e)
 
    # plain python fallback - same logic, no langgraph
    s = init
    s = node_profile(s)
    s = node_rag(s)
    s = node_analysis(s)
    s = node_plan(s)
    if s.get("error"):
        s = node_fallback(s)
    return s
 
 
# --- encoding maps (same as milestone 1) ---
gender_map   = {'Male': 0, 'Female': 1}
location_map = {'Other': 0, 'USA': 1, 'Europe': 2, 'Asia': 3}
genre_map    = {'Strategy': 0, 'Sports': 1, 'Action': 2, 'RPG': 3, 'Simulation': 4}
diff_map     = {'Medium': 0, 'Easy': 1, 'Hard': 2}
 
 
# init RAG on startup
if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state["vectorstore"] = setup_rag()
 
 
# -----------------------------------------------
# SIDEBAR
# -----------------------------------------------
with st.sidebar:
    st.markdown("## 🎮 Player Profile")
    st.markdown("---")
 
    with st.expander("Demographics", expanded=True):
        age      = st.slider("Age", 15, 65, 25)
        gender   = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Region", ["USA", "Europe", "Asia", "Other"])
 
    with st.expander("Game Preferences", expanded=True):
        genre      = st.selectbox("Genre", ["Action", "RPG", "Strategy", "Sports", "Simulation"])
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        purchases  = st.radio("In-Game Purchases?", ["No", "Yes"], horizontal=True)
 
    with st.expander("Engagement Stats", expanded=True):
        playtime     = st.slider("Total Playtime (hrs)", 0.0, 100.0, 15.0, 0.5)
        sessions     = st.slider("Sessions / Week", 0, 40, 5)
        avg_session  = st.slider("Avg Session (min)", 10, 300, 60, 5)
        level        = st.slider("Player Level", 1, 100, 20)
        achievements = st.slider("Achievements", 0, 200, 25)
 
    st.markdown("---")
    run_btn = st.button("🚀 Analyze Player", use_container_width=True, type="primary")
 
    # show what's available
    st.markdown("---")
    st.markdown(
        f"<small style='color:#555'>"
        f"LangGraph: {'✅' if LANGGRAPH_OK else '⚠ fallback'}<br>"
        f"RAG/FAISS: {'✅' if RAG_OK else '⚠ fallback'}"
        f"</small>",
        unsafe_allow_html=True
    )
 
 
# -----------------------------------------------
# HEADER
# -----------------------------------------------
st.markdown("# 🎮 Player Churn Prediction & Engagement AI")
st.markdown(
    "<p style='color:#8b949e;margin-top:-10px'>Milestone 2 — Agentic AI System · LangGraph + RAG + Structured Output</p>",
    unsafe_allow_html=True
)
st.markdown("---")
 
 
# -----------------------------------------------
# MAIN
# -----------------------------------------------
if run_btn:
 
    # build feature array (same order as training)
    X = np.array([[
        age,
        gender_map[gender],
        location_map[location],
        genre_map[genre],
        playtime,
        1 if purchases == "Yes" else 0,
        diff_map[difficulty],
        sessions,
        avg_session,
        level,
        achievements
    ]])
 
    X_scaled   = scaler.transform(X)
    churn_prob = float(model.predict_proba(X_scaled)[0][1])
    pct        = round(churn_prob * 100, 1)
 
    player_data = {
        "age": age, "gender": gender, "location": location,
        "genre": genre, "difficulty": difficulty,
        "purchases": 1 if purchases == "Yes" else 0,
        "playtime": playtime, "sessions": sessions,
        "avg_session": avg_session, "level": level,
        "achievements": achievements
    }
 
    # --- show agent steps running ---
    st.markdown("### 🤖 Agent Pipeline")
 
    node_labels = {
        "profile":  "🔍 Analysing player profile",
        "rag":      "📚 Retrieving strategies (RAG / FAISS)",
        "analysis": "🧠 Reasoning about churn risk",
        "plan":     "📋 Building retention plan",
    }
 
    placeholders = {}
    for key, label in node_labels.items():
        ph = st.empty()
        placeholders[key] = ph
        ph.markdown(
            f'<div class="step"><div class="dot dot-wait"></div>{label}</div>',
            unsafe_allow_html=True
        )
 
    # animate steps (just visual, real work happens after)
    for key, label in node_labels.items():
        placeholders[key].markdown(
            f'<div class="step"><div class="dot dot-active"></div><span style="color:#58a6ff">{label}...</span></div>',
            unsafe_allow_html=True
        )
        time.sleep(0.5)
        placeholders[key].markdown(
            f'<div class="step"><div class="dot dot-done"></div>{label} ✓</div>',
            unsafe_allow_html=True
        )
 
    # run the actual agent
    result = run_pipeline(player_data, churn_prob)
 
    st.markdown("---")
 
    # --- KPI cards ---
    risk = result.get("risk", "MEDIUM")
    color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(risk, "blue")
    status = {"LOW": "Highly Engaged", "MEDIUM": "Engagement Dropping", "HIGH": "High Churn Risk"}.get(risk, "—")
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Churn Probability</div>
          <div class="card-value {color}">{pct}%</div>
          <small style="color:#8b949e">ML model score</small>
        </div>
        """, unsafe_allow_html=True)
 
    with col2:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Risk Level</div>
          <div class="card-value {color}">{risk}</div>
          <small style="color:#8b949e">{status}</small>
        </div>
        """, unsafe_allow_html=True)
 
    with col3:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Player Investment</div>
          <div class="card-value blue">Lvl {level}</div>
          <small style="color:#8b949e">{achievements} achievements · {playtime}h played</small>
        </div>
        """, unsafe_allow_html=True)
 
    # risk bar
    st.markdown("---")
    st.markdown(f"**Churn Risk Gauge** — {pct}%")
    st.progress(int(pct))
 
    st.markdown("---")
 
    # --- structured output ---
    st.markdown("### 📄 Agent Report")
 
    # SUMMARY
    st.markdown(f"""
    <div class="section-panel">
      <span class="section-tag tag-blue">SUMMARY</span>
      <p style="font-size:0.9rem;line-height:1.7;margin:0">{result.get('summary','—')}</p>
    </div>
    """, unsafe_allow_html=True)
 
    # ANALYSIS
    analysis_text = result.get('analysis', '—').replace('\n', '<br>')
    st.markdown(f"""
    <div class="section-panel yellow-border">
      <span class="section-tag tag-purple">ANALYSIS</span>
      <p style="font-size:0.88rem;line-height:1.8;margin:0">{analysis_text}</p>
    </div>
    """, unsafe_allow_html=True)
 
    # PLAN
    plan_text = result.get('plan', '—').replace('\n', '<br>')
    st.markdown(f"""
    <div class="section-panel green-border">
      <span class="section-tag tag-green">PLAN</span>
      <p style="font-size:0.88rem;line-height:1.8;margin:0">{plan_text}</p>
    </div>
    """, unsafe_allow_html=True)
 
    # REFS
    refs = result.get('refs', [])
    refs_html = "".join(f"<li style='margin-bottom:5px'>{r}</li>" for r in refs)
    st.markdown(f"""
    <div class="section-panel">
      <span class="section-tag tag-yellow">REFERENCES</span>
      <ul style="font-size:0.82rem;color:#8b949e;margin:0;padding-left:16px">{refs_html}</ul>
    </div>
    """, unsafe_allow_html=True)
 
    # DISCLAIMER
    disc = result.get('disclaimer', '').replace('\n', '<br>')
    st.markdown(f"""
    <div class="section-panel">
      <span class="section-tag tag-gray">DISCLAIMER</span>
      <p style="font-size:0.82rem;color:#8b949e;line-height:1.7;margin:0">{disc}</p>
    </div>
    """, unsafe_allow_html=True)
 
    # show error if fallback was used
    if result.get("error"):
        st.warning(f"Note: fallback mode was used. ({result['error']})")
 
 
else:
    # landing page when nothing has been run yet
    st.info("👈 Set up a player profile in the sidebar and click **Analyze Player** to run the agent.")
 
    st.markdown("---")
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="section-panel">
          <b>🤖 LangGraph Agent</b><br>
          <small style="color:#8b949e">4-node pipeline with conditional routing and error fallback.</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="section-panel">
          <b>📚 RAG Retrieval</b><br>
          <small style="color:#8b949e">FAISS + sentence-transformers finds the most relevant engagement strategies.</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="section-panel">
          <b>📋 Structured Output</b><br>
          <small style="color:#8b949e">Every report: Summary → Analysis → Plan → Refs → Disclaimer.</small>
        </div>
        """, unsafe_allow_html=True)
 


