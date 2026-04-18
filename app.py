import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import os
import base64

# LangChain / LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Union

# --- GLOBAL BACKGROUND & THEME (INJECTED AT TOP) ---
# Full-page background must be injected before other page content
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@300;500;700&display=swap');

:root {
    --primary: #1B5E20;
    --secondary: #2E7D32;
    --accent: #4CAF50;
    --earth-dark: #1A2E1A;
    --sidebar-bg: rgba(27, 94, 32, 0.92);
    --glass-bg: rgba(255, 255, 255, 0.82);
    --glass-border: rgba(76, 175, 80, 0.25);
}

/* Full-page light, airy nature background */
.stApp {
    background: linear-gradient(rgba(245,252,248,0.85), rgba(245,252,248,0.85)),
                url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&q=80&w=2400') center/cover no-repeat fixed;
}

/* Sidebar Styling */
[data-testid="stSidebar"] { background: var(--sidebar-bg) !important; backdrop-filter: blur(6px); }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; font-weight: 500 !important; }
[data-testid="stSidebar"] .sidebar-header { font-weight: 700 !important; font-size: 20px !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] .st-a11y--skip-link { outline: none; }

/* Sidebar active nav highlight */
[data-testid="stSidebar"] .css-1dq8tca { border-left: 4px solid rgba(255,255,255,0.12); }

/* Text contrast rules */
h1,h2,h3 { color: #1B5E20 !important; font-weight: 700 !important; }
p, li, label, span { color: #1A2E1A !important; font-weight: 400 !important; font-size: 15px !important; }
.stMetric label { color: #388E3C !important; font-size: 13px !important; }
.stMetric [data-testid="metric-container"] > div { color: #2E7D32 !important; font-size: 28px !important; font-weight: 700 !important; }

/* Glass card style */
.eco-card { background: rgba(255,255,255,0.82); backdrop-filter: blur(12px); border: 1px solid rgba(76,175,80,0.25); border-radius: 16px; padding: 1.5rem 2rem; box-shadow: 0 4px 24px rgba(46,125,50,0.08); }

/* Metric card style */
.metric-card { background: rgba(255,255,255,0.85); border-radius: 16px; padding: 1.2rem 1.6rem; border-left: 4px solid #4CAF50; }
.metric-card .label { color: #388E3C; font-size: 13px; font-weight: 700; text-transform: uppercase; }
.metric-card .value { color: #1B5E20; font-size: 32px; font-weight: 700; }

/* Buttons */
.stButton > button { background: linear-gradient(135deg, #2E7D32, #00897B) !important; color: #ffffff !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 2rem !important; font-weight: 600 !important; font-size: 15px !important; }
.stButton > button:hover { background: #1B5E20 !important; }

/* Hide default Streamlit header/footer */
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EcoTrack | Sustainability Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THEME & STYLING ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def inject_custom_css():
    # (legacy placeholder kept; primary CSS injected at top of file)
    return

# --- UI HELPERS ---
def render_header(title, subtitle):
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.7); backdrop-filter: blur(10px); padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem; border-left: 8px solid #4CAF50; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌱</div>
        <h1 style="margin: 0; color: #1B5E20; font-size: 2.8rem;">{title}</h1>
        <p style="margin: 0; color: #388E3C; font-size: 1.1rem; font-weight: 500;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, delta=None):
    delta_html = ""
    if delta:
        color = "#2E7D32" if delta.startswith("-") or "low" in delta.lower() else "#D32F2F"
        delta_html = f'<div style="color: {color}; font-size: 14px; font-weight: 600; margin-top: 4px;">{delta}</div>'
    
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.85); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 16px; border-left: 4px solid #4CAF50; box-shadow: 0 4px 12px rgba(0,0,0,0.05); height: 100%;">
        <div style="color: #388E3C; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
        <div style="color: #1B5E20; font-size: 32px; font-weight: 700; margin-top: 8px;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def eco_card_start():
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)

def eco_card_end():
    st.markdown('</div>', unsafe_allow_html=True)

# --- PERSISTENCE LAYER ---
DATA_FILE = "eco_history.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            st.session_state.logs = data.get("logs", [])
            st.session_state.plan_history = data.get("plan_history", [])
    else:
        st.session_state.logs = []
        st.session_state.plan_history = []

def save_data():
    data = {"logs": st.session_state.logs, "plan_history": st.session_state.plan_history}
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

# --- SESSION STATE INITIALIZATION ---
if "logs" not in st.session_state:
    load_data()
if "plan_history" not in st.session_state:
    st.session_state.plan_history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"
if "plan_chat" not in st.session_state:
    st.session_state.plan_chat = []

# --- CARBON CALCULATION ENGINE ---
CO2_FACTORS = {
    "transport": {"Petrol Car": 0.17, "Diesel Car": 0.171, "Public Transport": 0.05, "Electric Vehicle": 0.047, "Flight": 0.15, "Walking/Cycling": 0.0},
    "diet": {"High Meat": 2.5, "Medium Meat": 1.8, "Low Meat": 1.2, "Vegetarian": 0.7, "Vegan": 0.5},
    "energy": {"Electricity (kWh)": 0.45, "Natural Gas (unit)": 0.19},
    "lifestyle": {"Waste (bag)": 0.5, "Water (100L)": 0.03}
}

def calculate_footprint(data):
    t_f = data.get("t_km", 0) * CO2_FACTORS["transport"].get(data.get("t_mode", "Walking/Cycling"), 0)
    d_f = data.get("d_meals", 3) * CO2_FACTORS["diet"].get(data.get("d_type", "Vegetarian"), 0)
    e_f = data.get("e_val", 0) * CO2_FACTORS["energy"].get(data.get("e_type", "Electricity (kWh)"), 0)
    l_f = (data.get("l_waste", 0) * CO2_FACTORS["lifestyle"]["Waste (bag)"]) + (data.get("l_water", 0) * CO2_FACTORS["lifestyle"]["Water (100L)"])
    total = t_f + d_f + e_f + l_f
    return {"total": round(total, 2), "breakdown": {"Transport": round(t_f, 2), "Diet": round(d_f, 2), "Energy": round(e_f, 2), "Lifestyle": round(l_f, 2)}}

# --- AI AGENT (LANGGRAPH) ---
class AgentState(TypedDict):
    habits: dict
    footprint: dict
    search_results: str
    action_plan: str
    messages: Annotated[List[Union[HumanMessage, SystemMessage]], "messages"]

def get_llm():
    if "GROQ_API_KEY" not in st.secrets:
        st.error("API Key Missing!"); st.stop()
    return ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile", temperature=0.2)

def analyze_habits(state: AgentState):
    llm = get_llm()
    prompt = f"Analyze habits: {json.dumps(state['habits'])} (Footprint: {state['footprint']['total']}kg). Identify the biggest impact area and return a search query for green alternatives. Only query text."
    state['search_results'] = llm.invoke([SystemMessage(content="You are a green analyst."), HumanMessage(content=prompt)]).content
    return state

def search_web(state: AgentState):
    try:
        with DDGS() as ddgs:
            results = ""
            for r in ddgs.text(state['search_results'], max_results=3):
                results += f"## {r['title']}\n{r['body']}\n\n"
            state['search_results'] = results
    except: state['search_results'] = "No search data available."
    return state

def generate_plan(state: AgentState):
    llm = get_llm()
    prompt = f"Create a motivational eco-strategy. Habits: {json.dumps(state['habits'])}. Footprint: {state['footprint']['total']}. Research: {state['search_results']}. Use Bold Markdown and Emojis."
    state['action_plan'] = llm.invoke([SystemMessage(content="You are a world-class sustainability coach."), HumanMessage(content=prompt)]).content
    return state

def run_agent(habits, footprint):
    wf = StateGraph(AgentState)
    wf.add_node("analyze", analyze_habits); wf.add_node("search", search_web); wf.add_node("planner", generate_plan)
    wf.set_entry_point("analyze"); wf.add_edge("analyze", "search"); wf.add_edge("search", "planner"); wf.add_edge("planner", END)
    res = wf.compile().invoke({"habits": habits, "footprint": footprint, "messages": [], "search_results": "", "action_plan": ""})
    return res['action_plan']

# --- PLAN CHAT HELPERS ---
def answer_question(question: str, context: str) -> str:
    """Use the same ChatGroq LLM to answer a user question using provided context."""
    try:
        llm = get_llm()
        prompt = f"You are an expert sustainability advisor. Use the following user plan/context to answer the user's question directly and concisely. Do NOT produce a full plan or long essay — give a clear, focused answer (2-4 sentences) and up to 3 short actionable steps if relevant.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nProvide the answer in plain text, do not reformat into a plan."
        resp = llm.invoke([SystemMessage(content="You are a helpful sustainability assistant."), HumanMessage(content=prompt)])
        return getattr(resp, 'content', str(resp))
    except Exception as e:
        return f"Error contacting LLM: {e}"

# --- UI COMPONENTS ---
def render_eco_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Daily Eco Score", 'font': {'size': 24, 'color': '#52B788'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2D6A4F"},
            'bar': {'color': "#2D6A4F"},
            'bgcolor': "rgba(0,0,0,0.1)",
            'borderwidth': 2,
            'bordercolor': "#2D6A4F",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255, 173, 173, 0.4)'},
                {'range': [40, 70], 'color': 'rgba(255, 214, 165, 0.4)'},
                {'range': [70, 100], 'color': 'rgba(183, 228, 199, 0.4)'}],
            'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#002116", 'family': "Outfit"}, margin=dict(t=30, b=0, l=30, r=30))
    return fig

# --- PAGES ---
def page_dashboard():
    render_header("Eco Dashboard", "Harnessing AI to heal our home. Tracking your journey to zero-impact.")
    
    if not st.session_state.logs:
        eco_card_start()
        st.markdown("### 👋 Welcome to EcoTrack AI!")
        st.write("Start by logging your habits in the sidebar to visualize your impact and get personalized AI advice.")
        eco_card_end()
        return

    latest = st.session_state.logs[-1]
    score = max(0, 100 - int(latest['total'] * 2.5))
    top_impact = max(latest['breakdown'], key=latest['breakdown'].get)
    
    # Custom Metric Cards
    m1, m2, m3 = st.columns(3)
    with m1: render_metric_card("Total Footprint", f"{latest['total']} kg", "Daily limit: 10kg")
    with m2: render_metric_card("Impact Score", score, "Target: 90+")
    with m3: render_metric_card("Main Area", top_impact, "High Reduction Potential")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        eco_card_start()
        st.subheader("📊 Footprint Analysis")
        st.plotly_chart(render_eco_gauge(score), use_container_width=True)
        eco_card_end()
    
    with col2:
        eco_card_start()
        st.subheader("💡 Impact Insight")
        st.markdown(f"Your largest carbon contribution currently stems from **{top_impact}**. Transitioning to more sustainable alternatives in this category could improve your score by up to **15%**.")
        st.markdown("---")
        st.markdown("Check your **Eco Action Plan** for a detailed AI-driven strategy to reduce this impact.")
        eco_card_end()

    eco_card_start()
    st.subheader("🏗️ Category Breakdown")
    df = pd.DataFrame(latest['breakdown'].items(), columns=['Category', 'CO2e (kg)'])
    fig = px.bar(df, x='Category', y='CO2e (kg)', color='Category', 
                 color_discrete_sequence=['#2D6A4F','#45A049','#81C784','#A5D6A7'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      font_color='#1A2E1A', margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
    eco_card_end()


def page_log_habits():
    render_header("Environmental Log", "Document your daily activities to sync with the green grid.")
    
    eco_card_start()
    with st.form("main_form", border=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🚗 Travel Dynamics")
            t_mode = st.selectbox("Frequency Transport Mode", list(CO2_FACTORS["transport"].keys()))
            t_km = st.number_input("Distance Traveled (km)", min_value=0.0, value=5.0)
            st.markdown("#### ⚡ Energy Grid")
            e_type = st.selectbox("Primary Energy Source", list(CO2_FACTORS["energy"].keys()))
            e_val = st.number_input("Units Consumed", min_value=0.0, value=2.0)
        with c2:
            st.markdown("#### 🥦 Dietary Intake")
            d_type = st.selectbox("Dietary Pattern", list(CO2_FACTORS["diet"].keys()))
            d_meals = st.number_input("Total Meals", min_value=1, value=3)
            st.markdown("#### ♻️ Waste & Water")
            l_waste = st.number_input("Waste Generated (Bags)", min_value=0, value=1)
            l_water = st.number_input("Water Usage (100L Units)", min_value=0.0, value=2.0)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.form_submit_button("🌍 SYNC ECOLOGICAL LOG"):
            res = calculate_footprint({"t_mode": t_mode, "t_km": t_km, "d_type": d_type, "d_meals": d_meals, "e_type": e_type, "e_val": e_val, "l_waste": l_waste, "l_water": l_water})
            entry = {**res, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.session_state.logs.append(entry); save_data(); st.success("Log synced — entry saved."); time.sleep(0.6); st.rerun()
    eco_card_end()

def page_eco_plan():
    render_header("Sustainability Intel", "AI-driven strategies to minimize your ecological footprint.")
    
    if not st.session_state.logs:
        eco_card_start()
        st.markdown("""
        <div style="background: rgba(255,255,255,0.82); padding: 1rem 1.25rem; border-radius: 12px; color: #1A2E1A;">
            <strong>⚠️ Please log your habits</strong>
            <div style="margin-top:6px">Please add a daily log in the sidebar to generate a personalized eco-strategy.</div>
        </div>
        """, unsafe_allow_html=True)
        eco_card_end()
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        eco_card_start()
        st.markdown("### 🌟 Action Planner")
        st.write("Generate a bespoke sustainability plan based on your latest environmental log.")
        if st.button("🌟 GENERATE PLAN"):
            with st.spinner("AI analyzing habits & researching alternatives..."):
                plan = run_agent(st.session_state.logs[-1], st.session_state.logs[-1])
                st.session_state.plan_history.insert(0, {"plan": plan, "date": datetime.now().strftime("%Y-%m-%d")})
                save_data(); st.rerun()
        eco_card_end()

    with col2:
        if st.session_state.plan_history:
            st.markdown("### 📜 Strategy History")
            for item in st.session_state.plan_history[:3]:
                with st.expander(f"📅 Strategy for {item['date']}", expanded=(item == st.session_state.plan_history[0])):
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.4); padding: 20px; border-radius: 12px; color: #1A2E1A;">
                        {item['plan']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            eco_card_start()
            st.markdown("""
            <div style="background: rgba(255,255,255,0.82); padding: 1rem 1.25rem; border-radius: 12px; color: #1A2E1A;">
                <strong>No plans generated yet.</strong>
                <div style="margin-top:6px">Click <em>Generate Plan</em> to create a tailored sustainability strategy.</div>
            </div>
            """, unsafe_allow_html=True)
            eco_card_end()

    # --- Chatbox: ask questions about your latest plan ---
    eco_card_start()
    st.markdown("### 💬 Chat With Your Eco Plan")
    st.markdown("<div style='color:#1A2E1A;margin-bottom:8px;'>Ask any question about your generated plan or sustainability actions. The assistant will answer using the latest plan as context.</div>", unsafe_allow_html=True)
    question = st.text_area("Your question", key="plan_question_input", height=90)
    if st.button("Ask", key="plan_ask_button"):
        if question and question.strip():
            context = st.session_state.plan_history[0]['plan'] if st.session_state.plan_history else ""
            with st.spinner("Answering..."):
                answer = answer_question(question, context)
            st.session_state.plan_chat.append({"q": question, "a": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.rerun()
        else:
            st.warning("Please enter a question first.")

    # Render chat history (most recent first)
    if st.session_state.plan_chat:
        for msg in reversed(st.session_state.plan_chat[-20:]):
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <div style='text-align:left; margin-bottom:6px; color:#1A2E1A; font-size:12px;'>📅 {msg['time']}</div>
                <div style='background: rgba(46,125,50,0.06); padding:10px 12px; border-radius:12px; color:#1A2E1A; border:1px solid rgba(46,125,50,0.08);'>
                    <strong>You:</strong> {msg['q']}
                </div>
                <div style='background: rgba(255,255,255,0.92); padding:10px 12px; border-radius:12px; margin-top:8px; color:#1A2E1A; border:1px solid rgba(76,175,80,0.12);'>
                    <strong>Assistant:</strong> {msg['a']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    eco_card_end()

def page_progress():
    render_header("Environmental Progress", "Visualizing your journey towards a carbon-neutral lifestyle.")
    
    if not st.session_state.logs:
        eco_card_start()
        st.markdown("""
        <div style="background: rgba(255,255,255,0.82); padding: 1rem 1.25rem; border-radius: 12px; color: #1A2E1A;">
            <strong>📊 No historical data available yet.</strong>
            <div style="margin-top:6px">Start logging your daily habits to populate these charts and insights.</div>
        </div>
        """, unsafe_allow_html=True)
        eco_card_end()
        return

    df = pd.DataFrame(st.session_state.logs)
    # Normalize and validate data for plotting
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M', errors='coerce')
    else:
        df['date'] = pd.NaT
    # Ensure total is numeric
    if 'total' in df.columns:
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
    else:
        df['total'] = pd.NA
    # Drop rows missing essential values
    df = df.dropna(subset=['date', 'total'])
    df = df.sort_values('date')

    eco_card_start()
    st.markdown("### 📈 Impact Over Time")
    if df.empty:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.82); padding: 0.9rem 1rem; border-radius: 10px; color: #1A2E1A;">
            No valid historical numeric data to plot. Ensure you've synced logs.
        </div>
        """, unsafe_allow_html=True)
        eco_card_end()
    else:
        fig = px.area(df, x='date', y='total', markers=True, color_discrete_sequence=['#2E7D32'])
        fig.update_traces(line_color='#2E7D32', fill='tozeroy', fillcolor='rgba(46,125,50,0.12)', marker=dict(size=6, color='#2E7D32'))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.0)',
            font_color='#1A2E1A',
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(title=dict(text='Date', font=dict(color='#1A2E1A')), tickfont=dict(color='#1A2E1A'), gridcolor='rgba(26,46,26,0.06)'),
            yaxis=dict(title=dict(text='CO2e (kg)', font=dict(color='#1A2E1A')), tickfont=dict(color='#1A2E1A'), gridcolor='rgba(26,46,26,0.06)')
        )
        st.plotly_chart(fig, use_container_width=True)
        eco_card_end()

    eco_card_start()
    st.markdown("### 📋 Historical Log Data")
    # Expand breakdown dict into columns for clearer historical table
    try:
        breakdown_df = pd.json_normalize(df['breakdown']).add_prefix('breakdown_')
        display_df = pd.concat([df.drop(columns=['breakdown'], errors='ignore').reset_index(drop=True), breakdown_df.reset_index(drop=True)], axis=1)
    except Exception:
        display_df = df
    st.dataframe(display_df, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔥 PURGE ALL ECOLOGICAL DATA"):
        if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
        st.session_state.logs = []; st.session_state.plan_history = []; st.rerun()
    eco_card_end()

def main():
    inject_custom_css()
    st.sidebar.markdown(f"<div style='background: rgba(255,255,255,0.05); padding: 18px; border-radius: 20px; text-align:center;'><div style='font-size:28px;margin-bottom:6px;'>🌱</div><h3 class='sidebar-header' style='margin:0;'>EcoTrack AI</h3><p style='margin:6px 0 0 0; color: rgba(255,255,255,0.9); font-size:12px;'>Persistence: Connected</p></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", ["Dashboard", "Log Habits", "Eco Action Plan", "Progress Tracker"])
    if page == "Dashboard": page_dashboard()
    elif page == "Log Habits": page_log_habits()
    elif page == "Eco Action Plan": page_eco_plan()
    elif page == "Progress Tracker": page_progress()

if __name__ == "__main__": main()