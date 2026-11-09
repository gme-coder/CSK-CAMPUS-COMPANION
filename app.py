"""CSK Campus Companion: a Streamlit app that organizes and answers questions about campus notices."""
import re
from datetime import date

import pandas as pd
import streamlit as st

from src.calendar_utils import build_table, to_ics
from src.classify import predict_category
from src.extract import extract_all
from src.qa import MIN_SCORE, answer
from src.search import NoticeIndex

st.set_page_config(page_title="Campus Companion", page_icon="🎓", layout="wide")


# ------------------------------------------------------------------ header
st.title("🎓 Campus Companion")
st.caption("Never miss a deadline again. Paste a notice, see what matters, and ask questions about it.")

# ------------------------------------------------------------------ state
@st.cache_data
def load_base_notices():
    df = pd.read_csv("data/notices.csv").dropna(subset=["text", "category"])
    return df[["text", "category"]].to_dict("records")


@st.cache_resource(show_spinner="Loading the language model (first run only)...")
def load_index():
    return NoticeIndex([n["text"] for n in load_base_notices()])


if "added" not in st.session_state:
    st.session_state.added = []      # notices the student added during this session
if "chat" not in st.session_state:
    st.session_state.chat = []

try:
    index = load_index()
except Exception as e:
    st.error("The language model could not be loaded. Check your internet connection "
             "(the first run downloads about 90 MB) and restart the app.")
    st.exception(e)
    st.stop()
# Index = base notices + anything added this session. Rebuild only the extra rows.
indexed_extra = getattr(index, "_extra", 0)
for n in st.session_state.added[indexed_extra:]:
    index.add(n["text"])
index._extra = len(st.session_state.added)

all_notices = load_base_notices() + st.session_state.added
today = date.today()

tab_dash, tab_ask, tab_add = st.tabs(["📅 Deadlines", "💬 Ask a question", "➕ Add a notice"])

# ------------------------------------------------------------------ dashboard
with tab_dash:
    table = build_table(all_notices, today)
    dated = table[table["date"].notna()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notices", len(table))
    c2.metric("Urgent (next 2 days)", int((table["status"] == "Urgent").sum()))
    c3.metric("This week", int((table["status"] == "This week").sum()))
    c4.metric("No date found", int((table["status"] == "No date").sum()))

    cats = ["All"] + sorted(table["category"].dropna().unique().tolist())
    f1, f2 = st.columns([1, 1])
    category = f1.selectbox("Category", cats)
    hide_past = f2.checkbox("Hide past notices", value=True)

    view = table.copy()
    if category != "All":
        view = view[view["category"] == category]
    if hide_past:
        view = view[view["status"] != "Past"]

    icons = {"Urgent": "🔴 Urgent", "This week": "🟠 This week", "Upcoming": "🟢 Upcoming",
             "Past": "⚪ Past", "No date": "⚫ No date"}
    show = view.assign(status=view["status"].map(icons))[
        ["status", "date", "time", "category", "venue", "audience", "text"]]
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={"text": st.column_config.TextColumn("Notice", width="large")})

    st.download_button(
        "📥 Export to calendar (.ics)",
        data=to_ics(view),
        file_name="campus_deadlines.ics",
        mime="text/calendar",
        help="Import into Google Calendar, Outlook or Apple Calendar.",
    )

# ------------------------------------------------------------------ Q&A
with tab_ask:
    st.write("Ask in plain language. Answers come only from the notices, with sources shown.")
    examples = ["When does unit registration close?", "How do I pay fees?",
                "Is there a scholarship I can apply for?"]
    cols = st.columns(len(examples))
    clicked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            clicked = ex

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for s in msg.get("sources", []):
                st.caption(f"📌 Source (match {s['score']:.2f}): {s['text']}")

    question = st.chat_input("e.g. When is the next CAT?") or clicked
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.spinner("Searching the notices..."):
            res = answer(question, index, today=today, min_score=MIN_SCORE)
        label = {"llm": "AI-generated answer", "retrieval": "Best matching notice",
                 "no_match": "No match"}[res["mode"]]
        st.session_state.chat.append({
            "role": "assistant",
            "content": f"**{label}**\n\n{res['answer']}",
            "sources": res["sources"],
        })
        st.rerun()

# ------------------------------------------------------------------ add notice
def split_notices(raw):
    """Treat blank-line-separated blocks as separate notices."""
    return [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", raw) if len(b.strip()) > 15]


with tab_add:
    st.write("Paste a notice, or upload a PDF or text file. The app classifies it, finds the key details "
             "and adds it to your deadlines and the Q&A.")
    pasted = st.text_area("Paste notice text (separate several notices with a blank line)", height=150)
    uploaded = st.file_uploader("...or upload a PDF / .txt file", type=["pdf", "txt"])

    raw = pasted
    if uploaded is not None:
        if uploaded.name.lower().endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(uploaded) as pdf:
                raw = "\n\n".join((p.extract_text() or "") for p in pdf.pages)
        else:
            raw = uploaded.read().decode("utf-8", errors="ignore")

    if st.button("Analyse and add", type="primary") and raw.strip():
        found = split_notices(raw)
        if not found:
            st.warning("I couldn't find any notice text. Try pasting a longer message.")
        for text in found:
            cat = predict_category(text)
            info = extract_all(text, today)
            st.session_state.added.append({"text": text, "category": cat})
            st.success(f"**{cat}**: {text[:120]}{'...' if len(text) > 120 else ''}")
            st.json({k: v for k, v in info.items() if v})
        st.info("Added. Open the Deadlines or Ask tab to see the result.")

st.divider()
st.caption("Built for the CSK AI/ML Mini-Hackathon. Classifier, extractor and search run locally; "
           "answers are grounded in the notices shown as sources.")