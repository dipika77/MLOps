"""
Streamlit web UI for the assistant. It only talks to the FastAPI backend over HTTP.

    streamlit run ui/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Acme AI Assistant", page_icon="🤖", layout="wide")


def call_api(method, path, **kwargs):
    """Calls the backend and returns (data, error_message)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to the backend at {API_URL}. Is the API running?"
    except requests.exceptions.Timeout:
        return None, "The backend took too long to answer. Please try again."

    if response.status_code == 429:
        return None, "You are sending too many requests. " + response.json().get("detail", "")
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"Error {response.status_code}: {detail}"
    return response.json(), None


def show_details(result):
    cols = st.columns(4)
    cols[0].caption(f"Model: {result.get('model_used') or 'none'}")
    cols[1].caption(f"Latency: {result.get('latency_ms')} ms")
    cols[2].caption(f"Confidence: {result.get('confidence')}")
    cols[3].caption("From cache ⚡" if result.get("cached") else "Fresh answer")

    if result.get("degraded"):
        st.warning("The LLM is unavailable, so this is a fallback answer built from the documents.")
    if result.get("sources"):
        st.caption("Sources: " + ", ".join(result["sources"]))
    if result.get("tools_used"):
        st.caption("Tools used: " + ", ".join(result["tools_used"]))
    with st.expander("Raw JSON response"):
        st.json(result)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
    use_cache = st.checkbox("Use response cache", value=True)

    st.divider()
    if st.button("Check backend health"):
        data, error = call_api("GET", "/health")
        if error:
            st.error(error)
        else:
            st.json(data)

    if st.button("Re-ingest documents"):
        with st.spinner("Ingesting..."):
            data, error = call_api("POST", "/ingest", params={"reset": True})
        if error:
            st.error(error)
        else:
            st.success(f"Done: {data}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Acme AI Assistant")
st.write("Ask about HR policies, IT support or the Acme CloudDrive product.")

chat_tab, batch_tab, agentic_tab = st.tabs(["💬 Chat", "📦 Batch questions", "🔎 Agentic research (W16)"])

# ---------------- chat tab ----------------
with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("details"):
                show_details(message["details"])

    question = st.chat_input("e.g. How many paid leaves do I get per year?")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        payload = {
            "question": question,
            "history": history,
            "temperature": temperature,
            "top_p": top_p,
            "use_cache": use_cache,
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result, error = call_api("POST", "/chat", json=payload)
            if error:
                st.error(error)
            else:
                st.markdown(result["answer"])
                show_details(result)
                if result.get("follow_up_questions"):
                    st.caption("You could also ask: " + " | ".join(result["follow_up_questions"]))

        st.session_state.messages.append({"role": "user", "content": question})
        if not error:
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "details": result})

# ---------------- batch tab ----------------
with batch_tab:
    st.write("Enter one question per line (max 10). They are processed concurrently on the backend.")
    text = st.text_area(
        "Questions",
        "What is the price of the Business plan?\nHow long is parental leave?\nHow do I reset my password?",
        height=150,
    )
    if st.button("Run batch"):
        questions = [q.strip() for q in text.splitlines() if q.strip()]
        if not questions:
            st.warning("Please enter at least one question.")
        else:
            with st.spinner(f"Processing {len(questions)} questions..."):
                data, error = call_api("POST", "/chat/batch", json={"questions": questions[:10], "use_cache": use_cache})
            if error:
                st.error(error)
            else:
                st.success(f"Finished in {data['total_latency_ms']} ms")
                for q, r in zip(questions, data["results"]):
                    st.markdown(f"**Q: {q}**")
                    st.markdown(r["answer"])
                    show_details(r)
                    st.divider()

# ---------------- agentic research tab (W16) ----------------
with agentic_tab:
    st.write(
        "Runs the Researcher/Verifier loop (`/agentic_chat`): the model decides whether to search again, "
        "calculate, ask you a clarifying question, or finish - based on what a separate Verifier agent "
        "says about the evidence so far. Try a question that needs a real number from a document, or "
        "turn on 'Simulate search failure' in the backend .env to see how it reacts to a broken tool."
    )
    max_iters = st.slider("Max iterations", 1, 8, 4)
    agentic_question = st.text_input(
        "Question", "How many paid annual leave days do employees get, and how are they accrued?"
    )
    if st.button("Run agentic research"):
        with st.spinner("Researching..."):
            data, error = call_api(
                "POST", "/agentic_chat", json={"question": agentic_question, "max_iterations": max_iters}
            )
        if error:
            st.error(error)
        else:
            st.markdown(f"**Answer:** {data['answer']}")
            cols = st.columns(4)
            cols[0].caption(f"Verification: {data['verification']}")
            cols[1].caption(f"Iterations: {data['iterations']}")
            cols[2].caption(f"Total tokens: {data['total_tokens']}")
            cols[3].caption(f"Latency: {data['latency_ms']} ms")
            if data.get("sources"):
                st.caption("Sources: " + ", ".join(data["sources"]))
            st.markdown("**Trajectory** (one row per agent decision):")
            st.table(data["trajectory"])
            with st.expander("Full evidence log + raw JSON"):
                st.json(data)
