import streamlit as st
import requests
import uuid
from typing import Optional, Dict, Any

st.set_page_config(page_title="LangGraph + Streamlit", page_icon="🧩")
st.title("Homework 1")

DEFAULT_BASE_URL = "http://127.0.0.1:2024"
GRAPH_ID = "agent"  # from your langgraph.json

# ----------------------------
# Session state
# ----------------------------
if "base_url" not in st.session_state:
    st.session_state.base_url = DEFAULT_BASE_URL

if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = GRAPH_ID

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None  # create on first use

if "chat" not in st.session_state:
    st.session_state.chat = []  # [{"role": "user"|"assistant", "content": str}]


# ----------------------------
# Helpers
# ----------------------------
def render_chat():
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def ensure_thread(base_url: str) -> str:
    """
    Creates a thread if one doesn't exist in session.
    """
    if st.session_state.thread_id:
        return st.session_state.thread_id

    url = base_url.rstrip("/") + "/threads"
    r = requests.post(url, json={}, timeout=20)
    r.raise_for_status()

    data = r.json()
    # Common response: {"thread_id": "..."} or {"id": "..."}
    thread_id = data.get("thread_id") or data.get("id")
    if not thread_id:
        raise RuntimeError(f"Could not parse thread id from /threads response: {data}")

    st.session_state.thread_id = thread_id
    return thread_id


def run_wait(base_url: str, thread_id: str, assistant_id: str, user_text: str) -> Dict[str, Any]:
    """
    Calls: POST /threads/{thread_id}/runs/wait
    """
    url = base_url.rstrip("/") + f"/threads/{thread_id}/runs/wait"

    payload = {
        "assistant_id": assistant_id,
        "input": {
            "messages": [{"role": "human", "content": user_text}]
        },
        # optional, but harmless:
        "config": {"configurable": {"thread_id": thread_id}},
    }

    r = requests.post(url, json=payload, timeout=120)
    if r.status_code == 404:
        # Helpful diagnostics: is it the thread or assistant?
        raise requests.HTTPError(
            f"404 Not Found calling {url}. "
            f"Check that thread_id exists and assistant_id='{assistant_id}' exists."
        )
    r.raise_for_status()
    return r.json()


def extract_assistant_text(resp_json: Dict[str, Any]) -> Optional[str]:
    """
    Try common shapes from runs/wait response.
    """
    if not isinstance(resp_json, dict):
        return None

    out = resp_json.get("output") or resp_json.get("result") or resp_json.get("state") or resp_json

    # Most common: output/state contains messages
    if isinstance(out, dict):
        msgs = out.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                return last.get("content") or last.get("text")
            return str(last)

    # Sometimes it's directly a string field
    for k in ("final", "content", "text", "answer"):
        v = resp_json.get(k)
        if isinstance(v, str) and v.strip():
            return v

    return None


def assistant_exists(base_url: str, assistant_id: str) -> bool:
    """
    Checks if assistant exists via GET /assistants/{assistant_id}
    """
    url = base_url.rstrip("/") + f"/assistants/{assistant_id}"
    r = requests.get(url, timeout=10)
    return r.status_code == 200


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.subheader("Settings")
    st.text_input("Base URL", key="base_url")
    st.text_input("Assistant ID", key="assistant_id")
    st.caption("GRAPH ID must exist on the server (e.g., 'agent').")

    if st.button("Reset chat + new thread"):
        st.session_state.chat = []
        st.session_state.thread_id = None
        st.rerun()

    st.divider()
    st.write("Current thread_id:")
    st.code(st.session_state.thread_id or "(not created yet)")

    # Quick diagnostics
    if st.button("Check assistant exists"):
        ok = assistant_exists(st.session_state.base_url, st.session_state.assistant_id)
        st.success("Assistant found ✅" if ok else "Assistant NOT found ❌")


# ----------------------------
# Main
# ----------------------------
render_chat()

user_text = st.chat_input("Type a message…")
if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})

    try:
        base = st.session_state.base_url
        assistant_id = st.session_state.assistant_id

        # Ensure thread exists on server
        thread_id = ensure_thread(base)

        # Run and wait
        data = run_wait(base, thread_id, assistant_id, user_text)

        assistant_text = extract_assistant_text(data)
        if not assistant_text:
            assistant_text = (
                "Run completed, but I couldn't confidently parse the assistant message.\n\n"
                f"**Raw JSON:**\n```json\n{data}\n```"
            )

        st.session_state.chat.append({"role": "assistant", "content": assistant_text})
        st.rerun()

    except requests.RequestException as e:
        st.session_state.chat.append(
            {"role": "assistant", "content": f"Error! Backend call failed:\n\n`{e}`"}
        )
        st.rerun()
    except Exception as e:
        st.session_state.chat.append(
            {"role": "assistant", "content": f"Error:\n\n`{e}`"}
        )
        st.rerun()