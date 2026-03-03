import os
import re
import json
import base64
import requests
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit_cookies_manager_ext import EncryptedCookieManager

from lib.ollama_api import chat_stream, list_models
from lib.render import render_chat_text
from lib.attachments import (
    extract_text_from_bytes,
    is_image_mime,
)
from lib.storage import (
    init_db,
    any_admin_exists,
    upsert_user,
    verify_user,
    change_user_password,
    get_user_api_key,
    set_user_api_key,
    create_session,
    get_session,
    delete_session,
    list_users,
    get_setting,
    set_setting,
    get_base_system_prompt,
    set_base_system_prompt,
    list_assignments,
    add_assignment,
    get_active_assignment,
    set_active_assignment,
    update_assignment_prompt,
    get_assignment,
    create_conversation,
    touch_conversation,
    get_conversation,
    list_conversations_for_user,
    list_conversations_admin,
    get_conversation_messages,
    add_message,
    update_message,
    delete_messages_after,
    add_attachment,
    list_attachments_for_message_ids,
)

# ---------------- App config ----------------

APP_NAME = "DS330 Chat"

DEFAULT_BASE_PROMPT = """You are a helpful assistant for DS330. Follow the course rules and be concise, correct, and student-friendly."""

# If the model supports it (e.g., gpt-oss), enable extended thinking by default.
DEFAULT_THINK = None  # disable extended thinking by default

# When building LLM payloads, include heavy attachment context only for recent user turns.
INCLUDE_FILE_TEXT_LAST_N_USER_MSGS = 3
INCLUDE_IMAGES_LAST_N_USER_MSGS = 1


def _secret(key: str, default: Optional[str] = None) -> Optional[str]:
    if key in st.secrets:
        v = st.secrets.get(key)
        return str(v) if v is not None else default
    return os.environ.get(key, default)


OLLAMA_HOST = _secret("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY = _secret("OLLAMA_API_KEY")


def _effective_api_key() -> Optional[str]:
    """Return the API key to use for model calls.

    Preference order:
      1) User-saved key (BYOK)
      2) Instructor/server key (OLLAMA_API_KEY)
    """
    uid = st.session_state.get("user_id")
    if uid:
        try:
            k = get_user_api_key(str(uid))
            if k:
                return k
        except Exception:
            pass
    return OLLAMA_API_KEY


MODEL_SETTING_KEY = "active_model"

st.set_page_config(page_title=APP_NAME, page_icon="💬", layout="wide")

# Tighten sidebar spacing + slightly widen it
st.markdown(
    """
<style>
  /* Reduce extra top padding in sidebar */
  section[data-testid="stSidebar"] > div {
    padding-top: 0.25rem;
  }

  /* Make the sidebar a bit wider on desktop (no manual margin-left hacks) */
  @media (min-width: 900px) {
    section[data-testid="stSidebar"] {
      width: 360px !important;
      min-width: 360px !important;
      max-width: 360px !important;
    }
    section[data-testid="stSidebar"] > div {
      width: 360px !important;
    }
  }

  /* Use the full main-area width + reduce top whitespace */
  div.block-container {
    max-width: 100% !important;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
    padding-top: 0.75rem !important;
  }

  /* Copy buttons (ChatGPT-like subtle) */
  .ds330-copy-wrap {
    display: flex;
    justify-content: flex-end;
    margin-top: 0.15rem;
    gap: 0.35rem;
  }
  .ds330-copy-icon-btn {
    font-size: 13px;
    color: rgba(0, 0, 0, 0.40);
    background: transparent;
    border: 1px solid transparent;
    padding: 2px 6px;
    border-radius: 8px;
    cursor: pointer;
    line-height: 1.2;
  }
  .ds330-copy-icon-btn:hover {
    color: rgba(0, 0, 0, 0.62);
    border-color: rgba(0, 0, 0, 0.12);
    background: rgba(0, 0, 0, 0.03);
  }
  .ds330-copy-all-btn {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.36);
    background: transparent;
    border: 1px solid rgba(0, 0, 0, 0.10);
    padding: 3px 10px;
    border-radius: 10px;
    cursor: pointer;
    line-height: 1.4;
  }
  .ds330-copy-all-btn:hover {
    color: rgba(0, 0, 0, 0.58);
    border-color: rgba(0, 0, 0, 0.18);
    background: rgba(0, 0, 0, 0.03);
  }

  

  /* Sidebar: flex column so we can pin logout to bottom */
@media (min-width: 0px) {
  section[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  .ds330-sidebar-spacer { flex: 1 1 auto; }
  .ds330-sidebar-logout-wrap {
    padding-top: 0.75rem;
    padding-bottom: 0.25rem;
    border-top: 1px solid rgba(0, 0, 0, 0.08);
  }
  .ds330-sidebar-logout-wrap div.stButton > button {
    width: 100%;
    border: 1px solid rgba(0, 0, 0, 0.12) !important;
    border-radius: 10px !important;
    background: rgba(0, 0, 0, 0.02) !important;
  }
  .ds330-sidebar-logout-wrap div.stButton > button:hover {
    border-color: rgba(0, 0, 0, 0.18) !important;
    background: rgba(0, 0, 0, 0.04) !important;
  }
}
</style>
""",
    unsafe_allow_html=True,
)
# ---------------- Clipboard button (per-message + whole convo) ----------------


def _copy_button(
    text: str,
    key: str,
    label: str,
    css_class: str,
    tooltip: str = "Copy",
) -> None:
    """Render a copy-to-clipboard button (no downloads)."""
    try:
        import streamlit.components.v1 as components

        payload = json.dumps(text)
        safe_tooltip = json.dumps(tooltip)

        # If it's the icon button, use a ✓ confirmation; otherwise "copied".
        confirm = "✓" if "copy-icon" in css_class else "copied"

        html = f"""
        <div class="ds330-copy-wrap">
          <button class="{css_class}" id="{key}" title={safe_tooltip} aria-label={safe_tooltip}>{label}</button>
        </div>
        <script>
          (() => {{
            const btn = document.getElementById("{key}");
            if (!btn) return;
            const original = btn.textContent;
            btn.addEventListener("click", async () => {{
              try {{
                await navigator.clipboard.writeText({payload});
                btn.textContent = "{confirm}";
                setTimeout(() => {{ btn.textContent = original; }}, 900);
              }} catch (e) {{
                console.error(e);
              }}
            }});
          }})();
        </script>
        """
        components.html(html, height=32)
    except Exception:
        # Never fall back to downloads.
        return


# ---------------- Cookies / Auth ----------------


cookies = EncryptedCookieManager(
    prefix="ds330_chat",
    password=_secret("COOKIE_PASSWORD", "change-me"),
)

if not cookies.ready():
    st.stop()


def _login(user_id: str, role: str) -> None:
    token = create_session(user_id, role)
    cookies["session_token"] = token
    cookies.save()
    st.session_state["user_id"] = user_id
    st.session_state["role"] = role
    st.session_state["session_token"] = token


def _logout() -> None:
    token = st.session_state.get("session_token") or cookies.get("session_token")
    if token:
        try:
            delete_session(token)
        except Exception:
            pass
    cookies["session_token"] = ""
    cookies.save()
    for k in ["user_id", "role", "session_token", "conversation_id", "messages", "conversation_meta"]:
        st.session_state.pop(k, None)


# ---------------- DB init ----------------

init_db()


# ---------------- Session restore ----------------

if "user_id" not in st.session_state:
    token = cookies.get("session_token")
    if token:
        sess = get_session(token)
        if sess:
            st.session_state["user_id"] = sess["user_id"]
            st.session_state["role"] = sess["role"]
            st.session_state["session_token"] = sess["token"]





# ---------------- Model list (Ollama Cloud) ----------------

@st.cache_data(ttl=60, show_spinner=False)
def _cached_models() -> List[str]:
    try:
        return list_models(OLLAMA_HOST, OLLAMA_API_KEY)
    except Exception:
        return []


def _load_active_model(models: List[str]) -> str:
    saved = get_setting(MODEL_SETTING_KEY, None)
    if saved and saved in models:
        return saved
    if models:
        set_setting(MODEL_SETTING_KEY, models[0])
        return models[0]
    return ""


# ---------------- Prompts / Assignments ----------------


def _combined_prompt(base_prompt: str, assignment_prompt: str) -> str:
    base = (base_prompt or "").strip()
    ap = (assignment_prompt or "").strip()
    if not base:
        base = DEFAULT_BASE_PROMPT.strip()
    return base + ("\n\n" + ap if ap else "")


def _active_assignment_label() -> str:
    a = get_active_assignment()
    return a.get("name") or "(no assignment)"


# ---------------- Helpers ----------------


def _load_conversation_into_state(conversation_id: int) -> None:
    conv = get_conversation(conversation_id)
    if not conv:
        st.session_state.pop("conversation_id", None)
        st.session_state["messages"] = []
        st.session_state["conversation_meta"] = {}
        return

    msgs = get_conversation_messages(conversation_id)
    # Attachments
    mids = [m["id"] for m in msgs]
    att_map = list_attachments_for_message_ids(mids)

    ui_msgs: List[Dict[str, Any]] = []
    for m in msgs:
        ui_msgs.append(
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "attachments": att_map.get(m["id"], []),
            }
        )

    st.session_state["conversation_id"] = conversation_id
    st.session_state["messages"] = ui_msgs
    st.session_state["conversation_meta"] = conv


def _conversation_to_text(messages: List[Dict[str, Any]]) -> str:
    """Plain-text transcript (no images)."""
    lines: List[str] = []
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_payload_messages(conversation_id: int) -> List[Dict[str, Any]]:
    """Build messages payload for Ollama /api/chat.

    Ollama's ChatRequest expects:
      - messages[].content as a STRING (not an array)
      - optional messages[].images as a list of base64-encoded image bytes
    """
    conv = st.session_state.get("conversation_meta") or get_conversation(conversation_id) or {}

    # Prefer per-conversation snapshot prompt (for reproducibility)
    base_prompt = (conv.get("base_prompt") or "").strip()
    ap = (conv.get("assignment_prompt") or "").strip()
    sys_prompt = (conv.get("system_prompt") or "").strip()

    if base_prompt or ap:
        system_prompt = _combined_prompt(base_prompt, ap)
    elif sys_prompt:
        system_prompt = sys_prompt
    else:
        # fallback to current global settings
        base = get_base_system_prompt(DEFAULT_BASE_PROMPT)
        ap2 = get_active_assignment().get("prompt") or ""
        system_prompt = _combined_prompt(base, ap2)

    payload: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    msgs = st.session_state.get("messages", [])
    user_idxs = [i for i, mm in enumerate(msgs) if mm.get("role") == "user"]
    include_files_from = set(user_idxs[-INCLUDE_FILE_TEXT_LAST_N_USER_MSGS:]) if INCLUDE_FILE_TEXT_LAST_N_USER_MSGS > 0 else set()
    include_images_from = set(user_idxs[-INCLUDE_IMAGES_LAST_N_USER_MSGS:]) if INCLUDE_IMAGES_LAST_N_USER_MSGS > 0 else set()

    for i, m in enumerate(msgs):
        role = m.get("role") or "user"
        content = m.get("content") or ""

        if role == "user":
            content_out = content
            msg_obj: Dict[str, Any] = {"role": "user", "content": content_out}

            atts = m.get("attachments") or []

            # Append extracted file text for the last N user messages only (keeps prompts small).
            if i in include_files_from:
                for att in atts:
                    if att.get("kind") == "file" and att.get("text_content"):
                        fn = att.get("filename") or "file"
                        content_out += f"\n\n[Attached file: {fn}]\n{att.get('text_content','')}\n"
                msg_obj["content"] = content_out

            # Include images as base64 for the last N user messages only.
            if i in include_images_from:
                images_b64: List[str] = []
                for att in atts:
                    if att.get("kind") == "image" and att.get("data"):
                        try:
                            images_b64.append(base64.b64encode(att["data"]).decode("utf-8"))
                        except Exception:
                            continue
                if images_b64:
                    msg_obj["images"] = images_b64

            payload.append(msg_obj)

        else:
            payload.append({"role": "assistant", "content": content})

    return payload


# ---------------- Login screen ----------------


def _render_login() -> None:
    st.title(APP_NAME)

    if not any_admin_exists():
        st.info("No admin account exists yet. Create the first admin below.")
        with st.form("bootstrap_admin"):
            user_id = st.text_input("Admin user ID")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create admin")
        if submitted:
            if not user_id or not password:
                st.error("User ID and password are required.")
            else:
                upsert_user(user_id, password, "admin")
                _login(user_id, "admin")
                st.rerun()
        return

    st.subheader("Sign in")
    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        auth = verify_user(user_id, password)
        if not auth:
            st.error("Invalid credentials")
            return
        _login(auth["user_id"], auth["role"])
        st.rerun()


# ---------------- Sidebar ----------------


def _sidebar(models: List[str]) -> Tuple[str, str, Dict[str, Any]]:
    user_id = st.session_state["user_id"]
    role = st.session_state["role"]
    is_admin = role == "admin"


    # Bring-your-own API key (per user, optional)
    # If a user saves a key, we prefer it over the instructor/server key for model calls.
    try:
        saved_key = get_user_api_key(user_id)
    except Exception:
        saved_key = None

    with st.sidebar.container():
        hdr_cols = st.columns([0.90, 0.10])
        with hdr_cols[0]:
            st.markdown("**Bring your own API key** *(optional — faster responses)*")
        with hdr_cols[1]:
            # Prefer a popover so the UI stays compact; fall back to a simple link.
            try:
                with st.popover("ℹ️"):
                    st.markdown("Get your key from: [Ollama API keys](https://ollama.com/settings/keys)")
            except Exception:
                st.markdown("[ℹ️](https://ollama.com/settings/keys)")

        if saved_key:
            st.caption("✅ Using your personal API key for model calls.")
        else:
            st.caption("Using the instructor-provided API key.")

        api_input = st.text_input(
            "Ollama API key",
            type="password",
            key="byok_api_key",
            placeholder="Paste your Ollama API key here",
            label_visibility="collapsed",
        )

        btn_cols = st.columns([0.55, 0.45])
        if btn_cols[0].button("Save key", key="byok_save_key", type="primary"):
            if not api_input.strip():
                st.error("Please paste a key (or use 'Use instructor key' to clear).")
            else:
                set_user_api_key(user_id, api_input.strip())
                st.success("Saved. Your next requests will use your key.")
                st.rerun()

        if saved_key:
            if btn_cols[1].button("Use instructor key", key="byok_clear_key"):
                set_user_api_key(user_id, None)
                st.success("Cleared. Your next requests will use the instructor key.")
                st.rerun()

    st.sidebar.divider()

    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.caption(f"Signed in as **{user_id}** ({role})")

    # Active assignment (students can see; admin can change)
    active_assignment = get_active_assignment()
    st.sidebar.caption(f"**Active assignment:** {active_assignment.get('name')}")

    # Model selection
    active_model = _load_active_model(models)
    if is_admin:
        if models:
            sel = st.sidebar.selectbox("Active model", models, index=models.index(active_model))
            if sel != active_model:
                set_setting(MODEL_SETTING_KEY, sel)
                active_model = sel
        else:
            st.sidebar.warning("No models found. Check OLLAMA_HOST / API key.")

        # Assignment selection (admin only)
        assignments = list_assignments()
        if assignments:
            id_to_name = {int(a["id"]): a["name"] for a in assignments}
            ids = list(id_to_name.keys())
            active_id = int(active_assignment["id"])
            idx = ids.index(active_id) if active_id in ids else 0
            new_id = st.sidebar.selectbox(
                "Set active assignment",
                ids,
                format_func=lambda i: id_to_name.get(int(i), str(i)),
                index=idx,
            )
            if int(new_id) != active_id:
                set_active_assignment(int(new_id))
                st.rerun()

    else:
        st.sidebar.caption(f"**Active model:** {active_model}")

    st.sidebar.divider()
    if is_admin:
        page = st.sidebar.radio("Navigation", ["Chat", "Admin Dashboard"], index=0, key="nav_page", label_visibility="collapsed")
        st.sidebar.divider()
    else:
        page = "Chat"

    # Push logout to the bottom of the sidebar (no query params / no new tab)
    st.sidebar.markdown('<div class="ds330-sidebar-spacer"></div>', unsafe_allow_html=True)

    # Self-service password change (all roles)
    with st.sidebar.expander("Change password", expanded=False):
        # Use a form so inputs clear safely after submit.
        try:
            form_ctx = st.form("change_password_form", clear_on_submit=True)
        except TypeError:
            # Older Streamlit
            form_ctx = st.form("change_password_form")

        with form_ctx:
            current_pw = st.text_input("Current password", type="password", key="cp_current")
            new_pw = st.text_input("New password", type="password", key="cp_new")
            confirm_pw = st.text_input("Confirm new password", type="password", key="cp_confirm")
            submitted = st.form_submit_button("Update password")

        if submitted:
            if not current_pw or not new_pw or not confirm_pw:
                st.error("Please fill in all fields.")
            elif new_pw != confirm_pw:
                st.error("New passwords do not match.")
            elif len(new_pw) < 6:
                st.error("Password must be at least 6 characters.")
            elif not change_user_password(user_id, current_pw, new_pw):
                st.error("Current password is incorrect.")
            else:
                st.success("Password successfully changed!")

    st.sidebar.markdown('<div class="ds330-sidebar-logout-wrap">', unsafe_allow_html=True)
    try:
        logout_clicked = st.sidebar.button("Logout", key="logout_btn", width="stretch")
    except TypeError:
        logout_clicked = st.sidebar.button("Logout", key="logout_btn", use_container_width=True)
    if logout_clicked:
        _logout()
        st.rerun()

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    return page, active_model, active_assignment


# ---------------- Chat UI ----------------


def _render_attachments(attachments: List[Dict[str, Any]]) -> None:
    if not attachments:
        return

    images = [a for a in attachments if a.get("kind") == "image" and a.get("data")]
    files = [a for a in attachments if a.get("kind") == "file"]

    if images:
        st.caption("Attachments")
        for img in images:
            try:
                st.image(img["data"], caption=img.get("filename"), width="stretch")
            except TypeError:
                st.image(img["data"], caption=img.get("filename"), use_container_width=True)

    if files:
        st.caption("Attachments")
        for f in files:
            name = f.get("filename") or "file"
            st.markdown(f"- **{name}**")


def _render_message(role: str, content: str) -> None:
    if role == "assistant":
        render_chat_text(content)
    else:
        st.markdown(content)


def _chat_page(active_model: str, active_assignment: Dict[str, Any]) -> None:
    user_id = st.session_state["user_id"]
    role = st.session_state["role"]
    is_admin = role == "admin"

        # Ensure state
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_meta", {})

    # Chat header (ChatGPT-like)
    title = f"{active_model}"
    if DEFAULT_THINK:
        title += " — Thinking"
    st.markdown("# DS 330 Chat")

    # Sidebar thread picker
    with st.sidebar:
        st.markdown("#### Conversations")
        convs = list_conversations_for_user(user_id)
        options = [None] + [c["id"] for c in convs]
        labels = {None: "➕ New conversation"}
        for c in convs:
            title = c.get("title") or f"Conversation {c['id']}"
            labels[c["id"]] = title

        current = st.session_state.get("conversation_id")
        idx = options.index(current) if current in options else 0
        picked = st.selectbox("Conversation", options, index=idx, format_func=lambda x: labels.get(x, str(x)), label_visibility="collapsed")
        if picked != current:
            if picked is None:
                st.session_state.pop("conversation_id", None)
                st.session_state["messages"] = []
                st.session_state["conversation_meta"] = {}
            else:
                _load_conversation_into_state(int(picked))
            st.rerun()

        # Optional: thread title editor (kept in sidebar to reduce main-panel whitespace)
        conv_id = st.session_state.get("conversation_id")
        if conv_id:
            meta = st.session_state.get("conversation_meta") or {}
            st.divider()
            st.caption("Thread title")
            tcols = st.columns([0.78, 0.22])
            with tcols[0]:
                new_title = st.text_input(
                    "Thread title",
                    value=meta.get("title") or "",
                    key="thread_title_sidebar",
                    label_visibility="collapsed",
                )
            with tcols[1]:
                if st.button("Save", key="save_title_sidebar"):
                    touch_conversation(int(conv_id), title=(new_title.strip() or None))
                    st.session_state["conversation_meta"] = get_conversation(int(conv_id)) or {}
                    st.rerun()

    # # Chat title (ChatGPT-style): show model, and show Thinking if enabled
    # if active_model:
    #     _title = active_model + (" — Thinking" if DEFAULT_THINK else "")
    #     st.markdown(f"# {_title}")

    # Render chat history
    msgs = st.session_state.get("messages", [])
    last_assistant_idx = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)

    for i, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            _render_message(m["role"], m.get("content") or "")
            _render_attachments(m.get("attachments") or [])

            # Per-message copy button (one per message)
            _copy_button(
                m.get("content") or "",
                key=f"copy_msg_{m.get('id','x')}",
                label="⧉",
                css_class="ds330-copy-icon-btn",
                tooltip="Copy message",
            )

            # Copy whole conversation (bottom of last assistant response)
            if i == last_assistant_idx:
                _copy_button(
                    _conversation_to_text(msgs),
                    key=f"copy_conv_{st.session_state.get('conversation_id','new')}",
                    label="Copy Whole Conversation",
                    css_class="ds330-copy-all-btn",
                    tooltip="Copy whole conversation (text only)",
                )

            # Conversation edit controls — ADMIN ONLY
            if is_admin and m["role"] == "user":
                edit_key = f"edit_btn_{m['id']}"
                if st.button("✏️ Edit", key=edit_key, help="Edit this user message and regenerate from here"):
                    st.session_state["editing"] = {
                        "message_id": m["id"],
                        "original": m.get("content") or "",
                        "draft": m.get("content") or "",
                    }
                    st.rerun()

    # Admin edit panel (admin only)
    if is_admin and st.session_state.get("editing"):
        ed = st.session_state["editing"]
        st.info("Admin edit mode: update the message and regenerate from that point.")
        ed["draft"] = st.text_area("Edit user message", value=ed["draft"], height=140)
        bcols = st.columns([0.25, 0.25, 0.5])
        if bcols[0].button("Save & Regenerate", type="primary"):
            conv_id = st.session_state.get("conversation_id")
            if not conv_id:
                st.session_state["editing"] = None
                st.rerun()

            update_message(ed["message_id"], ed["draft"])
            delete_messages_after(int(conv_id), int(ed["message_id"]))

            # Reload from DB
            _load_conversation_into_state(int(conv_id))

            # Regenerate assistant response
            with st.chat_message("assistant"):
                ph = st.empty()
                full = ""
                payload = _build_payload_messages(int(conv_id))
                try:
                    for chunk in chat_stream(
                        host=OLLAMA_HOST,
                        api_key=_effective_api_key(),
                        model=active_model,
                        messages=payload,
                        options=None,
                        think=DEFAULT_THINK,
                    ):
                        full += chunk
                        ph.markdown(full)
                except requests.exceptions.HTTPError as e:
                    st.error("Model request failed. Check that the model exists in Ollama Cloud and that OLLAMA_HOST / OLLAMA_API_KEY are set correctly.")
                    st.caption(f"Host: {OLLAMA_HOST} · Model: {active_model}")
                    with st.expander("Error details", expanded=False):
                        st.code(str(e))
                    return
                except Exception as e:
                    st.error("Unexpected error while calling the model.")
                    with st.expander("Error details", expanded=False):
                        st.code(repr(e))
                    return

            add_message(int(conv_id), "assistant", full)
            touch_conversation(int(conv_id), model=active_model)
            _load_conversation_into_state(int(conv_id))
            st.session_state["editing"] = None
            st.rerun()

        if bcols[1].button("Cancel"):
            st.session_state["editing"] = None
            st.rerun()

    # Chat input with files (ChatGPT-style)
    prompt_val = st.chat_input(
        f"Message {APP_NAME}",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "pdf", "txt", "md", "csv", "json", "docx"],
    )

    if not prompt_val:
        return

    # Streamlit returns either string (older versions) or ChatInputValue (newer)
    if isinstance(prompt_val, str):
        user_text = prompt_val
        files = []
    else:
        user_text = (prompt_val.text or "")
        files = list(prompt_val.files or [])

    # Ensure conversation exists
    conv_id = st.session_state.get("conversation_id")
    if not conv_id:
        base_prompt = get_base_system_prompt(DEFAULT_BASE_PROMPT)
        assignment_prompt = active_assignment.get("prompt") or ""
        sys_prompt = _combined_prompt(base_prompt, assignment_prompt)

        conv_id = create_conversation(
            user_id=user_id,
            role=role,
            title="New conversation",
            model=active_model,
            system_prompt=sys_prompt,
            base_prompt=base_prompt,
            assignment_id=int(active_assignment["id"]),
            assignment_name=active_assignment.get("name"),
            assignment_prompt=assignment_prompt,
        )
        st.session_state["conversation_id"] = conv_id
        st.session_state["conversation_meta"] = get_conversation(int(conv_id)) or {}

    # Persist user message
    user_msg_id = add_message(int(conv_id), "user", user_text)

    # Handle files -> attachments
    attachments: List[Dict[str, Any]] = []
    for f in files:
        raw = f.getvalue()
        mime = getattr(f, "type", "application/octet-stream")
        filename = getattr(f, "name", "file")
        kind = "image" if is_image_mime(mime) else "file"

        text_content = None
        if kind == "file":
            try:
                text_content = extract_text_from_bytes(filename, raw)
            except Exception:
                text_content = None

        add_attachment(
            message_id=user_msg_id,
            kind=kind,
            filename=filename,
            mime=mime,
            data=raw,
            text_content=text_content,
        )

        attachments.append(
            {
                "kind": kind,
                "filename": filename,
                "mime": mime,
                "data": raw,
                "text_content": text_content,
            }
        )

    # Add to UI state
    st.session_state["messages"].append(
        {"id": user_msg_id, "role": "user", "content": user_text, "attachments": attachments}
    )

    # Render user message
    with st.chat_message("user"):
        st.markdown(user_text)
        _render_attachments(attachments)
        # per message copy
        _copy_button(
            user_text,
            key=f"copy_msg_{user_msg_id}",
            label="⧉",
            css_class="ds330-copy-icon-btn",
            tooltip="Copy message",
        )

    # Build payload and stream assistant
    payload = _build_payload_messages(int(conv_id))
    with st.chat_message("assistant"):
        ph = st.empty()
        full = ""
        try:
            for chunk in chat_stream(
                host=OLLAMA_HOST,
                api_key=_effective_api_key(),
                model=active_model,
                messages=payload,
                options=None,
                think=DEFAULT_THINK,
            ):
                full += chunk
                ph.markdown(full)
        except requests.exceptions.HTTPError as e:
            st.error("Model request failed. Check that the model exists in Ollama Cloud and that OLLAMA_HOST / OLLAMA_API_KEY are set correctly.")
            st.caption(f"Host: {OLLAMA_HOST} · Model: {active_model}")
            with st.expander("Error details", expanded=False):
                st.code(str(e))
            return
        except Exception as e:
            st.error("Unexpected error while calling the model.")
            with st.expander("Error details", expanded=False):
                st.code(repr(e))
            return

    asst_id = add_message(int(conv_id), "assistant", full)

    # Update UI state
    st.session_state["messages"].append(
        {"id": asst_id, "role": "assistant", "content": full, "attachments": []}
    )

    touch_conversation(int(conv_id), model=active_model)
    st.rerun()


# ---------------- Admin dashboard ----------------



def _admin_dashboard(active_model: str) -> None:
    st.markdown("### Admin Dashboard")

    tab_prompts, tab_users, tab_convs = st.tabs(["Assignments & Prompts", "Users", "Conversation Browser"])

    # ---------------- Tab: Assignments & Prompts ----------------
    with tab_prompts:
        st.subheader("Assignments")

        active = get_active_assignment()
        assignments = list_assignments()

        if assignments:
            name_by_id = {int(a["id"]): a["name"] for a in assignments}
            ids = list(name_by_id.keys())
            idx = ids.index(int(active["id"])) if int(active["id"]) in ids else 0

            c1, c2 = st.columns([0.60, 0.40])
            with c1:
                new_active = st.selectbox(
                    "Active assignment",
                    ids,
                    index=idx,
                    format_func=lambda i: name_by_id.get(int(i), str(i)),
                )
                if int(new_active) != int(active["id"]):
                    set_active_assignment(int(new_active))
                    st.rerun()

                st.caption(f"Currently active: **{get_active_assignment().get('name')}**")

            with c2:
                with st.expander("➕ Add new assignment", expanded=False):
                    new_name = st.text_input("Assignment name", placeholder="Assignment 2")
                    new_prompt = st.text_area("Assignment-specific prompt (optional)", height=180)
                    if st.button("Create assignment", key="create_assignment_btn"):
                        if not new_name.strip():
                            st.error("Assignment name is required")
                        else:
                            aid = add_assignment(new_name.strip(), new_prompt or "")
                            set_active_assignment(aid)
                            st.success(f"Created and activated '{new_name.strip()}'")
                            st.rerun()
        else:
            st.warning("No assignments found. Create your first assignment below.")
            new_name = st.text_input("Assignment name", placeholder="Assignment 1")
            new_prompt = st.text_area("Assignment-specific prompt (optional)", height=180)
            if st.button("Create assignment", key="create_assignment_btn_empty"):
                if not new_name.strip():
                    st.error("Assignment name is required")
                else:
                    aid = add_assignment(new_name.strip(), new_prompt or "")
                    set_active_assignment(aid)
                    st.success(f"Created and activated '{new_name.strip()}'")
                    st.rerun()

        st.divider()
        st.subheader("System Prompts")

        active = get_active_assignment()
        header = active.get("name") or "(no assignment)"
        st.caption(f"Editing prompts for: **{header}**")

        p1, p2 = st.columns(2, gap="large")

        with p1:
            base_prompt = get_base_system_prompt(DEFAULT_BASE_PROMPT)
            base_edit = st.text_area(
                "Base system prompt (applies to ALL assignments)",
                value=base_prompt,
                height=520,
            )
            if st.button("Save base prompt", type="primary", key="save_base_prompt_btn"):
                set_base_system_prompt(base_edit)
                st.success("Saved base system prompt")

        with p2:
            ap_edit = st.text_area(
                "Assignment-specific prompt (applies only to this assignment)",
                value=active.get("prompt") or "",
                height=520,
            )
            if st.button("Save assignment prompt", type="primary", key="save_assignment_prompt_btn"):
                update_assignment_prompt(int(active["id"]), ap_edit)
                st.success("Saved assignment prompt")

    # ---------------- Tab: Users ----------------
    with tab_users:
        st.subheader("Users")
        users = list_users()
        try:
            st.dataframe(users, width="stretch", hide_index=True)
        except TypeError:
            st.dataframe(users, use_container_width=True, hide_index=True)

        with st.expander("Add / Update user", expanded=False):
            uid = st.text_input("User ID", key="new_uid")
            pw = st.text_input("Password", type="password", key="new_pw")
            role = st.selectbox("Role", ["student", "admin"], key="new_role")
            if st.button("Save user", key="save_user_btn"):
                if not uid or not pw:
                    st.error("User ID and password are required.")
                else:
                    upsert_user(uid, pw, role)
                    st.success("User saved")
                    st.rerun()

    # ---------------- Tab: Conversation Browser ----------------
    with tab_convs:
        st.subheader("Conversation browser")

        c1, c2, c3, c4 = st.columns([0.34, 0.24, 0.20, 0.22])
        with c1:
            users = list_users()
            if users:
                user_ids = [u.get("user_id") for u in users if u.get("user_id")]
                user_opts = [None] + sorted(set([str(u) for u in user_ids]))
                picked_user = st.selectbox(
                    "Filter by user_id",
                    user_opts,
                    index=0,
                    format_func=lambda u: "All users" if u is None else str(u),
                    key="admin_user_filter_select",
                )
                # list_conversations_admin uses a contains filter; passing the full ID yields an exact match.
                user_filter = "" if picked_user is None else str(picked_user)
            else:
                st.selectbox(
                    "Filter by user_id",
                    ["All users"],
                    index=0,
                    disabled=True,
                    key="admin_user_filter_select_empty",
                )
                user_filter = ""
        with c2:
            model_filter = st.text_input("Filter by model (exact)", "")
        with c3:
            role_filter = st.selectbox("Filter by role", ["", "student", "admin"], index=0)
        with c4:
            assignments = list_assignments()
            if assignments:
                id_to_name = {int(a["id"]): a["name"] for a in assignments}
                opts = [None] + list(id_to_name.keys())
                assignment_filter = st.selectbox(
                    "Filter by assignment",
                    opts,
                    index=0,
                    format_func=lambda i: "All assignments" if i is None else id_to_name.get(int(i), str(i)),
                )
            else:
                assignment_filter = None
                st.selectbox("Filter by assignment", ["All assignments"], index=0, disabled=True)

        
        assignment_id = (int(assignment_filter) if assignment_filter is not None else None)

        try:
            convs = list_conversations_admin(
                user_filter=user_filter or None,
                role_filter=role_filter or None,
                model_filter=model_filter or None,
                assignment_id_filter=assignment_id,
                limit=200,
            )
        except TypeError:
            # Back-compat: older storage.py may not support assignment_id_filter.
            convs = list_conversations_admin(
                user_filter=user_filter or None,
                role_filter=role_filter or None,
                model_filter=model_filter or None,
                limit=200,
            )
            if assignment_id is not None and assignments:
                wanted_name = id_to_name.get(int(assignment_id))

                def _matches_assignment(c: Dict[str, Any]) -> bool:
                    # Prefer explicit fields if present
                    try:
                        if c.get("assignment_id") is not None and int(c.get("assignment_id")) == int(assignment_id):
                            return True
                    except Exception:
                        pass
                    if wanted_name and (c.get("assignment_name") == wanted_name):
                        return True

                    # Fallback: parse title suffix "(Assignment X)"
                    t = (c.get("title") or "").strip()
                    m = re.search(r"\(([^()]+)\)\s*$", t)
                    if wanted_name and m and m.group(1).strip() == wanted_name:
                        return True

                    # Final fallback: treat missing assignment as Assignment 1
                    return bool(wanted_name == "Assignment 1")

                convs = [c for c in convs if _matches_assignment(c)]


        if not convs:
            st.caption("No conversations match filters.")
            return

        conv_ids = [c["id"] for c in convs]
        labels = {}
        for c in convs:
            cid = c["id"]
            title = c.get("title") or f"Conversation {cid}"
            labels[cid] = f"{c.get('updated_at','')} · {c.get('user_id','')} · {title}"

        picked = st.selectbox("Select conversation", conv_ids, format_func=lambda x: labels.get(x, str(x)))
        conv = get_conversation(int(picked)) or {}
        msgs = get_conversation_messages(int(picked))

        st.caption(f"Model: **{conv.get('model')}** · Assignment: **{conv.get('assignment_name') or '—'}**")

        # Render the conversation as-is (including images), and offer a transcript download.
        mids = [int(m["id"]) for m in msgs]
        att_map = list_attachments_for_message_ids(mids)

        transcript = _conversation_to_text([{"role": m["role"], "content": m.get("content") or ""} for m in msgs])

        st.download_button(
            "download transcript",
            data=transcript,
            file_name=f"conversation_{picked}.txt",
            mime="text/plain",
            help="Downloads a text-only transcript (images excluded).",
        )

        st.divider()
        for m in msgs:
            with st.chat_message(m["role"]):
                _render_message(m["role"], m.get("content") or "")
                _render_attachments(att_map.get(int(m["id"]), []) or [])


# ---------------- Main routing


def main() -> None:
    if "user_id" not in st.session_state:
        _render_login()
        return

    models = _cached_models()
    page, active_model, active_assignment = _sidebar(models)

    if page == "Admin Dashboard" and st.session_state.get("role") == "admin":
        _admin_dashboard(active_model)
    else:
        _chat_page(active_model, active_assignment)


if __name__ == "__main__":
    main()
