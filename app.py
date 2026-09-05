import asyncio
import json
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import deepseek_tokenizer
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

load_dotenv()

API_KEY = os.getenv("DEEPSEEKER_API_KEY", "dseeker")
ADMIN_USER = os.getenv("DEEPSEEKER_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("DEEPSEEKER_ADMIN_PASSWORD", "admin")
COOKIE_FILE = os.path.join(os.getenv("DEEPSEEKER_DATA_DIR", ""), "aws_cookies_deepseek.json") if os.getenv("DEEPSEEKER_DATA_DIR", "").strip() else "aws_cookies_deepseek.json"
DEFAULT_MODEL_ID = "DeepSeek-V4-Flash-Vision-Exp"

security = HTTPBasic()


from functions import (
    add_token,
    count_tokens,
    create_new_chat,
    delete_token,
    find_session,
    get_auth_token,
    get_token,
    get_tokens,
    init_db,
    mark_limited,
    mark_active,
    parse_tools,
    pick_token,
    save_session,
    delete_session,
    send_message,
    StreamToolParser,
    upload_file,
    get_file_content,
)
from plugin_helper import build_prompt, extract_and_upload_files, generate_signature, generate_signature_sync


logger = logging.getLogger("uvicorn.error")


def count_tok(text):
    return len(deepseek_tokenizer.ds_token.encode(text))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="DeeperSeeker", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length", "")
    if cl.isdigit() and int(cl) > 32 * 1024 * 1024:
        return JSONResponse({"error": "Request body too large"}, status_code=413)
    return await call_next(request)


SESSIONS = {}
SESSION_TTL = 7 * 24 * 3600
_sig_locks = {}
_login_fails = {"count": 0, "locked_until": 0}
RESPONSE_STORE = {}
RESPONSE_TTL = 7 * 24 * 3600
RESPONSE_CANCELLED = set()


def _prune_response_store():
    cutoff = time.time() - RESPONSE_TTL
    for response_id, item in list(RESPONSE_STORE.items()):
        if item.get("created_at", 0) < cutoff:
            RESPONSE_STORE.pop(response_id, None)


def _store_response(response_id, response, messages=None, persist=True):
    """Keep public response data separate from the private conversation state."""
    if not persist:
        return
    _prune_response_store()
    public_response = dict(response)
    public_response.pop("_messages", None)
    RESPONSE_STORE[response_id] = {
        "created_at": time.time(),
        "response": public_response,
        "messages": list(messages or []),
    }


def _get_stored_response(response_id):
    _prune_response_store()
    item = RESPONSE_STORE.get(response_id)
    return item["response"] if item else None


def _get_response_state(response_id):
    _prune_response_store()
    return RESPONSE_STORE.get(response_id)


def _update_stored_response(response_id, response, messages=None, persist=True):
    if not persist:
        return
    _store_response(response_id, response, messages=messages)


def _response_error(message, code="server_error", status_code=400, param=None):
    error = {"message": message, "type": code}
    if param:
        error["param"] = param
    return JSONResponse({"error": error}, status_code=status_code)


def _messages_equal(left, right):
    try:
        return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(right, sort_keys=True, ensure_ascii=False)
    except Exception:
        return left == right


def _append_response_history(previous, current):
    if not previous:
        return current
    if not current:
        return previous
    if len(current) >= len(previous) and all(_messages_equal(a, b) for a, b in zip(current[:len(previous)], previous)):
        return current
    return previous + current


def get_current_admin(request: Request):
    sid = request.cookies.get("session_id")
    if not sid or sid not in SESSIONS or time.time() - SESSIONS[sid] > SESSION_TTL:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    SESSIONS[sid] = time.time()
    origin = request.headers.get("origin", "")
    if origin:
        parsed = urlparse(origin).netloc
        if parsed and parsed != request.headers.get("host", ""):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return "admin"


def get_api_key(request: Request):
    auth = request.headers.get("authorization", "")
    api_key_header = request.headers.get("x-api-key", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    elif auth:
        return auth
    return api_key_header


def check_key(request: Request):
    key = get_api_key(request)
    return secrets.compare_digest(key.encode("utf-8"), API_KEY.encode("utf-8"))


async def handle_chat(messages, model, thinking=False, search=False, stream=False, tools=None, is_anthropic=False, req_model=None, scope="", _retried=False):
    auth_token = get_auth_token()
    if not auth_token:
        return JSONResponse({"error": "No auth token. Add via dashboard."}, status_code=401)

    sig = await generate_signature(messages, model, scope)
    sess = find_session(sig)
    if not sess:
        for i in range(len(messages) - 1, 0, -1):
            sess = find_session(generate_signature_sync(messages[:i], model, scope))
            if sess:
                break

    logger.info(
        "chat request model=%s stream=%s messages=%d session=%s parent=%s",
        model,
        stream,
        len(messages),
        "matched" if sess else "new",
        sess.get("parent_message_id") if sess else 0,
    )

    if sess:

        token_id = sess["token_id"]
        session_id = sess["session_id"]
        parent_message_id = sess["parent_message_id"]
        tok = get_token(token_id)
        if not tok or tok["status"] == "RATE_LIMITED":
            new_token_id = pick_token()
            if new_token_id and (not tok or new_token_id != token_id):
                new_tok = get_token(new_token_id)
                if new_tok:
                    new_session_id = await create_new_chat(new_tok["token"])
                    prompt = await build_prompt(messages, tools or [], model, is_first_message=True)

                    file_ids = await extract_and_upload_files(messages, new_tok["token"])
                    gen = send_message(new_session_id, new_tok["token"], prompt, 0, thinking, search, None if model == "instant" else model, file_ids)
                    if stream:
                        if is_anthropic:
                            return StreamingResponse(stream_anthropic_response(gen, model, messages, new_token_id, new_session_id, sig, tools, req_model, 0, scope), media_type="text/event-stream")
                        return StreamingResponse(stream_response(gen, model, messages, new_token_id, new_session_id, sig, tools, 0, scope), media_type="text/event-stream")
                    else:
                        resp_text = await collect_response(gen)
                        mark_active(new_token_id)

                        parsed_tools, clean_text = parse_tools(resp_text)
                        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
                        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()
                        next_messages = messages.copy()
                        ast_msg = {"role": "assistant"}
                        if parsed_tools:
                            ast_msg["tool_calls"] = parsed_tools
                        else:
                            ast_msg["content"] = clean_text
                        next_messages.append(ast_msg)
                        next_sig = await generate_signature(next_messages, model, scope)

                        save_session(sig, new_token_id, new_session_id, 2)
                        save_session(next_sig, new_token_id, new_session_id, 2)
                        return format_response(resp_text, model, messages, tools)
    else:

        create_lock = _sig_locks.setdefault(sig, asyncio.Lock())
        async with create_lock:
            sess = find_session(sig)
            if not sess:
                token_id = pick_token()
                if not token_id:
                    return JSONResponse({"error": "No tokens available"}, status_code=503)
                tok = get_token(token_id)
                if not tok:
                    return JSONResponse({"error": "Token not found"}, status_code=503)
                session_id = await create_new_chat(tok["token"])
                save_session(sig, token_id, session_id, 0)
                parent_message_id = 0
            else:
                token_id = sess["token_id"]
                session_id = sess["session_id"]
                parent_message_id = sess["parent_message_id"]

    tok = get_token(token_id)
    if not tok:
        return JSONResponse({"error": "Token expired"}, status_code=503)

    is_first = parent_message_id == 0
    file_ids = await extract_and_upload_files(messages, tok["token"], last_user_only=not is_first)
    prompt = await build_prompt(messages, tools or [], model, is_first)

    try:
        gen = send_message(session_id, tok["token"], prompt, parent_message_id, thinking, search, None if model == "instant" else model, file_ids)
        if stream:
            if is_anthropic:
                return StreamingResponse(stream_anthropic_response(gen, model, messages, token_id, session_id, sig, tools, req_model, parent_message_id, scope), media_type="text/event-stream")
            return StreamingResponse(stream_response(gen, model, messages, token_id, session_id, sig, tools, parent_message_id, scope), media_type="text/event-stream")
        else:
            resp_text = await collect_response(gen)
            mark_active(token_id)

            parsed_tools, clean_text = parse_tools(resp_text)
            clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
            clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = await generate_signature(next_messages, model, scope)

            save_session(sig, token_id, session_id, parent_message_id + 2)
            save_session(next_sig, token_id, session_id, parent_message_id + 2)
            return format_response(resp_text, model, messages, tools)
    except Exception as e:
        delete_session(sig)
        m = re.match(r"HTTP (\d{3}):", str(e))
        code = int(m.group(1)) if m else None
        if code in (401, 403, 429):
            mark_limited(token_id)
        if _retried or code not in (401, 403, 429):
            raise
        return await handle_chat(messages, model, thinking, search, stream, tools, is_anthropic, req_model, scope, _retried=True)


async def collect_response(gen):
    text = ""
    async for chunk in gen:
        text += chunk
    return text


def _messages_text(messages):
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            parts.append(" ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"))
        else:
            parts.append(str(c))
    return "\n".join(parts)


async def _hold_think_tags(gen):
    carry = ""
    async for chunk in gen:
        chunk = carry + chunk
        carry = ""
        hold = 0
        for tag in ("<think>", "</think>"):
            for i in range(1, len(tag)):
                if chunk.endswith(tag[:i]):
                    hold = max(hold, i)
        if hold:
            carry = chunk[-hold:]
            chunk = chunk[:-hold]
        if chunk:
            yield chunk
    if carry:
        yield carry


async def stream_response(gen, model, messages, token_id, session_id, sig, tools, parent_message_id=0, scope=""):
    parser = StreamToolParser()
    full_text = ""
    is_thinking = False
    aborted = False
    failed = False
    try:
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "").lstrip("\n")

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': think_part}}]})}\n\n"

            if is_thinking and chunk:
                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': chunk}}]})}\n\n"
                continue

            if end_thinking and not chunk:
                continue

            for r in parser.feed(chunk):
                if "text" in r:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': r['text']}}]})}\n\n"
        mark_active(token_id)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        raise
    except Exception as e:
        failed = True
        m = re.match(r"HTTP (\d{3}):", str(e))
        code = int(m.group(1)) if m else None
        if code in (401, 403, 429):
            mark_limited(token_id)
        logger.exception("stream_response failed")
        try:
            yield f"data: {json.dumps({'error': {'message': str(e)[:300]}})}\n\n"
        except Exception:
            pass
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            save_session(sig, token_id, session_id, parent_message_id + 2)
            save_session(next_sig, token_id, session_id, parent_message_id + 2)

        if not aborted and not failed:
            try:
                if not parsed_tools:
                    for r in parser.flush():
                        if "text" in r:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': r['text']}}]})}\n\n"

                if parsed_tools:
                    for i, tc in enumerate(parsed_tools):
                        delta_tc = {"index": i, "id": tc["id"], "type": "function",
                                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                        yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [delta_tc]}}]})}\n\n"
                    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
                else:
                    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                pass


def _responses_event(event_type, payload, sequence_number=None):
    payload = {"type": event_type, **payload}
    if sequence_number is not None:
        payload["sequence_number"] = sequence_number
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_responses_response(body_iterator, model, messages, requested_model=None, response_id=None, persist=True, response_options=None):
    """Translate the internal stream into a stateful OpenAI Responses SSE stream."""
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    requested_model = requested_model or model
    response_options = response_options or {}
    full_text = ""
    reasoning_text = ""
    stream_error = None
    cancelled = False
    tool_calls = {}
    # Responses SSE sequence numbers are zero-based and strictly increasing.
    sequence_number = -1
    reasoning_id = None
    text_id = None
    text_output_index = None
    tool_output_indexes = {}
    used_output_indexes = set()
    sse_buffer = ""

    def event(event_type, payload):
        nonlocal sequence_number
        sequence_number += 1
        return _responses_event(event_type, payload, sequence_number)

    response_base = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "model": requested_model,
        "output": [],
        "usage": None,
    }
    for key in ("previous_response_id", "instructions", "metadata", "tools", "tool_choice", "parallel_tool_calls", "temperature", "top_p", "text", "reasoning", "store"):
        if key in response_options and response_options[key] is not None:
            response_base[key] = response_options[key]
    _store_response(response_id, response_base, messages=messages, persist=persist)
    yield event("response.created", {"response": response_base})
    yield event("response.in_progress", {"response": response_base})

    async def process_block(block):
        nonlocal full_text, reasoning_text, stream_error, reasoning_id, text_id, text_output_index
        data_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if not data_lines:
            return
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return

        if chunk.get("error"):
            stream_error = chunk["error"]
            return
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            text_delta = delta.get("content")
            if text_delta:
                if text_id is None:
                    text_id = f"msg_{uuid.uuid4().hex[:24]}"
                    text_output_index = 0
                    while text_output_index in used_output_indexes:
                        text_output_index += 1
                    used_output_indexes.add(text_output_index)
                    yield event("response.output_item.added", {
                        "output_index": text_output_index,
                        "item": {"id": text_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
                    })
                    yield event("response.content_part.added", {
                        "item_id": text_id,
                        "output_index": text_output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    })
                full_text += text_delta
                yield event("response.output_text.delta", {
                    "item_id": text_id,
                    "output_index": text_output_index,
                    "content_index": 0,
                    "delta": text_delta,
                })
            reasoning_delta = delta.get("reasoning_content")
            if reasoning_delta:
                if reasoning_id is None:
                    reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
                    used_output_indexes.add(0)
                    yield event("response.output_item.added", {
                        "output_index": 0,
                        "item": {"id": reasoning_id, "type": "reasoning", "status": "in_progress", "summary": []},
                    })
                    yield event("response.reasoning_summary_part.added", {
                        "item_id": reasoning_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    })
                reasoning_text += reasoning_delta
                yield event("response.reasoning_summary_text.delta", {
                    "item_id": reasoning_id,
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": reasoning_delta,
                })
            for tool_delta in delta.get("tool_calls") or []:
                tool_index = int(tool_delta.get("index", len(tool_calls)))
                function = tool_delta.get("function") or {}
                call = tool_calls.get(tool_index)
                if call is None:
                    call = {
                        "id": tool_delta.get("id") or f"fc_{uuid.uuid4().hex}",
                        "call_id": tool_delta.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "name": function.get("name", ""),
                        "arguments": "",
                    }
                    tool_calls[tool_index] = call
                    tool_output_indexes[tool_index] = 0
                    while tool_output_indexes[tool_index] in used_output_indexes:
                        tool_output_indexes[tool_index] += 1
                    used_output_indexes.add(tool_output_indexes[tool_index])
                    yield event("response.output_item.added", {
                        "output_index": tool_output_indexes[tool_index],
                        "item": {"id": call["id"], "type": "function_call", "status": "in_progress", "call_id": call["call_id"], "name": call["name"], "arguments": ""},
                    })
                elif function.get("name") and not call["name"]:
                    call["name"] = function["name"]
                arguments_delta = function.get("arguments")
                if arguments_delta:
                    if not isinstance(arguments_delta, str):
                        arguments_delta = json.dumps(arguments_delta, ensure_ascii=False)
                    call["arguments"] += arguments_delta
                    yield event("response.function_call_arguments.delta", {
                        "item_id": call["id"],
                        "output_index": tool_output_indexes[tool_index],
                        "delta": arguments_delta,
                    })
            if choice.get("finish_reason") == "error":
                stream_error = {"message": "Upstream stream failed"}

    try:
        async for raw_event in body_iterator:
            if response_id in RESPONSE_CANCELLED:
                cancelled = True
                break
            if isinstance(raw_event, bytes):
                raw_event = raw_event.decode("utf-8", errors="replace")
            sse_buffer += str(raw_event)
            while "\n\n" in sse_buffer:
                block, sse_buffer = sse_buffer.split("\n\n", 1)
                async for output in process_block(block):
                    yield output
        if not cancelled and sse_buffer.strip():
            async for output in process_block(sse_buffer):
                yield output
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        cancelled_response = {**response_base, "status": "cancelled", "incomplete_details": {"reason": "cancelled"}}
        _update_stored_response(response_id, cancelled_response, messages=messages, persist=persist)
        RESPONSE_CANCELLED.discard(response_id)
        raise
    except Exception as exc:
        stream_error = {"message": str(exc)[:300]}

    if cancelled:
        cancelled_response = {**response_base, "status": "cancelled", "incomplete_details": {"reason": "cancelled"}}
        _update_stored_response(response_id, cancelled_response, messages=messages, persist=persist)
        RESPONSE_CANCELLED.discard(response_id)
        yield event("response.completed", {"response": cancelled_response})
        return

    if stream_error:
        failed_response = {**response_base, "status": "failed", "error": stream_error}
        _update_stored_response(response_id, failed_response, messages=messages, persist=persist)
        yield event("response.failed", {"response": failed_response})
        return

    response_output = []
    if reasoning_id:
        yield event("response.reasoning_summary_text.done", {"item_id": reasoning_id, "output_index": 0, "summary_index": 0, "text": reasoning_text})
        yield event("response.reasoning_summary_part.done", {"item_id": reasoning_id, "output_index": 0, "summary_index": 0, "part": {"type": "summary_text", "text": reasoning_text}})
        reasoning_item = {"id": reasoning_id, "type": "reasoning", "status": "completed", "summary": [{"type": "summary_text", "text": reasoning_text}]}
        response_output.append(reasoning_item)
        yield event("response.output_item.done", {"output_index": 0, "item": reasoning_item})
    if text_id:
        text_item = {"id": text_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": full_text, "annotations": []}]}
        yield event("response.output_text.done", {"item_id": text_id, "output_index": text_output_index, "content_index": 0, "text": full_text})
        yield event("response.content_part.done", {"item_id": text_id, "output_index": text_output_index, "content_index": 0, "part": {"type": "output_text", "text": full_text, "annotations": []}})
        response_output.append(text_item)
        yield event("response.output_item.done", {"output_index": text_output_index, "item": text_item})
    for tool_index in sorted(tool_calls):
        call = tool_calls[tool_index]
        yield event("response.function_call_arguments.done", {
            "item_id": call["id"],
            "output_index": tool_output_indexes[tool_index],
            "arguments": call["arguments"],
        })
        function_item = {
            "id": call["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": call["call_id"],
            "name": call["name"],
            "arguments": call["arguments"],
        }
        response_output.append(function_item)
        yield event("response.output_item.done", {
            "output_index": tool_output_indexes[tool_index],
            "item": function_item,
        })
    if not response_output:
        empty_id = f"msg_{uuid.uuid4().hex[:24]}"
        empty_item = {"id": empty_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "", "annotations": []}]}
        yield event("response.output_item.added", {"output_index": 0, "item": {**empty_item, "status": "in_progress", "content": []}})
        yield event("response.content_part.added", {"item_id": empty_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
        yield event("response.output_text.done", {"item_id": empty_id, "output_index": 0, "content_index": 0, "text": ""})
        yield event("response.content_part.done", {"item_id": empty_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
        yield event("response.output_item.done", {"output_index": 0, "item": empty_item})
        response_output.append(empty_item)
    input_tokens = count_tok(_messages_text(messages))
    output_tokens = count_tok(full_text or reasoning_text)
    completed_response = {
        **response_base,
        "status": "completed",
        "model": requested_model,
        "output": response_output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    assistant_message = {"role": "assistant", "content": full_text if full_text else None}
    if tool_calls:
        assistant_message["tool_calls"] = [
            {"id": call["call_id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            for _, call in sorted(tool_calls.items())
        ]
    stored_messages = messages + [assistant_message]
    _update_stored_response(response_id, completed_response, messages=stored_messages, persist=persist)
    yield event("response.completed", {"response": completed_response})


async def stream_anthropic_response(gen, model, messages, token_id, session_id, sig, tools, req_model=None, parent_message_id=0, scope=""):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    in_tokens = count_tok(_messages_text(messages))
    model_name = req_model if req_model else model
    start_evt = f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': in_tokens, 'output_tokens': 1}}})}\n\n"
    yield start_evt

    parser = StreamToolParser()
    full_text = ""
    text_block_started = False
    block_index = 0
    aborted = False
    failed = False

    try:
        is_thinking = False
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "").lstrip("\n")
                start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking'}})}\n\n"
                yield start_block

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': think_part}})}\n\n"
                    yield delta_evt

            if is_thinking and chunk:
                delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': chunk}})}\n\n"
                yield delta_evt
                continue

            if end_thinking:
                stop_evt = f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                yield stop_evt
                block_index += 1
                if not chunk:
                    continue

            for r in parser.feed(chunk):
                if "text" in r:
                    if not text_block_started:
                        start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield start_block
                        text_block_started = True
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': r['text']}})}\n\n"
                    yield delta_evt
        mark_active(token_id)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        raise
    except Exception as e:
        failed = True
        m = re.match(r"HTTP (\d{3}):", str(e))
        code = int(m.group(1)) if m else None
        if code in (401, 403, 429):
            mark_limited(token_id)
        logger.exception("stream_anthropic_response failed")
        try:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)[:300]}})}\n\n"
        except Exception:
            pass
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        out_tokens = count_tok(full_text)

        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            save_session(sig, token_id, session_id, parent_message_id + 2)
            save_session(next_sig, token_id, session_id, parent_message_id + 2)

        def _tb(text):
            return (f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                    f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n")

        block_index_local = [block_index]
        tail_events = ""
        if is_thinking:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
            block_index_local[0] += 1

        flushed_text = ""
        if not parsed_tools:
            for r in parser.flush():
                if "text" in r:
                    flushed_text += r["text"]

        if not text_block_started and not parsed_tools and (clean_text or flushed_text):
            tail_events += _tb(clean_text or flushed_text)
        elif text_block_started and not parsed_tools:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"

        if parsed_tools:
            for tc in parsed_tools:
                tool_input = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                json_str = json.dumps(tool_input)
                tail_events += f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': tc['function']['name'], 'input': {}}})}\n\n"
                tail_events += f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'input_json_delta', 'partial_json': json_str}})}\n\n"
                tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
                block_index_local[0] += 1
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        else:
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        tail_events += f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        if not aborted and not failed:
            try:
                for evt in tail_events.split("\n\n"):
                    if evt.strip():
                        yield evt + "\n\n"
            except asyncio.CancelledError:
                pass


def format_response(text, model, messages, tools=None):
    from functions import DEEPSEEK_TARIFFS
    parsed_tools, clean_text = parse_tools(text)

    reasoning = None
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
    clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

    in_tokens = count_tok(_messages_text(messages))
    out_tokens = count_tok(text)
    tariff_key = "deepseek-v4-pro" if model == "expert" else "deepseek-v4-flash"
    tariff = DEEPSEEK_TARIFFS[tariff_key]
    cost = (in_tokens / 1_000_000 * tariff["cache_miss_input"]) + (out_tokens / 1_000_000 * tariff["output_generation"])

    msg_dict = {
        "role": "assistant",
        "content": clean_text if not parsed_tools else None,
        "tool_calls": parsed_tools if parsed_tools else None,
    }
    if reasoning:
        msg_dict["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg_dict,
            "finish_reason": "tool_calls" if parsed_tools else "stop",
        }],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "cost": round(cost, 6),
        },
    }


format_openai_response = format_response


def format_anthropic_response(result, model):
    choice = result["choices"][0]
    msg = choice["message"]
    ant_content = []

    if msg.get("reasoning_content"):
        ant_content.append({"type": "thinking", "thinking": msg["reasoning_content"]})

    if msg.get("content"):
        ant_content.append({"type": "text", "text": msg["content"]})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"]
            tool_input = json.loads(args) if isinstance(args, str) else args
            ant_content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": tool_input,
            })
    usage = result.get("usage", {})
    msg_id = result["id"]
    if not msg_id.startswith("msg_"):
        msg_id = f"msg_{msg_id.replace('chatcmpl-', '')}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": ant_content,
        "model": model,
        "stop_reason": "tool_use" if msg.get("tool_calls") else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }



@app.post("/v1/files")
@app.post("/v1/files/upload")
async def files_upload(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    tok_id = pick_token()
    if not tok_id:
        return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = get_token(tok_id)
    form = await request.form()
    file_obj = form.get("file")
    if not file_obj:
        return JSONResponse({"error": "No file provided"}, status_code=400)
    file_bytes = await file_obj.read(25 * 1024 * 1024 + 1)
    if len(file_bytes) > 25 * 1024 * 1024:
        return JSONResponse({"error": "File too large"}, status_code=413)
    filename = getattr(file_obj, "filename", "file.bin")
    content_type = getattr(file_obj, "content_type", "application/octet-stream")
    file_info = None
    async for status, data in upload_file(file_bytes, filename, content_type, tok["token"]):
        if status == "success":
            file_info = data
            break
    if not file_info:
        return JSONResponse({"error": "Upload failed"}, status_code=500)

    if request.url.path.startswith("/v1/files/upload"):
        return {
            "id": file_info["file_id"],
            "type": "file",
            "filename": filename,
            "size": file_info["size"],
            "created_at": file_info["anthropic_timestamp"],
        }
    return {
        "id": file_info["file_id"],
        "object": "file",
        "bytes": file_info["size"],
        "created_at": file_info["openai_timestamp"],
        "filename": filename,
        "purpose": "answers",
    }


@app.get("/v1/files/{file_id}/content")
@app.get("/v1/files/{file_id}")
async def files_content(file_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    tok_id = pick_token()
    if not tok_id:
        return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = get_token(tok_id)
    gen = get_file_content(tok["token"], file_id)
    try:
        mime = await gen.__anext__()
    except StopAsyncIteration:
        return JSONResponse({"error": "File not found"}, status_code=404)
    except Exception:
        return JSONResponse({"error": "File fetch failed"}, status_code=502)
    async def stream_chunks():
        async for chunk in gen:
            yield chunk
    return StreamingResponse(stream_chunks(), media_type=mime or "application/octet-stream")


def is_thinking_enabled(body, request=None):
    effort = body.get("effort")
    if effort is not None:
        e_str = str(effort).strip().lower()
        if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
            return True
        if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
            return False

    out_cfg = body.get("output_config")
    if isinstance(out_cfg, dict):
        out_effort = out_cfg.get("effort") or out_cfg.get("reasoning_effort")
        if out_effort is not None:
            e_str = str(out_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                return True
            if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
                return False

    thinking_val = body.get("thinking")
    if isinstance(thinking_val, dict):
        t_type = str(thinking_val.get("type", "")).strip().lower()
        if t_type in ["enabled", "adaptive", "true"]:
            return True
        if t_type == "disabled":
            return False
        budget = thinking_val.get("budget_tokens", 0)
        if isinstance(budget, (int, float)) and budget > 0:
            return True
        t_effort = thinking_val.get("effort") or thinking_val.get("reasoning_effort") or thinking_val.get("level")
        if t_effort is not None:
            e_str = str(t_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                return True
            if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
                return False
    elif isinstance(thinking_val, str):
        t_str = thinking_val.strip().lower()
        if t_str in ["medium", "high", "max", "ultra", "extreme", "true", "enabled", "adaptive", "on"]:
            return True
        if t_str in ["low", "none", "off", "disable", "disabled", "false"]:
            return False
    elif isinstance(thinking_val, bool):
        return thinking_val

    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort is not None:
        effort_str = str(reasoning_effort).strip().lower()
        if effort_str in ["medium", "high", "max", "ultra", "extreme"]:
            return True
        if effort_str in ["low", "none", "off", "disable", "disabled"]:
            return False

    if request:
        req_effort = request.headers.get("anthropic-thinking") or request.headers.get("x-anthropic-thinking") or request.headers.get("effort") or request.headers.get("x-effort")
        if req_effort:
            e_str = str(req_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                return True
    return False


def resolve_model(model_raw, body=None, request=None):
    if not model_raw or not isinstance(model_raw, str):
        return "vision"
    m = model_raw.strip().lower().replace("_", "-")
    body = body or {}
    mode = body.get("mode") or body.get("speed") or body.get("model_type")
    if isinstance(mode, str) and mode.strip().lower() in {"fast", "quick", "instant", "lite"}:
        return "vision"
    if body.get("fast") is True or body.get("quick") is True:
        return "vision"
    if request is not None:
        header_mode = request.headers.get("x-model-mode") or request.headers.get("x-deepseek-mode")
        if header_mode and header_mode.strip().lower() in {"fast", "quick", "instant", "lite"}:
            return "vision"
    effort = str(body.get("reasoning_effort", "")).strip().lower()
    if effort in {"none", "off", "low"} and (m.startswith("gpt-") or "codex" in m):
        return "vision"
    # Keep only the current V4 model family; legacy V3 aliases are intentionally unsupported.
    fast_aliases = (
        "instant", "flash", "haiku", "fast", "quick", "lite", "mini",
        "deepseek-v4-flash", "gemini-flash",
    )
    if "flash-vision-exp" in m or ("vision" in m and "flash" in m):
        return "vision"
    if any(alias in m for alias in fast_aliases):
        return "instant"
    if "vision" in m:
        return "vision"
    if any(alias in m for alias in ("expert", "pro", "reasoner", "reasoning", "opus")):
        return "expert"
    # Unknown provider aliases use the configured default V4 Flash Vision model.
    return "vision"


def _responses_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        if item_type in {"input_text", "output_text", "text"}:
            text = item.get("text", "")
            if text:
                parts.append({"type": "text", "text": str(text)})
        elif item_type in {"input_image", "image"}:
            image_url = item.get("image_url") or item.get("url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not image_url and item.get("file_id"):
                parts.append({"type": "file", "file_id": item.get("file_id")})
                continue
            if image_url:
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
        elif item_type in {"input_file", "file"}:
            file_part = {key: item.get(key) for key in ("file_id", "file_data", "file_url", "filename") if item.get(key) is not None}
            if file_part:
                parts.append({"type": "file", **file_part})
        elif item_type == "refusal":
            refusal = item.get("refusal") or item.get("text", "")
            if refusal:
                parts.append({"type": "text", "text": str(refusal)})
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts


def _responses_value_to_text(value):
    """Convert Responses content parts into the text form accepted by DeepSeek."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        content = _responses_content(value)
        if isinstance(content, str):
            return content
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text")
    return "" if value is None else str(value)


def _responses_input_messages(body):
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": _responses_value_to_text(instructions)})

    inputs = body.get("input", [])
    if isinstance(inputs, (str, dict)):
        inputs = [inputs]
    if not isinstance(inputs, list):
        inputs = []

    pending_parts = []

    def flush_pending():
        if pending_parts:
            messages.append({"role": "user", "content": _responses_content(pending_parts)})
            pending_parts.clear()

    for item in inputs:
        if isinstance(item, str):
            flush_pending()
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type in {"input_text", "input_image", "input_file"} and "role" not in item:
            pending_parts.append(item)
            continue
        flush_pending()
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}"
            arguments = item.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("id") or call_id,
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": arguments},
                }],
            })
            continue
        if item_type == "function_call_output":
            output = _responses_value_to_text(item.get("output", ""))
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": str(output),
            })
            continue
        if item_type in {"reasoning", "compaction", "output_text", "summary_text"}:
            continue

        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        content = item.get("content", item.get("text", ""))
        messages.append({"role": role, "content": _responses_content(content)})
    flush_pending()
    return messages


def _merge_response_messages(previous, current):
    """Append new Responses input while accepting clients that resend full history."""
    previous = list(previous or [])
    current = list(current or [])
    if not previous:
        return current
    if not current:
        return previous
    if len(current) >= len(previous) and all(_messages_equal(a, b) for a, b in zip(current[:len(previous)], previous)):
        return current

    current_system = [m for m in current if m.get("role") == "system"]
    current_non_system = [m for m in current if m.get("role") != "system"]
    if current_system:
        previous_non_system = [m for m in previous if m.get("role") != "system"]
        previous_system = [m for m in previous if m.get("role") == "system"]
        if previous_system:
            return current_system + previous_non_system + current_non_system
    return previous + current


def _responses_options(body, previous_response_id=None):
    options = {}
    for key in ("instructions", "metadata", "tools", "tool_choice", "parallel_tool_calls", "temperature", "top_p", "text", "reasoning", "service_tier"):
        if key in body:
            options[key] = body[key]
    options["store"] = body.get("store", True) is not False
    if previous_response_id:
        options["previous_response_id"] = previous_response_id
    return options


def _responses_response_from_chat(result, response_id, requested_model, messages, body, previous_response_id=None):
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        output.append({
            "id": f"rs_{uuid.uuid4().hex[:24]}",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": reasoning}],
        })
    text = message.get("content") or ""
    if text or not message.get("tool_calls"):
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append({
            "id": tc.get("id") or f"fc_{uuid.uuid4().hex[:24]}",
            "type": "function_call",
            "status": "completed",
            "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })
    usage = result.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": requested_model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    response.update(_responses_options(body, previous_response_id))
    return response


_TITLE_REQUEST_MARKERS = (
    "generate a concise, single-line task title",
    "generate a concise task title",
    "生成一个简洁的任务标题",
    "生成简洁的任务标题",
)


def _local_title_for_request(messages):
    """Handle Codex's metadata-only title request without opening an upstream chat."""
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return None
    # A normal task may carry an earlier title prompt in its history. Only the
    # current user turn can trigger this local metadata shortcut.
    current_text = _messages_text([user_messages[-1]])
    lowered = current_text.lower()
    if not any(marker in lowered for marker in _TITLE_REQUEST_MARKERS):
        return None
    match = re.search(r"user prompt\s*:\s*(.+?)(?:\r?\n|$)", current_text, flags=re.IGNORECASE)
    source = (match.group(1) if match else "").strip()
    if not source:
        return "新建任务"
    source = re.sub(r"[\r\n]+", " ", source).strip(" \t`\"'。，！？.!?")
    file_match = re.search(r"([A-Za-z0-9_.-]+\.md)\b", source, flags=re.IGNORECASE)
    if file_match and re.search(r"[\u4e00-\u9fff]", source):
        title = "阅读" + file_match.group(1)
    else:
        words = source.split()
        title = " ".join(words[:4]) if words else "新建任务"
        if title and title[0].isascii() and title[0].isalpha():
            title = title[0].upper() + title[1:]
    return title[:36].rstrip(" .。!?！？") or "新建任务"


async def _local_response_stream(text):
    payload = {"choices": [{"delta": {"content": text}}]}
    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await request.json()
    messages = body.get("messages", [])
    model = resolve_model(body.get("model", DEFAULT_MODEL_ID), body=body, request=request)
    thinking = is_thinking_enabled(body, request)
    search = body.get("search", False)
    stream = body.get("stream", False)
    tools = body.get("tools", None)
    return await handle_chat(messages, model, thinking, search, stream, tools, scope=get_api_key(request))


@app.post("/v1/responses")
async def openai_responses(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await request.json()
    if not isinstance(body, dict):
        return _response_error("Request body must be a JSON object", "invalid_request_error", 400)
    if "input" in body and not isinstance(body.get("input"), (str, list, dict)):
        return _response_error("input must be a string, object, or array", "invalid_request_error", 400, "input")
    if body.get("tools") is not None and not isinstance(body.get("tools"), list):
        return _response_error("tools must be an array", "invalid_request_error", 400, "tools")

    model = resolve_model(body.get("model", DEFAULT_MODEL_ID), body=body, request=request)
    previous_response_id = body.get("previous_response_id")
    previous_state = None
    if previous_response_id:
        previous_state = _get_response_state(str(previous_response_id))
        if not previous_state:
            return _response_error(
                f"Response '{previous_response_id}' was not found",
                "invalid_request_error",
                404,
                "previous_response_id",
            )

    messages = _responses_input_messages(body)
    if previous_state:
        messages = _merge_response_messages(previous_state.get("messages", []), messages)
    thinking = is_thinking_enabled(body, request)
    search = body.get("search", False)
    stream = bool(body.get("stream", False))
    tools = body.get("tools", None)
    response_id = f"resp_{uuid.uuid4().hex}"
    requested_model = body.get("model") or DEFAULT_MODEL_ID
    options = _responses_options(body, previous_response_id)
    persist = options.get("store", True)

    local_title = _local_title_for_request(messages)
    if local_title is not None:
        if stream:
            return StreamingResponse(
                stream_responses_response(
                    _local_response_stream(local_title),
                    model,
                    messages,
                    requested_model=requested_model,
                    response_id=response_id,
                    persist=persist,
                    response_options=options,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        local_result = {
            "choices": [{
                "message": {"role": "assistant", "content": local_title},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": count_tok(local_title)},
        }
        response = _responses_response_from_chat(
            local_result,
            response_id,
            requested_model,
            messages,
            body,
            previous_response_id,
        )
        _store_response(
            response_id,
            response,
            messages=messages + [{"role": "assistant", "content": local_title}],
            persist=persist,
        )
        return response

    result = await handle_chat(messages, model, thinking, search, stream, tools, scope=get_api_key(request))

    if stream:
        if isinstance(result, StreamingResponse):
            return StreamingResponse(
                stream_responses_response(
                    result.body_iterator,
                    model,
                    messages,
                    requested_model=requested_model,
                    response_id=response_id,
                    persist=persist,
                    response_options=options,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return result

    if isinstance(result, JSONResponse):
        return result
    if isinstance(result, dict) and "choices" in result:
        response = _responses_response_from_chat(
            result,
            response_id,
            requested_model,
            messages,
            body,
            previous_response_id,
        )
        message = (result.get("choices") or [{}])[0].get("message") or {}
        assistant_message = {"role": "assistant", "content": message.get("content")}
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        stored_messages = messages + [assistant_message]
        _store_response(response_id, response, messages=stored_messages, persist=persist)
        return response
    return result


@app.get("/v1/responses/{response_id}")
async def get_openai_response(response_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    state = _get_response_state(response_id)
    if not state:
        return _response_error(f"Response '{response_id}' was not found", "not_found", 404)
    return state["response"]


@app.delete("/v1/responses/{response_id}")
async def delete_openai_response(response_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    state = _get_response_state(response_id)
    if not state:
        return _response_error(f"Response '{response_id}' was not found", "not_found", 404)
    RESPONSE_STORE.pop(response_id, None)
    RESPONSE_CANCELLED.discard(response_id)
    return {"id": response_id, "object": "response", "deleted": True}


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_openai_response(response_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    state = _get_response_state(response_id)
    if not state:
        return _response_error(f"Response '{response_id}' was not found", "not_found", 404)
    response = dict(state["response"])
    if response.get("status") in {"completed", "failed", "cancelled", "incomplete"}:
        return response
    RESPONSE_CANCELLED.add(response_id)
    response["status"] = "cancelled"
    response["incomplete_details"] = {"reason": "cancelled"}
    _update_stored_response(response_id, response, messages=state.get("messages", []))
    return response


@app.post("/v1/messages")
@app.post("/messages")
async def anthropic_messages(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await request.json()
    system = body.get("system", "")

    messages = body.get("messages", [])
    model = resolve_model(body.get("model", DEFAULT_MODEL_ID), body=body, request=request)

    thinking = is_thinking_enabled(body, request)
    stream = body.get("stream", False)
    tools = body.get("tools", [])

    openai_msgs = []
    if system:
        if isinstance(system, list):
            system_str = " ".join(c.get("text", "") for c in system if isinstance(c, dict) and c.get("type") == "text")
        else:
            system_str = str(system)
        if system_str:
            openai_msgs.append({"role": "system", "content": system_str})

    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            image_parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "image":
                        image_parts.append(c)
                    elif c.get("type") == "tool_use":
                        parts.append(f"{json.dumps({'name': c.get('name'), 'arguments': c.get('input', {})})}")
                    elif c.get("type") == "tool_result":
                        res_content = c.get("content", "")
                        if isinstance(res_content, list):
                            for item in res_content:
                                if isinstance(item, dict) and item.get("type") == "image":
                                    image_parts.append(item)
                            res_content = " ".join(item.get("text", "") for item in res_content if isinstance(item, dict) and item.get("type") == "text")
                        parts.append(f"[Tool Result for {c.get('tool_use_id', 'tool')}]: {res_content}")
            if image_parts:
                content = [{"type": "text", "text": s} for s in parts if s] + image_parts
            else:
                content = "\n".join(parts)
        if m.get("role") == "system":
            if content:
                openai_msgs.append({"role": "system", "content": content})
            continue
        if m["role"] == "assistant" and (not content or content.strip() == "(no content)"):
            continue
        openai_msgs.append({"role": m["role"], "content": content})

    openai_tools = []
    for t in tools:
        if t.get("type") == "function":
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", "NO DESCRIPTION"),
                    "parameters": t.get("input_schema", {}),
                },
            })
        elif "name" in t:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })

    output_config = body.get("output_config")
    if isinstance(output_config, dict) and output_config.get("format", {}).get("type") == "json_schema":
        json_schema = output_config["format"].get("schema")
        if json_schema:
            openai_msgs.insert(0, {"role": "system", "content": f"You MUST return valid JSON adhering strictly to this JSON Schema:\n{json.dumps(json_schema)}"})

    req_model = body.get("model")
    if stream:
        return await handle_chat(openai_msgs, model, thinking, False, True, openai_tools or None, is_anthropic=True, req_model=req_model, scope=get_api_key(request))

    result = await handle_chat(openai_msgs, model, thinking, False, False, openai_tools or None, is_anthropic=True, req_model=req_model, scope=get_api_key(request))
    if not isinstance(result, dict) or "choices" not in result:
        return result
    return format_anthropic_response(result, req_model)


@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)

    base_models = [
        {
            "id": DEFAULT_MODEL_ID,
            "object": "model",
            "type": "model",
            "name": DEFAULT_MODEL_ID,
            "display_name": "DeepSeek V4 Flash Vision Exp",
            "created": 1785456000,
            "created_at": "2026-07-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "capabilities": {
                "batch": {"supported": True},
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True}
                    }
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True}
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True
                }
            }
        },
        {
            "id": "DeepSeek-V4-Flash",
            "object": "model",
            "type": "model",
            "name": "DeepSeek-V4-Flash",
            "display_name": "DeepSeek V4 Flash",
            "created": 1788134400,
            "created_at": "2026-08-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "capabilities": {
                "batch": {"supported": True},
                "code_execution": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True}
                    }
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True}
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True
                }
            }
        },
        {
            "id": "DeepSeek-V4-Pro",
            "object": "model",
            "type": "model",
            "name": "DeepSeek-V4-Pro",
            "display_name": "DeepSeek V4 Pro",
            "created": 1785456000,
            "created_at": "2026-07-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "capabilities": {
                "batch": {"supported": True},
                "code_execution": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True}
                    }
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True}
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True
                }
            }
        }
    ]

    claude_aliases = []
    for m in base_models:
        alias = dict(m)
        alias["id"] = f"anthropic/claude-{m['id']}"
        alias["name"] = f"anthropic/claude-{m['name']}"
        alias["display_name"] = f"Claude {m['display_name']}"
        claude_aliases.append(alias)

    all_models = base_models + claude_aliases

    return {
        "object": "list",
        "data": all_models,
        "has_more": False,
        "first_id": all_models[0]["id"],
        "last_id": all_models[-1]["id"]
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    if time.time() < _login_fails["locked_until"]:
        return templates.TemplateResponse(request, "login.html", {"error": "Too many attempts. Try again later."})
    if secrets.compare_digest(username.encode("utf-8"), ADMIN_USER.encode("utf-8")) and secrets.compare_digest(password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")):
        _login_fails["count"] = 0
        sid = str(uuid.uuid4())
        SESSIONS[sid] = time.time()
        resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")
        resp.set_cookie("session_id", sid, httponly=True, samesite="lax")
        return resp
    _login_fails["count"] += 1
    if _login_fails["count"] >= 5:
        _login_fails["locked_until"] = time.time() + 300
        _login_fails["count"] = 0
    return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password"})


@app.get("/logout")
async def logout(request: Request):
    sid = request.cookies.get("session_id")
    SESSIONS.pop(sid, None)
    resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    resp.delete_cookie("session_id")
    return resp


@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    tokens = get_tokens()
    return templates.TemplateResponse(request, "dashboard.html", {"tokens": tokens})


@app.post("/tokens/add")
async def tokens_add(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    form = await request.form()
    auth_token = form.get("auth_token", "").strip().strip("'\"")
    alias = form.get("alias", "").strip() or None
    if auth_token:
        add_token(auth_token, alias)
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.post("/tokens/{token_id}/delete")
async def tokens_delete(token_id: int, request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    delete_token(token_id)
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return await dashboard(request)


@app.get("/health")
async def health(request: Request):
    active = sum(1 for t in get_tokens() if t["status"] == "ACTIVE")
    cookies_valid = False
    try:
        with open(COOKIE_FILE) as f:
            c = json.load(f)
        exp = c.get("expiry")
        cookies_valid = bool(exp and exp > time.time())
    except Exception:
        cookies_valid = False
    ok = active > 0 and cookies_valid
    data = {"status": "ok" if ok else "degraded"}
    if check_key(request):
        data["active_tokens"] = active
        data["cookies_valid"] = cookies_valid
    return JSONResponse(data, status_code=200 if ok else 503)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "4000")))
