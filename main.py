"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
=============================================
让你的 AI 拥有长期记忆。

工作原理：
1. 接收客户端（Kelivo / ChatBox / 任何 OpenAI 兼容客户端）的消息
2. 自动搜索数据库中的相关记忆，注入 system prompt
3. 转发给 LLM API（支持 OpenRouter / OpenAI / 任何兼容接口）
4. 后台自动存储对话 + 用 AI 提取新记忆

环境变量 MEMORY_ENABLED=false 时退化为纯转发网关（第一阶段）。
"""

import os
import json
import uuid
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from database import init_tables, close_pool, save_message, search_memories, save_memory, get_all_memories_count, get_recent_memories, get_all_memories, get_pool, get_all_memories_detail, update_memory, delete_memory, delete_memories_batch, get_gateway_config, set_gateway_config, get_all_gateway_config, get_conversation_messages, get_session_cache_state, save_session_cache_state, delete_session_cache_state, save_token_usage, ensure_token_usage_table, ensure_conversation_titles_table, get_conversations_paginated, delete_conversation, batch_delete_conversations, merge_sessions_to_target, list_all_session_cache_states, export_all_conversations, import_conversations, get_last_user_content, update_last_assistant_message, db_row_to_message, backfill_memory_embeddings, get_pending_memory_embedding_count, search_conversations, update_message_content, rename_session_id, get_fragments_by_date, get_fragments_by_date_range, create_event_memory, deactivate_memories, promote_to_core, merge_memories, check_duplicate_memory, update_memory_with_layer, get_layer_statistics, cleanup_old_fragments, revert_merge
import database as _db_module
from memory_extractor import extract_memories, score_memories

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")
PORT = int(os.getenv("PORT", "8080"))
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))
MEMORY_EXTRACT_ENABLED = os.getenv("MEMORY_EXTRACT_ENABLED", "true").lower() == "true"
CACHE_PARTITION_ENABLED = os.getenv("CACHE_PARTITION_ENABLED", "false").lower() == "true"
CACHE_PARTITION_X = int(os.getenv("CACHE_PARTITION_X", "15"))
CACHE_SUMMARY_MODEL = os.getenv("CACHE_SUMMARY_MODEL", "anthropic/claude-haiku-4.5")
CACHE_PARTITION_TRIGGER = os.getenv("CACHE_PARTITION_TRIGGER", "rounds")
CACHE_PARTITION_WINDOW = int(os.getenv("CACHE_PARTITION_WINDOW", "30"))
PARTITION_SESSION_ID = os.getenv("PARTITION_SESSION_ID", "")

def get_active_session_id() -> str:
    return PARTITION_SESSION_ID

TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))
_round_counter = 0
FORCE_STREAM = os.getenv("FORCE_STREAM", "false").lower() == "true"
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")
EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")


def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    print("ℹ️  未找到 system_prompt.txt 或文件为空，将不注入 system prompt")
    return ""


SYSTEM_PROMPT = load_system_prompt()
_DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️  无人设，纯转发模式")

_cached_system_prompt = None
_cached_system_prompt_loaded = False

async def get_system_prompt() -> str:
    global _cached_system_prompt, _cached_system_prompt_loaded
    if _cached_system_prompt_loaded:
        return _cached_system_prompt or ""
    try:
        db_prompt = await get_gateway_config("systemPrompt", "")
        if db_prompt:
            _cached_system_prompt = db_prompt
        else:
            _cached_system_prompt = _DEFAULT_SYSTEM_PROMPT
            if _DEFAULT_SYSTEM_PROMPT:
                await set_gateway_config("systemPrompt", _DEFAULT_SYSTEM_PROMPT)
        _cached_system_prompt_loaded = True
        return _cached_system_prompt or ""
    except Exception:
        _cached_system_prompt = _DEFAULT_SYSTEM_PROMPT
        _cached_system_prompt_loaded = True
        return _cached_system_prompt or ""

def invalidate_system_prompt_cache():
    global _cached_system_prompt, _cached_system_prompt_loaded
    _cached_system_prompt = None
    _cached_system_prompt_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PARTITION_SESSION_ID
    if MEMORY_ENABLED:
        try:
            await init_tables()
            await ensure_token_usage_table()
            await ensure_conversation_titles_table()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")

            try:
                db_cfg = await get_all_gateway_config()
                if db_cfg:
                    _RESTORE_MAIN = {
                        "API_BASE_URL": str, "API_KEY": str, "DEFAULT_MODEL": str,
                        "MEMORY_ENABLED": lambda v: _parse_bool(v),
                        "MAX_MEMORIES_INJECT": int, "MEMORY_EXTRACT_INTERVAL": int,
                        "CACHE_PARTITION_ENABLED": lambda v: _parse_bool(v),
                        "CACHE_PARTITION_X": int, "CACHE_PARTITION_TRIGGER": str,
                        "CACHE_PARTITION_WINDOW": int, "CACHE_SUMMARY_MODEL": str,
                        "FORCE_STREAM": lambda v: _parse_bool(v),
                        "REASONING_EFFORT": str,
                    }
                    _RESTORE_DB = {
                        "EMBEDDING_API_KEY": str, "EMBEDDING_BASE_URL": str,
                        "EMBEDDING_MODEL": str, "EMBEDDING_DIM": int,
                        "MIN_SCORE_THRESHOLD": float,
                        "MEMORY_VECTOR_ENABLED": lambda v: _parse_bool(v),
                        "MEMORY_HW_KEYWORD": float, "MEMORY_HW_SEMANTIC": float,
                        "MEMORY_HW_IMPORTANCE": float, "MEMORY_HW_RECENCY": float,
                        "MEMORY_SEMANTIC_THRESHOLD": float,
                    }
                    restored = []
                    for key, val in db_cfg.items():
                        if not val:
                            continue
                        if key in _RESTORE_MAIN:
                            globals()[key] = _RESTORE_MAIN[key](val)
                            restored.append(key)
                        elif key in _RESTORE_DB:
                            setattr(_db_module, key, _RESTORE_DB[key](val))
                            restored.append(key)
                        elif key == "MEMORY_MODEL":
                            os.environ["MEMORY_MODEL"] = str(val)
                            restored.append(key)
                    if restored:
                        print(f"🔄 从数据库恢复 {len(restored)} 项面板配置: {', '.join(restored)}")
            except Exception as e:
                print(f"[warning] 恢复面板配置失败: {e}")

            if not MEMORY_EXTRACT_ENABLED:
                print(f"ℹ️  记忆提取+注入已关闭（MEMORY_EXTRACT_ENABLED=false）")

            if CACHE_PARTITION_ENABLED:
                db_sid = await get_gateway_config("partition_session_id", "")
                if db_sid:
                    PARTITION_SESSION_ID = db_sid
                    print(f"🔗 活跃对话线(DB): {PARTITION_SESSION_ID}")
                elif PARTITION_SESSION_ID:
                    await set_gateway_config("partition_session_id", PARTITION_SESSION_ID)
                    print(f"🔗 活跃对话线(ENV→DB): {PARTITION_SESSION_ID}")
                print(f"🔒 分区缓存已启用: X={CACHE_PARTITION_X}, 摘要模型={CACHE_SUMMARY_MODEL}")
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
            print("⚠️  记忆系统将不可用，但网关仍可正常转发")
    else:
        print("ℹ️  记忆系统已关闭（设置 MEMORY_ENABLED=true 开启）")

    yield

    if MEMORY_ENABLED:
        await close_pool()


app = FastAPI(title="AI Memory Gateway", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


async def build_system_prompt_with_memories(user_message: str) -> str:
    if not MEMORY_ENABLED or not MEMORY_EXTRACT_ENABLED:
        return SYSTEM_PROMPT

    if MAX_MEMORIES_INJECT <= 0:
        return SYSTEM_PROMPT

    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)

        if not memories:
            return SYSTEM_PROMPT

        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    utc_str = str(mem['created_at'])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
            memory_lines.append(f"- {date_str}{mem['content']}")
        memory_text = "\n".join(memory_lines)

        enhanced_prompt = f"""{SYSTEM_PROMPT}

【从过往对话中检索到的相关记忆】
{memory_text}

# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用，避免主动提及
- 对重要信息（如健康、日期、约定）保持一致性
- 新信息与记忆冲突时，以新信息为准
- 模糊记忆可表达不确定性："记得你似乎说过..."

# 交流方式
- 自然引用："记得你说过..."或"上次我们聊到..."
- 避免机械式表达如"根据我的记忆..."或"检索到的信息显示..."
- 共同经历可温情回忆："上次那个事挺好玩的"

记忆是丰富对话的工具，而非对话焦点。"""

        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt

    except Exception as e:
        print(f"⚠️  记忆检索失败: {e}，使用纯人设")
        return SYSTEM_PROMPT


def _is_anthropic_model(model: str) -> bool:
    model_lower = model.lower()
    return "claude" in model_lower or "anthropic" in model_lower


def _strip_cache_control(messages: list):
    stripped = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                del block["cache_control"]
                stripped += 1
        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
            msg["content"] = content[0]["text"]
    if stripped > 0:
        print(f"🔧 兼容性处理: 剥离了 {stripped} 个 cache_control 字段（非 Claude 模型）")


def build_time_injection() -> str:
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc + timedelta(hours=TIMEZONE_HOURS)
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekday_names[now_local.weekday()]
    time_str = now_local.strftime("%Y年%m月%d日 %H:%M")
    return f"【当前时间】{time_str} {weekday}"


async def generate_summary(messages: list, session_id: str = "") -> str:
    if not messages:
        return ""

    conversation_text = ""
    for msg in messages:
        role_label = "用户" if msg['role'] == 'user' else "AI"
        content = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])
        conversation_text += f"{role_label}: {content}\n\n"

    prompt = f"""请将以下对话压缩成简洁摘要。保留关键信息（事件、决定、情感、约定），去掉日常寒暄和重复内容。用第三人称叙述，控制在300字以内。

---
{conversation_text}
---

摘要："""

    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        if "openrouter" in API_BASE_URL:
            headers["HTTP-Referer"] = EXTRA_REFERER
            headers["X-Title"] = EXTRA_TITLE

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(API_BASE_URL, headers=headers, json={
                "model": CACHE_SUMMARY_MODEL,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            })
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    summary = data["choices"][0]["message"]["content"].strip()
                    print(f"📝 摘要生成完成: {len(summary)}字 (压缩{len(messages)}条消息)")
                    return summary

        print(f"⚠️ 摘要生成失败: HTTP {response.status_code}")
        return ""
    except Exception as e:
        print(f"⚠️ 摘要生成异常: {e}")
        return ""


def group_by_rounds(history: list) -> list:
    rounds = []
    current_round = []
    for msg in history:
        if msg['role'] == 'user' and current_round:
            rounds.append(current_round)
            current_round = []
        current_round.append(msg)
    if current_round:
        rounds.append(current_round)
    return rounds


def _should_rotate(b_rounds_count: int, X: int, a_msgs: list) -> bool:
    if b_rounds_count == 0:
        return False

    if CACHE_PARTITION_TRIGGER == "time":
        a_first_time = None
        for msg in a_msgs:
            t = msg.get('created_at')
            if t:
                a_first_time = t
                break

        if a_first_time:
            now = datetime.now(timezone.utc)
            if a_first_time.tzinfo is None:
                a_first_time = a_first_time.replace(tzinfo=timezone.utc)
            age_minutes = (now - a_first_time).total_seconds() / 60
            return age_minutes >= CACHE_PARTITION_WINDOW

        return b_rounds_count >= X

    return b_rounds_count >= X


CACHE_MAX_ROTATIONS = int(os.getenv("CACHE_MAX_ROTATIONS", "2"))


def _apply_breakpoint(msg: dict) -> bool:
    content = msg.get('content')

    if isinstance(content, str) and content.strip():
        msg['content'] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return True

    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                block["cache_control"] = {"type": "ephemeral"}
                return True

    return False


async def build_partitioned_messages(
    session_id: str,
    all_messages: list,
    base_prompt: str,
    user_message: str,
) -> list:
    X = CACHE_PARTITION_X

    non_system = [m for m in all_messages if m.get('role') != 'system']

    current_user_msg = None
    history = non_system[:]
    if history and history[-1].get('role') == 'user':
        current_user_msg = history.pop()

    cleaned = []
    orphan_count = 0
    for msg in history:
        if msg.get('role') == 'tool':
            prev = cleaned[-1] if cleaned else None
            if prev and (prev.get('role') == 'tool' or
                        (prev.get('role') == 'assistant' and prev.get('tool_calls'))):
                cleaned.append(msg)
            else:
                orphan_count += 1
        else:
            cleaned.append(msg)
    if orphan_count > 0:
        print(f"⚠️ 清理了 {orphan_count} 条孤立tool消息")
    history = cleaned

    rounds = group_by_rounds(history)
    total_rounds = len(rounds)

    state = await get_session_cache_state(session_id)
    summary_parts = state['summary_parts']
    a_start_round = state['a_start_round']

    if total_rounds < X:
        return await _build_basic_cached(history, base_prompt, user_message, current_user_msg)

    a_end_round = a_start_round + X
    a_round_groups = rounds[a_start_round : a_end_round]
    b_round_groups = rounds[a_end_round :]
    a_msgs = [msg for rnd in a_round_groups for msg in rnd]
    b_msgs = [msg for rnd in b_round_groups for msg in rnd]
    b_rounds_count = len(b_round_groups)

    rotation_count = 0
    max_rotations = CACHE_MAX_ROTATIONS if CACHE_PARTITION_TRIGGER == "time" else 999
    while _should_rotate(b_rounds_count, X, a_msgs) and rotation_count < max_rotations:
        rotation_count += 1
        trigger_info = f"B区{b_rounds_count}轮 >= X={X}" if CACHE_PARTITION_TRIGGER != "time" else f"A区首条消息超出{CACHE_PARTITION_WINDOW}分钟窗口"
        print(f"🔄 轮转#{rotation_count}: session={session_id}, {trigger_info}")

        new_summary = await generate_summary(a_msgs, session_id)
        if new_summary:
            summary_parts.append(new_summary)

        a_start_round += X
        a_end_round = a_start_round + X
        a_round_groups = rounds[a_start_round : a_end_round]
        b_round_groups = rounds[a_end_round :]
        a_msgs = [msg for rnd in a_round_groups for msg in rnd]
        b_msgs = [msg for rnd in b_round_groups for msg in rnd]
        b_rounds_count = len(b_round_groups)

    if rotation_count > 0:
        await save_session_cache_state(session_id, summary_parts, a_start_round)
        summary_total = sum(len(p) for p in summary_parts)
        print(f"🔄 轮转完成(共{rotation_count}次): 摘要{len(summary_parts)}段/{summary_total}字, A区{len(a_msgs)}条, B区{len(b_msgs)}条")

    result = []
    if base_prompt:
        result.append({
            "role": "system",
            "content": [{"type": "text", "text": base_prompt, "cache_control": {"type": "ephemeral"}}]
        })

    if summary_parts:
        blocks = [{"type": "text", "text": "[以下是之前对话的摘要，帮助你回忆上下文]"}]
        for i, part in enumerate(summary_parts):
            item = {"type": "text", "text": part}
            if i == len(summary_parts) - 1:
                item["cache_control"] = {"type": "ephemeral"}
            blocks.append(item)
        result.append({"role": "user", "content": blocks})
        result.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

    cleaned_a = []
    for msg in a_msgs:
        if msg.get('role') == 'tool':
            continue
        m = {k: v for k, v in msg.items() if k not in ('created_at', 'tool_calls')}
        if m.get('role') == 'assistant' and not (m.get('content') or '').strip():
            continue
        cleaned_a.append(m)

    for j in range(len(cleaned_a) - 1, -1, -1):
        if cleaned_a[j].get('role') != 'tool' and _apply_breakpoint(cleaned_a[j]):
            break

    for m in cleaned_a:
        result.append(m)

    b_cleaned = [{k: v for k, v in msg.items() if k not in ('created_at',)} for msg in b_msgs]

    for j in range(len(b_cleaned) - 1, -1, -1):
        if b_cleaned[j].get('role') != 'tool' and _apply_breakpoint(b_cleaned[j]):
            break

    for m in b_cleaned:
        result.append(m)

    if current_user_msg:
        parts = [build_time_injection()]

        if MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message:
            mem_text = await build_memory_text(user_message)
            if mem_text:
                parts.append(mem_text)

        current_text = current_user_msg['content']
        if isinstance(current_text, list):
            current_text = " ".join(
                item.get("text", "") for item in current_text
                if isinstance(item, dict) and item.get("type") == "text"
            )

        parts.append(current_text)
        result.append({"role": "user", "content": "\n\n".join(parts)})

    bp_count = 1 + (1 if summary_parts else 0) + (1 if cleaned_a else 0) + (1 if b_msgs else 0)
    summary_total = sum(len(p) for p in summary_parts)
    tool_stripped = len(a_msgs) - len(cleaned_a)
    a_info = f"A区{len(cleaned_a)}条({len(a_round_groups)}轮)" + (f"[剥离{tool_stripped}条tool]" if tool_stripped else "")
    print(f"🔒 分区缓存: BP×{bp_count} | 摘要{'有' if summary_parts else '无'}({len(summary_parts)}段/{summary_total}字) | {a_info} | B区{len(b_msgs)}条({b_rounds_count}轮) | 总{len(result)}条messages")
    return result


async def _build_basic_cached(
    history: list,
    base_prompt: str,
    user_message: str,
    current_user_msg: dict,
) -> list:
    result = []
    if base_prompt:
        result.append({
            "role": "system",
            "content": [{"type": "text", "text": base_prompt, "cache_control": {"type": "ephemeral"}}]
        })

    h_cleaned = [{k: v for k, v in msg.items() if k not in ('created_at',)} for msg in history]

    for j in range(len(h_cleaned) - 1, -1, -1):
        if h_cleaned[j].get('role') != 'tool' and _apply_breakpoint(h_cleaned[j]):
            break

    for m in h_cleaned:
        result.append(m)

    if current_user_msg:
        parts = [build_time_injection()]

        if MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message:
            mem_text = await build_memory_text(user_message)
            if mem_text:
                parts.append(mem_text)

        current_text = current_user_msg['content']
        if isinstance(current_text, list):
            current_text = " ".join(
                item.get("text", "") for item in current_text
                if isinstance(item, dict) and item.get("type") == "text"
            )

        parts.append(current_text)
        result.append({"role": "user", "content": "\n\n".join(parts)})

    bp_count = 1 + (1 if history else 0)
    print(f"🔒 基础缓存(降级): BP×{bp_count} | 历史{len(history)}条 | 总{len(result)}条messages")
    return result


async def build_memory_text(user_message: str) -> str:
    if MAX_MEMORIES_INJECT <= 0:
        return ""
    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)
        if not memories:
            return ""

        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    utc_str = str(mem['created_at'])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
            memory_lines.append(f"- {date_str}{mem['content']}")

        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return "【从过往对话中检索到的相关记忆】\n" + "\n".join(memory_lines)
    except Exception as e:
        print(f"⚠️ 记忆检索失败: {e}")
        return ""


async def process_memories_background(session_id: str, user_msg: str, assistant_msg: str, model: str, context_messages: list = None, skip_conversation_log: bool = False, tool_messages: list = None, assistant_tool_calls: list = None, assistant_reasoning: str = None):
    global _round_counter

    try:
        print(f"💾 process_memories_background: user_msg={bool(user_msg)}, tool_messages={len(tool_messages) if tool_messages else 0}, "
              f"assistant_tool_calls={len(assistant_tool_calls) if assistant_tool_calls else 0}, skip={skip_conversation_log}")
        if tool_messages:
            print(f"💾 tool详情: {[{'role': m.get('role'), 'tool_call_id': m.get('tool_call_id', '?')} for m in tool_messages]}")

        if skip_conversation_log:
            print(f"⏭️  跳过对话存储（辅助请求）")
        elif tool_messages:
            for tm in tool_messages:
                meta_dict = {}
                if tm.get("tool_call_id"):
                    meta_dict["tool_call_id"] = tm["tool_call_id"]
                if tm.get("name"):
                    meta_dict["name"] = tm["name"]
                meta = json.dumps(meta_dict) if meta_dict else None
                await save_message(session_id, "tool", tm.get("content", ""), model, metadata=meta)

            if assistant_msg or assistant_tool_calls:
                ast_meta_dict = {}
                if assistant_tool_calls:
                    ast_meta_dict["tool_calls"] = assistant_tool_calls
                if assistant_reasoning:
                    ast_meta_dict["reasoning_content"] = assistant_reasoning
                ast_meta = json.dumps(ast_meta_dict) if ast_meta_dict else None
                await save_message(session_id, "assistant", assistant_msg or "", model, metadata=ast_meta)
                print(f"🔧 存储: {len(tool_messages)}条tool + 1条assistant" + (" (含tool_calls)" if assistant_tool_calls else "") + (" (含reasoning)" if assistant_reasoning else ""))
        else:
            ast_meta_dict = {}
            if assistant_tool_calls:
                ast_meta_dict["tool_calls"] = assistant_tool_calls
            if assistant_reasoning:
                ast_meta_dict["reasoning_content"] = assistant_reasoning
            assistant_meta = json.dumps(ast_meta_dict) if ast_meta_dict else None

            if assistant_tool_calls:
                await save_message(session_id, "user", user_msg, model)
                await save_message(session_id, "assistant", assistant_msg or "", model, metadata=assistant_meta)
                print(f"🔧 存储: user + assistant (含{len(assistant_tool_calls)}个tool_calls)" + (" (含reasoning)" if assistant_reasoning else ""))
            else:
                last_user = await get_last_user_content(session_id)
                if last_user and last_user.strip() == user_msg.strip():
                    updated = await update_last_assistant_message(session_id, assistant_msg, model)
                    if updated:
                        print(f"🔄 检测到re-roll，已覆盖最后一条assistant回复")
                    else:
                        await save_message(session_id, "user", user_msg, model)
                        await save_message(session_id, "assistant", assistant_msg, model, metadata=assistant_meta)
                else:
                    await save_message(session_id, "user", user_msg, model)
                    await save_message(session_id, "assistant", assistant_msg, model, metadata=assistant_meta)

        if not MEMORY_EXTRACT_ENABLED:
            print(f"⏭️  记忆提取已关闭（MEMORY_EXTRACT_ENABLED=false）")
            return

        if MEMORY_EXTRACT_INTERVAL == 0:
            print(f"⏭️  记忆自动提取已禁用，跳过")
            return

        _round_counter += 1

        if MEMORY_EXTRACT_INTERVAL > 1 and (_round_counter % MEMORY_EXTRACT_INTERVAL != 0):
            print(f"⏭️  轮次 {_round_counter}，跳过记忆提取（每 {MEMORY_EXTRACT_INTERVAL} 轮提取一次）")
            return

        if MEMORY_EXTRACT_INTERVAL > 1:
            print(f"📝 轮次 {_round_counter}，执行记忆提取")

        existing = await get_recent_memories(limit=80)
        existing_contents = [r["content"] for r in existing]

        if context_messages:
            tail_count = MEMORY_EXTRACT_INTERVAL * 2
            recent_msgs = list(context_messages)[-tail_count:] if len(context_messages) > tail_count else list(context_messages)
            messages_for_extraction = recent_msgs + [
                {"role": "assistant", "content": assistant_msg}
            ]
            print(f"📝 截取最近 {MEMORY_EXTRACT_INTERVAL} 轮对话提取记忆（{len(messages_for_extraction)} 条消息）")
        else:
            messages_for_extraction = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]

        new_memories = await extract_memories(messages_for_extraction, existing_memories=existing_contents)

        META_BLACKLIST = [
            "记忆库", "记忆系统", "检索", "没有被记录", "没有被提取",
            "记忆遗漏", "尚未被记录", "写入不完整", "检索功能",
            "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
            "阈值", "数据库", "seed", "导入", "部署",
            "bug", "debug", "端口", "网关",
        ]

        filtered_memories = []
        for mem in new_memories:
            content = mem["content"]
            if any(kw in content for kw in META_BLACKLIST):
                print(f"🚫 过滤掉meta记忆: {content[:60]}...")
                continue
            filtered_memories.append(mem)

        for mem in filtered_memories:
            await save_memory(
                content=mem["content"],
                importance=mem["importance"],
                source_session=session_id,
            )

        if filtered_memories:
            total = await get_all_memories_count()
            print(f"💾 已保存 {len(filtered_memories)} 条新记忆（过滤了 {len(new_memories) - len(filtered_memories)} 条），总计 {total} 条")

    except Exception as e:
        print(f"⚠️  后台记忆处理失败: {e}")


@app.get("/")
async def health_check():
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except:
            pass

    return {
        "status": "running",
        "gateway": "AI Memory Gateway v2.0",
        "system_prompt_loaded": len(SYSTEM_PROMPT) > 0,
        "system_prompt_length": len(SYSTEM_PROMPT),
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
        "memory_extract_interval": MEMORY_EXTRACT_INTERVAL,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ai-memory-gateway",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "API_KEY 未设置，请在环境变量中配置"},
        )

    body = await request.json()
    messages = body.get("messages", [])

    skip_conversation_log = request.headers.get("X-Skip-Conversation-Log", "").lower() == "true"

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            break

    original_messages = [msg for msg in messages if msg.get("role") != "system"]

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    if tool_messages:
        print(f"🔧 检测到 {len(tool_messages)} 条工具结果消息")

    session_id = str(uuid.uuid4())[:8]

    if CACHE_PARTITION_ENABLED:
        active_sid = get_active_session_id()
        if active_sid:
            session_id = active_sid

        try:
            db_history = await get_conversation_messages(session_id, limit=10000)
            db_msgs = []
            for m in (db_history or []):
                msg = db_row_to_message(m)
                msg['created_at'] = m.get('created_at')
                db_msgs.append(msg)
        except Exception as e:
            print(f"[warning] 分区模式读取历史失败: {e}")
            db_msgs = []

        client_new_msgs = [m for m in messages if m.get("role") != "system"]
        client_new_msgs = [m for m in client_new_msgs if m.get("role") != "assistant"]
        user_msgs = [m for m in client_new_msgs if m.get("role") == "user"]
        if len(user_msgs) > 1:
            last_user = user_msgs[-1]
            client_new_msgs = [m for m in client_new_msgs if m.get("role") != "user"]
            client_new_msgs.append(last_user)
            print(f"🔧 去重: 过滤{len(user_msgs)-1}条冗余user，保留最后1条")

        client_tools = [m for m in client_new_msgs if m.get("role") == "tool"]
        if client_tools:
            db_last = db_msgs[-1] if db_msgs else None
            db_expecting_tool = (db_last and db_last.get("role") == "assistant" and db_last.get("tool_calls"))

            if not db_expecting_tool:
                stale_ids = [m.get('tool_call_id', '?') for m in client_tools]
                print(f"🔧 去重: DB未在等待tool结果，丢弃{len(client_tools)}条客户端tool (ids: {stale_ids})")
                client_new_msgs = [m for m in client_new_msgs if m.get("role") != "tool"]
            else:
                expected_tool_ids = {tc.get("id") for tc in db_last.get("tool_calls", []) if tc.get("id")}
                new_tools = [m for m in client_tools if m.get("tool_call_id") in expected_tool_ids]
                stale_tools = [m for m in client_tools if m.get("tool_call_id") not in expected_tool_ids]

                if stale_tools:
                    print(f"🔧 去重: 丢弃{len(stale_tools)}条非当前轮次tool (ids: {[m.get('tool_call_id','?') for m in stale_tools]})")
                if new_tools:
                    print(f"🔧 保留{len(new_tools)}条当前轮次tool (ids: {[m.get('tool_call_id','?') for m in new_tools]})")

                last_msg = client_new_msgs[-1] if client_new_msgs else None
                client_new_msgs = new_tools[:]
                if last_msg and last_msg.get("role") == "user":
                    client_new_msgs.append(last_msg)

                if new_tools:
                    new_tool_ids = {m.get("tool_call_id") for m in new_tools if m.get("tool_call_id")}
                    db_has_matching_ast = False
                    for m in db_msgs:
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            ast_tc_ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
                            if new_tool_ids & ast_tc_ids:
                                db_has_matching_ast = True
                                break
                    if not db_has_matching_ast and new_tool_ids:
                        for m in messages:
                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                ast_tc_ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
                                if new_tool_ids & ast_tc_ids:
                                    client_new_msgs.insert(0, m)
                                    print(f"⚠️ Race防护: 从客户端补充assistant(tool_calls)")
                                    break

        all_msgs = db_msgs + client_new_msgs
        tool_messages = [m for m in client_new_msgs if m.get("role") == "tool"]

        print(f"📦 分区模式: DB历史{len(db_msgs)}条 + 客户端消息{len(client_new_msgs)}条")

        messages = await build_partitioned_messages(
            session_id, all_msgs, SYSTEM_PROMPT, user_message
        )
        body["messages"] = messages

    else:
        if SYSTEM_PROMPT or (MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message):
            if MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message:
                enhanced_prompt = await build_system_prompt_with_memories(user_message)
            else:
                enhanced_prompt = SYSTEM_PROMPT

            if enhanced_prompt:
                has_system = any(msg.get("role") == "system" for msg in messages)
                if has_system:
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                            break
                else:
                    messages.insert(0, {"role": "system", "content": enhanced_prompt})

        body["messages"] = messages

    model = body.get("model", DEFAULT_MODEL)
    if not model:
        model = DEFAULT_MODEL
    body["model"] = model

    if CACHE_PARTITION_ENABLED and not _is_anthropic_model(model):
        _strip_cache_control(body.get("messages", []))

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if "openrouter" in API_BASE_URL:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE

    is_stream = body.get("stream", False)

    if FORCE_STREAM and not is_stream:
        is_stream = True
        body["stream"] = True
        print(f"⚡ 强制开启流式传输（FORCE_STREAM=true）")

    if REASONING_EFFORT:
        body.pop("reasoning_effort", None)
        body.pop("google", None)
        body["reasoning_effort"] = REASONING_EFFORT
        print(f"🧠 注入推理参数: reasoning_effort={REASONING_EFFORT}")

    print(f"📡 请求: model={model}, stream={is_stream}, memory={'on' if MEMORY_ENABLED else 'off'}", flush=True)

    debug_keys = {k: v for k, v in body.items() if k in ('reasoning_effort', 'google', 'reasoning')}
    if debug_keys:
        print(f"📡 推理字段: {debug_keys}", flush=True)

    if is_stream:
        return StreamingResponse(
            stream_and_capture(headers, body, session_id, user_message, model, original_messages, skip_conversation_log, tool_messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(API_BASE_URL, headers=headers, json=body)

            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = ""
                assistant_tool_calls = None
                assistant_reasoning = None
                try:
                    msg_obj = resp_data["choices"][0]["message"]
                    assistant_msg = msg_obj.get("content") or ""
                    if msg_obj.get("tool_calls"):
                        assistant_tool_calls = msg_obj["tool_calls"]
                        print(f"🔧 Response 包含 {len(assistant_tool_calls)} 个工具调用")
                    if msg_obj.get("reasoning_content"):
                        assistant_reasoning = msg_obj["reasoning_content"]
                        print(f"🧠 Response 包含 reasoning_content ({len(assistant_reasoning)}字符)")
                except (KeyError, IndexError):
                    pass

                if MEMORY_ENABLED and (user_message or tool_messages):
                    asyncio.create_task(
                        process_memories_background(session_id, user_message, assistant_msg, model,
                                                    context_messages=original_messages, skip_conversation_log=skip_conversation_log,
                                                    tool_messages=tool_messages, assistant_tool_calls=assistant_tool_calls,
                                                    assistant_reasoning=assistant_reasoning)
                    )

                return JSONResponse(status_code=200, content=resp_data)
            else:
                return JSONResponse(status_code=response.status_code, content=response.json())


async def stream_and_capture(headers: dict, body: dict, session_id: str, user_message: str, model: str, original_messages: list = None, skip_conversation_log: bool = False, tool_messages: list = None):
    full_response = []
    full_reasoning = []
    stream_usage = {}
    line_buffer = ""
    accumulated_tool_calls = {}

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", API_BASE_URL, headers=headers, json=body) as response:
            upstream_ct = response.headers.get("content-type", "")
            print(f"📨 上游响应: status={response.status_code}, content-type={upstream_ct}", flush=True)

            if response.status_code != 200:
                msg_summary = [{"role": m.get("role"), "tool_calls": bool(m.get("tool_calls")), "tool_call_id": m.get("tool_call_id", ""), "content_type": type(m.get("content")).__name__} for m in body.get("messages", [])]
                print(f"❌ 发送的messages结构({len(msg_summary)}条): {msg_summary}", flush=True)

            error_body_parts = []
            is_error = response.status_code != 200

            async for chunk in response.aiter_bytes():
                yield chunk

                if is_error:
                    error_body_parts.append(chunk)
                    continue

                text = chunk.decode("utf-8", errors="ignore")
                line_buffer += text
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])

                            if "usage" in data:
                                stream_usage = data["usage"]

                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response.append(content)

                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                full_reasoning.append(reasoning)

                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "index": idx,
                                            "id": tc.get("id", ""),
                                            "type": tc.get("type", "function"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if tc.get("id"):
                                        accumulated_tool_calls[idx]["id"] = tc["id"]
                                    if "function" in tc:
                                        fn = tc["function"]
                                        if fn.get("name"):
                                            accumulated_tool_calls[idx]["function"]["name"] = fn["name"]
                                        if "arguments" in fn:
                                            accumulated_tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

    assistant_msg = "".join(full_response)
    assistant_reasoning = "".join(full_reasoning) if full_reasoning else None
    assistant_tool_calls = list(accumulated_tool_calls.values()) if accumulated_tool_calls else None

    if assistant_reasoning:
        print(f"🧠 Stream response 包含 reasoning_content ({len(assistant_reasoning)}字符)")

    if error_body_parts:
        error_text = b"".join(error_body_parts).decode("utf-8", errors="ignore")[:500]
        print(f"❌ 上游错误内容: {error_text}", flush=True)

    if assistant_tool_calls:
        print(f"🔧 Stream response 包含 {len(assistant_tool_calls)} 个工具调用")

    if stream_usage:
        pt = stream_usage.get("prompt_tokens", 0)
        ct = stream_usage.get("completion_tokens", 0)
        tt = stream_usage.get("total_tokens", 0)
        if tt > 0:
            asyncio.create_task(save_token_usage(session_id, model, pt, ct, tt))
            print(f"📊 Stream Token: {pt} + {ct} = {tt}")

    if MEMORY_ENABLED and (user_message or tool_messages):
        asyncio.create_task(
            process_memories_background(session_id, user_message, assistant_msg, model,
                                        context_messages=original_messages, skip_conversation_log=skip_conversation_log,
                                        tool_messages=tool_messages, assistant_tool_calls=assistant_tool_calls,
                                        assistant_reasoning=assistant_reasoning)
        )


@app.get("/import/seed-memories")
async def import_seed_memories():
    try:
        from seed_memories import run_seed_import
        result = await run_seed_import()
        return result
    except ImportError:
        return {"error": "未找到 seed_memories.py，请参考 seed_memories_example.py 创建"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/export/memories")
async def export_memories():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    try:
        memories = await get_all_memories()
        for mem in memories:
            if mem.get("created_at"):
                mem["created_at"] = str(mem["created_at"])
        return {
            "total": len(memories),
            "exported_at": str(__import__("datetime").datetime.now()),
            "memories": memories,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用（设置 MEMORY_ENABLED=true 开启）</h3>")
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/api/memories")
async def api_get_memories(layer: int = None, active_only: bool = None):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    memories = await get_all_memories_detail(layer=layer, active_only=active_only)
    tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))
    for m in memories:
        if m.get("created_at"):
            dt = m["created_at"]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            m["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
    try:
        layer_stats = await get_layer_statistics()
    except Exception:
        layer_stats = None
    result = {"memories": memories}
    if layer_stats:
        result["layer_stats"] = layer_stats
    return result


@app.get("/api/memories/search")
async def api_search_memories(q: str = "", limit: int = 20):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if not q.strip():
        return {"error": "搜索关键词不能为空", "results": []}
    try:
        results = await search_memories(q.strip(), limit)
        tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))
        out = []
        for r in results:
            item = dict(r)
            if item.get("created_at"):
                dt = item["created_at"]
                if hasattr(dt, 'tzinfo'):
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    item["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
            out.append(item)
        return {"results": out, "total": len(out)}
    except Exception as e:
        return {"error": str(e), "results": []}


@app.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    await update_memory_with_layer(
        memory_id,
        content=data.get("content"),
        importance=data.get("importance"),
        title=data.get("title"),
        layer=data.get("layer"),
    )
    return {"status": "ok", "id": memory_id}


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int, soft: bool = False):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if soft:
        await update_memory_with_layer(memory_id, is_active=False)
    else:
        await delete_memory(memory_id)
    return {"status": "ok", "id": memory_id}


@app.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    updates = data.get("updates", [])
    if not updates:
        return {"error": "没有要更新的记忆"}
    for item in updates:
        await update_memory_with_layer(
            item["id"],
            content=item.get("content"),
            importance=item.get("importance"),
            title=item.get("title"),
            layer=item.get("layer"),
        )
    return {"status": "ok", "updated": len(updates)}


@app.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    await delete_memories_batch(ids)
    return {"status": "ok", "deleted": len(ids)}


CONSOLIDATION_PROMPT = """
你是记忆整理助手。请将以下对话碎片整理成完整的事件记录。

要求：
1. 按主题/事件分组，相关的碎片合并到一起
2. 每个事件一条记录，不要太细碎也不要太笼统
3. 每条记录包含：标题（10字内）+ 完整描述
4. 合并重复内容，保留重要细节
5. 保留原文中的主观感受、情绪表达和个人化用语，不要改写为客观陈述或第三方总结
6. content字段中不要使用双引号，用单引号或书名号代替

碎片记忆：
{fragments}

请用 JSON 格式输出：
[
  {{
    "title": "事件标题（10字内）",
    "content": "完整的事件描述",
    "importance": 5,
    "merged_ids": [1, 2, 3]
  }}
]

只输出 JSON，不要其他内容。确保 JSON 语法正确。
"""

_consolidate_status = {
    "running": False,
    "started_at": None,
    "result": None,
    "error": None,
}


async def consolidate_memories_for_date_range(start_date, end_date):
    from datetime import date
    import re

    fragments = await get_fragments_by_date_range(start_date, end_date)

    if not fragments:
        return {"status": "no_fragments", "start_date": str(start_date), "end_date": str(end_date)}

    fragments_text = "\n".join([
        f"[ID={f['id']}] ({f['created_at'].strftime('%m-%d') if hasattr(f['created_at'], 'strftime') else str(f['created_at'])[:10]}) {f['content']}"
        for f in fragments
    ])

    prompt = CONSOLIDATION_PROMPT.format(fragments=fragments_text)
    consolidation_model = os.getenv("MEMORY_MODEL", "") or os.getenv("DEFAULT_MODEL", "anthropic/claude-haiku-4.5")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            last_error = None
            for attempt in range(3):
                response = await client.post(
                    API_BASE_URL,
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={"model": consolidation_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000}
                )

                if response.status_code == 429:
                    wait_time = (attempt + 1) * 10
                    print(f"⚠️ 整理API 429限流，{wait_time}秒后重试（第{attempt+1}次）")
                    last_error = f"429 Too Many Requests (重试{attempt+1}次)"
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    break

                last_error = None
                break

            if last_error:
                return {"status": "error", "error": f"API调用失败: {last_error}"}

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                json_str = json_match.group()
                try:
                    events = json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        events = json.loads(json_str, strict=False)
                    except json.JSONDecodeError:
                        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', json_str)
                        try:
                            events = json.loads(cleaned)
                        except json.JSONDecodeError as e:
                            return {"status": "error", "error": f"JSON解析失败: {e}", "raw": content[:500]}
            else:
                return {"status": "error", "error": "无法解析 AI 返回的 JSON", "raw": content}

            created_count = 0
            for event in events:
                merged_ids = event.get("merged_ids", [])
                if merged_ids:
                    await create_event_memory(
                        title=event.get("title", ""),
                        content=event.get("content", ""),
                        importance=event.get("importance", 5),
                        event_date=start_date,
                        merged_from=merged_ids
                    )
                    created_count += 1

            all_fragment_ids = [f['id'] for f in fragments]
            await deactivate_memories(all_fragment_ids)

            return {
                "status": "ok",
                "start_date": str(start_date),
                "end_date": str(end_date),
                "fragments_processed": len(fragments),
                "events_created": created_count
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/memories/consolidate")
async def api_manual_consolidate(request: Request):
    from datetime import date as date_type

    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    if _consolidate_status.get("running"):
        return {"status": "already_running", "started_at": _consolidate_status.get("started_at")}

    data = await request.json()

    if "date" in data and "start_date" not in data:
        start_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
        end_date = start_date
    else:
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")
        if not start_date_str or not end_date_str:
            return {"error": "请提供开始和结束日期"}
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        if start_date > end_date:
            return {"error": "开始日期不能晚于结束日期"}

    async def _run():
        _consolidate_status.update({"running": True, "started_at": f"{start_date}~{end_date}", "result": None, "error": None})
        try:
            result = await consolidate_memories_for_date_range(start_date, end_date)
            _consolidate_status["result"] = result
        except Exception as e:
            _consolidate_status["error"] = str(e)
        finally:
            _consolidate_status["running"] = False

    asyncio.create_task(_run())
    return {"status": "started", "start_date": str(start_date), "end_date": str(end_date)}


@app.get("/api/memories/consolidate/status")
async def api_consolidate_status():
    return _consolidate_status


@app.post("/api/memories/{memory_id}/promote")
async def api_promote_to_core(memory_id: int, request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    title = data.get("title")
    await promote_to_core(memory_id, title=title)
    return {"status": "ok", "memory_id": memory_id, "layer": 3}


@app.post("/api/memories/merge")
async def api_merge_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    memory_ids = data.get("ids", [])
    new_title = data.get("title", "")
    new_content = data.get("content", "")
    importance = data.get("importance", 5)
    layer = data.get("layer", 2)
    if not memory_ids or not new_content:
        return {"error": "请提供记忆ID列表和合并后内容"}
    new_id = await merge_memories(memory_ids, new_title, new_content, importance, layer)
    return {"status": "ok", "new_id": new_id, "merged": len(memory_ids)}


@app.post("/api/memories/check-duplicate")
async def api_check_duplicate(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    content = data.get("content", "")
    threshold = data.get("threshold", 0.7)
    if not content:
        return {"error": "请提供记忆内容"}
    result = await check_duplicate_memory(content, threshold)
    return result


@app.post("/api/memories/cleanup-fragments")
async def api_cleanup_fragments(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    days = data.get("days", 30)
    try:
        deleted = await cleanup_old_fragments(days)
        return {"status": "ok", "deleted": deleted, "days": days}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/memories/{memory_id}/revert-merge")
async def api_revert_merge(memory_id: int):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        result = await revert_merge(memory_id)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/memories/{memory_id}/restore")
async def api_restore_memory(memory_id: int):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        await update_memory_with_layer(memory_id, is_active=True)
        return {"status": "ok", "id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/memories/layer-stats")
async def api_layer_statistics():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    try:
        stats = await get_layer_statistics()
        return stats
    except Exception as e:
        return {"error": str(e)}


@app.post("/import/text")
async def import_text_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    try:
        data = await request.json()
        lines = data.get("lines", [])
        skip_scoring = data.get("skip_scoring", False)
        if not lines:
            return {"error": "没有找到记忆条目"}
        if skip_scoring:
            scored = [{"content": t, "importance": 5} for t in lines]
        else:
            scored = await score_memories(lines)
        imported = 0
        skipped = 0
        for mem in scored:
            content = mem.get("content", "")
            if not content:
                continue
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
            if existing > 0:
                skipped += 1
                continue
            await save_memory(content=content, importance=mem.get("importance", 5), source_session="text-import")
            imported += 1
        total = await get_all_memories_count()
        return {"status": "done", "imported": imported, "skipped": skipped, "total": total}
    except Exception as e:
        return {"error": str(e)}


@app.post("/import/memories")
async def import_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    try:
        data = await request.json()
        memories = data.get("memories", [])
        if not memories:
            return {"error": "没有找到记忆数据，请确认 JSON 格式正确"}
        imported = 0
        skipped = 0
        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
            if existing > 0:
                skipped += 1
                continue
            await save_memory(content=content, importance=mem.get("importance", 5), source_session=mem.get("source_session", "json-import"))
            imported += 1
        total = await get_all_memories_count()
        return {"status": "done", "imported": imported, "skipped": skipped, "total": total}
    except Exception as e:
        return {"error": str(e)}
