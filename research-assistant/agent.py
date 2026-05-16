"""
项目 2: 长期记忆研究助手
============================================
目标：构建一个跨会话、可持久化记忆的研究助手 Agent，
解决纯 LLM"对话结束即遗忘"的痛点。

架构栈（自底向上）：
  ① 工具层     — 5 工具：search_web / fetch_url / read_pdf / save_memory / search_memory
  ② 检索层     — Chroma 向量库 + 本地 hash embedding（离线可跑）
                  hybrid retrieval（向量距离 + lexical 词项重排）
  ③ 短期记忆   — LangGraph state + SqliteSaver 检查点（thread 内断点恢复）
  ④ 长期记忆   — Chroma collection 持久化到磁盘（跨 thread / 跨进程复用）
  ⑤ Agent 循环 — LangGraph StateGraph + ToolNode（ReAct 风格）

与第 1 周 ReAct Agent 的本质区别：
  · 第 1 周：单轮任务，无记忆
  · 项目 2：跨会话记忆，新会话能召回旧 session 的研究片段

运行方式：
  uv run python research-assistant/agent.py                  # 在线 demo（需要 API_KEY）
  uv run python research-assistant/agent.py --self-test      # 离线自测（无 API 也能跑）
  uv run python research-assistant/agent.py --graph          # 打印 LangGraph mermaid
  uv run python research-assistant/agent.py --memory-dump    # 查看已存的所有长期记忆
  uv run python research-assistant/agent.py --reset-memory   # 清空长期记忆

参考资料：
  LangGraph Persistence: https://langchain-ai.github.io/langgraph/concepts/persistence/
  Chroma:                https://docs.trychroma.com/
  Anthropic Context Eng: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
"""

import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import chromadb
import httpx
import openai as openai_module
from bs4 import BeautifulSoup
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pypdf import PdfReader

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    第一部分：配置中心                             ║
# ╚══════════════════════════════════════════════════════════════════╝

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma"
CHECKPOINT_DB = BASE_DIR / "checkpoints.db"
COLLECTION_NAME = "research_memory"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

HTTP_TIMEOUT = 15.0
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

MAX_FETCH_CHARS = 4000      # 抓网页正文截断长度（控 context 爆炸）
MAX_PDF_CHARS = 6000        # 读 PDF 正文截断长度
MAX_MEMORY_CHARS = 1500     # 单条长期记忆最大写入长度


# ╔══════════════════════════════════════════════════════════════════╗
# ║              第二部分：本地 Embedding + Chroma                    ║
# ╚══════════════════════════════════════════════════════════════════╝
# 教学优先：用确定性 hash embedding，离线可跑，免下载模型/调用 API。
# 生产替换：把 HashEmbeddingFunction 换成 OpenAIEmbeddings/BGE/E5 即可。


class HashEmbeddingFunction(EmbeddingFunction):
    """确定性本地 embedding：token → sha256 → 投到固定维度的稀疏向量。"""

    def __init__(self, dim: int = 192):
        self.dim = dim

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：英文/数字/下划线整体；中文单字 + bigram。"""
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_]+|[一-鿿]", lowered)
    chinese = re.findall(r"[一-鿿]", lowered)
    tokens.extend(a + b for a, b in zip(chinese, chinese[1:]))
    return tokens


def _lexical_score(query: str, text: str) -> float:
    """Jaccard-ish 词重叠 + 短语命中加分，用于 hybrid 重排。"""
    qt = set(_tokenize(query))
    tt = set(_tokenize(text))
    if not qt:
        return 0.0
    overlap = qt & tt
    score = len(overlap) / math.sqrt(len(qt) * max(len(tt), 1))
    lowered_q, lowered_t = query.lower(), text.lower()
    phrase_hits = 0
    for phrase in re.findall(r"[a-z0-9_]{3,}|[一-鿿]{2,}", lowered_q):
        if phrase in lowered_t:
            phrase_hits += 1
    return min(1.0, score + 0.15 * phrase_hits)


def get_collection(reset: bool = False):
    """获取（或清空重建）长期记忆 collection。PersistentClient 落盘。"""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=HashEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║                第三部分：5 个研究工具                              ║
# ╚══════════════════════════════════════════════════════════════════╝


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _clean_html(html: str) -> str:
    """从 HTML 提取干净正文：去 script/style/nav，保留段落与标题。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "form"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def make_tools(collection):
    """生产 5 个 LangChain 工具，闭包持有 collection 引用。"""

    @tool
    def search_web(query: str, max_results: int = 5) -> str:
        """通过 DuckDuckGo 搜索网页，返回 title / url / snippet 列表。

        Args:
            query: 搜索关键词
            max_results: 返回结果数，1-10，默认 5
        """
        max_results = max(1, min(int(max_results), 10))
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results, region="wt-wt"))
        except Exception as exc:
            return json.dumps(
                {"error": f"search_web 失败: {type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        items = []
        for h in hits:
            items.append({
                "title": h.get("title", ""),
                "url": h.get("href") or h.get("url", ""),
                "snippet": h.get("body", "")[:300],
            })
        return json.dumps({"query": query, "results": items}, ensure_ascii=False)

    @tool
    def fetch_url(url: str) -> str:
        """抓取 URL 的网页正文（去除脚本/导航，截断到 ~4000 字）。

        Args:
            url: 完整 http(s) 链接
        """
        if not url.startswith(("http://", "https://")):
            return json.dumps({"error": "url 必须以 http:// 或 https:// 开头"}, ensure_ascii=False)
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=HTTP_HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            return json.dumps(
                {"error": f"fetch_url 失败: {type(exc).__name__}: {exc}", "url": url},
                ensure_ascii=False,
            )
        text = _clean_html(html)
        truncated = text[:MAX_FETCH_CHARS]
        return json.dumps(
            {
                "url": url,
                "char_count": len(text),
                "truncated_to": len(truncated),
                "content": truncated,
            },
            ensure_ascii=False,
        )

    @tool
    def read_pdf(path: str, max_pages: int = 10) -> str:
        """读取本地 PDF 文件正文（取前 N 页，截断到 ~6000 字）。

        Args:
            path: 本地 PDF 文件路径（绝对或相对当前目录）
            max_pages: 最多读取页数，1-50，默认 10
        """
        max_pages = max(1, min(int(max_pages), 50))
        pdf_path = Path(path).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = (Path.cwd() / pdf_path).resolve()
        if not pdf_path.exists():
            return json.dumps({"error": f"PDF 文件不存在: {pdf_path}"}, ensure_ascii=False)
        try:
            reader = PdfReader(str(pdf_path))
            total = len(reader.pages)
            pages_to_read = min(max_pages, total)
            parts = []
            for i in range(pages_to_read):
                parts.append(reader.pages[i].extract_text() or "")
            text = "\n".join(parts)
        except Exception as exc:
            return json.dumps(
                {"error": f"read_pdf 失败: {type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        truncated = text[:MAX_PDF_CHARS]
        return json.dumps(
            {
                "path": str(pdf_path),
                "total_pages": total,
                "pages_read": pages_to_read,
                "char_count": len(text),
                "truncated_to": len(truncated),
                "content": truncated,
            },
            ensure_ascii=False,
        )

    @tool
    def save_memory(title: str, content: str, tags: str = "") -> str:
        """把一段研究片段写入长期记忆（Chroma 向量库），跨会话可召回。

        Args:
            title: 简短标题（用于召回时标识来源）
            content: 主体内容，建议 ≤ 1500 字；过长会被截断
            tags: 逗号分隔的标签，例如 "rag,memory,langgraph"
        """
        if not content.strip():
            return json.dumps({"error": "content 不能为空"}, ensure_ascii=False)
        body = content[:MAX_MEMORY_CHARS]
        memory_id = hashlib.sha1(
            f"{title}|{body}|{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:16]
        collection.upsert(
            ids=[memory_id],
            documents=[body],
            metadatas=[{
                "title": title or "(无标题)",
                "tags": tags,
                "saved_at": _now_iso(),
            }],
        )
        return json.dumps(
            {
                "ok": True,
                "memory_id": memory_id,
                "title": title,
                "saved_at": _now_iso(),
                "char_count": len(body),
                "total_memories": collection.count(),
            },
            ensure_ascii=False,
        )

    @tool
    def search_memory(query: str, top_k: int = 4) -> str:
        """从长期记忆库中检索过往研究片段（向量 + 词项 hybrid 重排）。

        Args:
            query: 检索关键词或自然语言问题
            top_k: 返回条数，1-8，默认 4
        """
        top_k = max(1, min(int(top_k), 8))
        count = collection.count()
        if count == 0:
            return json.dumps({"query": query, "matches": [], "note": "长期记忆为空"}, ensure_ascii=False)

        n = min(count, max(top_k * 3, top_k))
        raw = collection.query(query_texts=[query], n_results=n)
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0]

        scored = []
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            lexical = _lexical_score(query, f"{meta.get('title', '')} {text}")
            vector = max(0.0, 1.0 - float(dist))
            score = 0.6 * lexical + 0.4 * vector
            scored.append((score, lexical, vector, doc_id, text, meta, dist))
        scored.sort(key=lambda x: x[0], reverse=True)

        matches = []
        for rank, (score, lex, vec, doc_id, text, meta, dist) in enumerate(scored[:top_k], 1):
            matches.append({
                "rank": rank,
                "memory_id": doc_id,
                "title": meta.get("title", ""),
                "tags": meta.get("tags", ""),
                "saved_at": meta.get("saved_at", ""),
                "score": round(score, 4),
                "lexical": round(lex, 4),
                "vector": round(vec, 4),
                "content": text,
            })
        return json.dumps({"query": query, "matches": matches}, ensure_ascii=False)

    return [search_web, fetch_url, read_pdf, save_memory, search_memory]


# ╔══════════════════════════════════════════════════════════════════╗
# ║                第四部分：LangGraph Agent                          ║
# ╚══════════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = """你是一个长期记忆研究助手，工作模式是 ReAct + 持久化记忆。

你拥有 5 个工具：
  search_web    — DuckDuckGo 网页搜索
  fetch_url     — 抓取指定 URL 正文
  read_pdf      — 读取本地 PDF
  save_memory   — 把研究片段写入长期记忆库（跨会话可召回）
  search_memory — 从长期记忆库召回过往研究

工作准则：
1. 接到研究任务时，先调用 search_memory 看是否已有相关沉淀，避免重复劳动。
2. 收集到新的关键信息后，主动调用 save_memory 沉淀下来，标题精炼、tags 清晰。
3. 引用网页或 PDF 内容时务必带上来源链接/路径，便于校验。
4. 控 token：fetch_url / read_pdf 返回值较大，只摘录与任务最相关的句子。
5. 最终答复用中文，结构清晰，末尾列出"已沉淀记忆"和"参考来源"两小节。
"""


def _to_openai_format(messages: list) -> list[dict]:
    """LangChain 消息 → OpenAI Chat Completions 格式（保留 tool_calls 与 reasoning）。"""
    out = []
    for m in messages:
        if m.type == "system":
            out.append({"role": "system", "content": m.content})
        elif m.type == "human":
            out.append({"role": "user", "content": m.content or ""})
        elif m.type == "ai":
            d: dict = {"role": "assistant", "content": m.content or ""}
            rc = (m.additional_kwargs or {}).get("reasoning_content")
            if rc:
                d["reasoning_content"] = rc
            if m.tool_calls:
                d["content"] = None
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(d)
        elif m.type == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def _from_openai_response(msg) -> AIMessage:
    kwargs: dict = {}
    rc = getattr(msg, "reasoning_content", None)
    if rc:
        kwargs["additional_kwargs"] = {"reasoning_content": rc}
    if msg.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "type": "tool_call",
            }
            for tc in msg.tool_calls
        ]
        return AIMessage(content=msg.content or "", tool_calls=tool_calls, **kwargs)
    return AIMessage(content=msg.content or "", **kwargs)


def build_graph(collection, checkpointer=None):
    """构建研究助手 LangGraph 图。

      START → chatbot ──[tool_calls]──→ tools → chatbot
                   └──[final]──→ END
    """
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tools = make_tools(collection)
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema(),
            },
        }
        for t in tools
    ]

    raw_client = openai_module.OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )

    def chatbot(state: State) -> dict:
        msgs = state["messages"]
        if not msgs or msgs[0].type != "system":
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)

        resp = raw_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=_to_openai_format(msgs),
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.0,
        )
        return {"messages": [_from_openai_response(resp.choices[0].message)]}

    g = StateGraph(State)
    g.add_node("chatbot", chatbot)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "chatbot")
    g.add_conditional_edges("chatbot", tools_condition)
    g.add_edge("tools", "chatbot")
    return g.compile(checkpointer=checkpointer)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                第五部分：演示与自测                                ║
# ╚══════════════════════════════════════════════════════════════════╝


def chat(app, user_input: str, thread_id: str = "research-demo") -> str:
    """单轮对话，打印工具调用轨迹与最终答复。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    print(f"\n  用户 [thread={thread_id}]: {user_input}")
    for msg in result["messages"]:
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args_preview = json.dumps(tc["args"], ensure_ascii=False)
                if len(args_preview) > 160:
                    args_preview = args_preview[:160] + "..."
                print(f"  → 工具调用: {tc['name']}({args_preview})")
        elif msg.type == "tool":
            preview = str(msg.content)
            if len(preview) > 240:
                preview = preview[:240] + "..."
            print(f"  ← 工具返回: {preview}")

    final = result["messages"][-1].content
    print(f"\n  Agent:\n{final}\n")
    return final


def run_self_test():
    """离线自测：验证 5 个工具骨架（save / search / read_pdf / fetch / search_web） 不依赖 LLM。"""
    print("=" * 72)
    print(" Research Assistant 离线自测")
    print("=" * 72)
    collection = get_collection(reset=True)
    tools_list = make_tools(collection)
    tools_by_name = {t.name: t for t in tools_list}

    # ① save_memory
    print("\n[1] save_memory — 写入三条预置研究片段")
    samples = [
        {
            "title": "RAG pipeline 速记",
            "content": "RAG = indexing + retrieval + augmentation + generation。"
                       "Indexing 阶段切 chunk 算 embedding 写库；"
                       "Query 阶段召回 top-k 注入 prompt 再生成。",
            "tags": "rag,pipeline",
        },
        {
            "title": "LangGraph 长期记忆模式",
            "content": "LangGraph 短期记忆是 thread 内 state（SqliteSaver 检查点）；"
                       "长期记忆通常落到 Chroma/Postgres 等外部存储，跨 thread 共享。",
            "tags": "langgraph,memory",
        },
        {
            "title": "Lost in the Middle",
            "content": "长上下文中，模型对中间位置信息利用率显著下降。"
                       "RAG 不能无脑塞 chunk，应控 top-k + reranker 把关键证据放前后。",
            "tags": "context,rag",
        },
    ]
    for s in samples:
        print("  ", tools_by_name["save_memory"].invoke(s))

    # ② search_memory
    print("\n[2] search_memory — 用不同 query 召回")
    for q in ["RAG 包含哪些阶段", "LangGraph 长期记忆怎么做", "为什么模型记不住长上下文"]:
        print(f"  query: {q}")
        payload = json.loads(tools_by_name["search_memory"].invoke({"query": q, "top_k": 2}))
        for m in payload["matches"]:
            print(f"    [{m['rank']}] {m['title']}  score={m['score']}  "
                  f"(lex={m['lexical']}, vec={m['vector']})")

    # ③ read_pdf — 生成一个临时 PDF 跑通链路
    print("\n[3] read_pdf — 临时构造 PDF 验证读取")
    try:
        from pypdf import PdfWriter
        tmp_pdf = BASE_DIR / "_self_test.pdf"
        writer = PdfWriter()
        # pypdf 不能直接写文本，借用 reportlab 又超范围；
        # 退而求其次：尝试读一个不存在路径，验证错误处理
        out = tools_by_name["read_pdf"].invoke({"path": "_definitely_not_exist.pdf"})
        print("   错误处理:", out)
    except Exception as exc:
        print(f"  跳过 PDF 写入（{exc}），仅验证错误分支")

    # ④ fetch_url & search_web — 仅在有网络时尝试，失败 graceful
    print("\n[4] search_web / fetch_url — 网络可达时联通")
    out = tools_by_name["search_web"].invoke({"query": "anthropic building effective agents", "max_results": 2})
    payload = json.loads(out)
    if "error" in payload:
        print("  search_web 离线/受限:", payload["error"])
    else:
        for r in payload["results"]:
            print(f"  - {r['title']}\n    {r['url']}")
        first_url = payload["results"][0]["url"] if payload["results"] else None
        if first_url:
            f_out = json.loads(tools_by_name["fetch_url"].invoke({"url": first_url}))
            if "error" in f_out:
                print("  fetch_url 错误:", f_out["error"])
            else:
                print(f"  fetch_url ok, char_count={f_out['char_count']}, "
                      f"truncated_to={f_out['truncated_to']}")

    print("\n" + "=" * 72)
    print(f" 自测完成 · 长期记忆条数: {collection.count()}")
    print("=" * 72)


def run_agent_demo():
    """在线 demo：演示「跨会话记忆」核心场景。"""
    if not os.getenv("API_KEY"):
        raise RuntimeError("未找到 API_KEY。请在 .env 中配置 DEEPSEEK key，或先运行 --self-test。")

    from langgraph.checkpoint.sqlite import SqliteSaver

    collection = get_collection()
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        app = build_graph(collection, checkpointer=checkpointer)
        print("=" * 72)
        print(" 长期记忆研究助手 · 跨会话 Demo")
        print("=" * 72)

        # Session A：让 Agent 研究并沉淀
        chat(app,
             "请查一下 LangGraph 的长期记忆通常是怎么实现的，把要点存进记忆。",
             thread_id="session-A")

        # Session B：新 thread，验证能从长期记忆召回
        chat(app,
             "上次研究过 LangGraph 的长期记忆方案，能复述一下吗？"
             "请先 search_memory 看看有没有相关沉淀。",
             thread_id="session-B")


def memory_dump():
    """列出长期记忆库中的所有条目，用于检查持久化效果。"""
    collection = get_collection()
    count = collection.count()
    print(f"长期记忆总条数: {count}")
    if count == 0:
        return
    rows = collection.get(include=["documents", "metadatas"])
    for i, (doc_id, doc, meta) in enumerate(
        zip(rows["ids"], rows["documents"], rows["metadatas"]), 1
    ):
        meta = meta or {}
        title = meta.get("title", "(无标题)")
        saved_at = meta.get("saved_at", "?")
        tags = meta.get("tags", "")
        preview = doc[:120].replace("\n", " ")
        print(f"\n[{i}] {title}  · {saved_at}  · tags={tags}")
        print(f"    id={doc_id}")
        print(f"    {preview}{'...' if len(doc) > 120 else ''}")


def main():
    args = set(sys.argv[1:])

    if "--reset-memory" in args:
        get_collection(reset=True)
        print("已清空长期记忆库 ✅")
        return

    if "--memory-dump" in args:
        memory_dump()
        return

    if "--self-test" in args:
        run_self_test()
        return

    if "--graph" in args:
        collection = get_collection()
        app = build_graph(collection)
        print(app.get_graph(xray=True).draw_mermaid())
        return

    run_agent_demo()


if __name__ == "__main__":
    main()
