"""
Agent 动手任务 - 2026-05-15 (第2周 周五)
============================================
主题：RAG 基础 — embedding / chunking / 混合检索 / LangGraph 接入
工具：Chroma 向量检索(retrieve_context)
API：DeepSeek API（OpenAI 兼容接口）

核心概念（配合八股题 18、19、20 食用）：
  - RAG pipeline: Indexing → Retrieval → Augmentation → Generation
  - Chunking 决定召回颗粒度，过大噪声多，过小上下文断裂
  - Embedding 负责语义召回；关键词/BM25 负责精确词匹配；混合检索互补
  - LangGraph 中把检索封装成 tool，让 Agent 自主决定何时查知识库

运行方式：
  uv run python exercises/w2d5-rag-agent/agent.py              # RAG Agent demo（需要 API_KEY）
  uv run python exercises/w2d5-rag-agent/agent.py --self-test  # 只测试 Chroma 检索（离线）
  uv run python exercises/w2d5-rag-agent/agent.py --explain    # 打印 RAG 八股速记
  uv run python exercises/w2d5-rag-agent/agent.py --graph      # 打印 LangGraph 拓扑
  uv run python exercises/w2d5-rag-agent/agent.py --reset-index # 重建 Chroma 索引

参考资料：
  Chroma Quickstart: https://docs.trychroma.com/docs/overview/getting-started
  LangGraph Tools:   https://langchain-ai.github.io/langgraph/how-tos/tool-calling/
"""

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Annotated

import chromadb
import openai as openai_module
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma"
CHECKPOINT_DB = BASE_DIR / "checkpoints.db"
COLLECTION_NAME = "rag_agent_notes"


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第一部分：小型知识库（今天的 RAG 笔记，按语义块组织）          ║
# ╚══════════════════════════════════════════════════════════════════╝

KNOWLEDGE_DOCS = [
    {
        "id": "rag-pipeline",
        "source": "八股题18",
        "title": "RAG pipeline",
        "text": (
            "RAG 的完整链路是 indexing、retrieval、augmentation、generation。"
            "Indexing 阶段把文档清洗、切 chunk、算 embedding、写入向量库。"
            "Query 阶段把用户问题向量化，召回 top-k chunk，再把 chunk 作为 context 注入 prompt，"
            "最后由 LLM 基于检索证据回答。核心收益是把可更新知识放在外部存储，降低幻觉并支持引用。"
        ),
    },
    {
        "id": "chunking",
        "source": "八股题19",
        "title": "Chunking strategy",
        "text": (
            "Chunking 的目标是在语义完整和检索精度之间折中。chunk 太大，召回内容噪声多，"
            "会稀释真正相关信息；chunk 太小，句子和段落被切断，LLM 拿不到完整因果链。"
            "常用策略是按标题、段落、代码块等结构切分，再加 10%-20% overlap 保留上下文。"
        ),
    },
    {
        "id": "embedding",
        "source": "八股题18",
        "title": "Embedding retrieval",
        "text": (
            "Embedding 检索把文本映射到稠密向量空间，适合语义相似问题，比如“怎么降低幻觉”"
            "能召回“citation、grounding、retrieval confidence”。缺点是对精确词、编号、错误码、"
            "专有名词不如关键词检索稳定，所以生产 RAG 很少只靠纯向量。"
        ),
    },
    {
        "id": "hybrid-retrieval",
        "source": "八股题20",
        "title": "Hybrid retrieval",
        "text": (
            "混合检索通常把 BM25/关键词召回与 embedding 语义召回合并。关键词检索擅长精确匹配术语、"
            "函数名、错误码；向量检索擅长同义改写和语义泛化。工程上常见做法是两路各取 top-k，"
            "用 RRF 或加权分数融合，再交给 reranker 重排。"
        ),
    },
    {
        "id": "reranker",
        "source": "八股题21",
        "title": "Reranker",
        "text": (
            "Reranker 是召回后的二阶段排序模型。向量库先粗召回几十个候选 chunk，reranker 再根据"
            "query 与 chunk 的细粒度相关性打分，选出最值得放进 prompt 的少量证据。它能明显提升"
            "答案 grounding，但会增加延迟和成本。"
        ),
    },
    {
        "id": "lost-in-middle",
        "source": "八股题22",
        "title": "Lost in the Middle",
        "text": (
            "Lost in the Middle 指模型对长上下文中间位置的信息利用率下降。RAG 不能无脑塞很多 chunk，"
            "应该控制 top-k、做去重和重排，把最关键证据放在更靠前或更靠后的位置，并要求回答时引用证据。"
        ),
    },
    {
        "id": "agent-memory",
        "source": "八股题17",
        "title": "Short-term and long-term memory",
        "text": (
            "Agent 记忆可分为短期记忆和长期记忆。短期记忆通常是当前对话 history 或 LangGraph state；"
            "长期记忆通常落在外部数据库、向量库或文件系统里，可以跨 session 恢复。RAG 是长期记忆"
            "最常见的工程形态之一。"
        ),
    },
    {
        "id": "chroma",
        "source": "Chroma quickstart",
        "title": "Chroma collection",
        "text": (
            "Chroma 的核心抽象是 collection。应用把 documents、ids、metadatas 和 embeddings 写入 collection，"
            "查询时用 query_texts 或 query_embeddings 返回 documents、metadatas、distances。"
            "本练习使用 PersistentClient 把索引落盘，方便后续跨进程复用。"
        ),
    },
    {
        "id": "langgraph-rag",
        "source": "LangGraph tool calling",
        "title": "RAG as a tool",
        "text": (
            "在 LangGraph 里，RAG 最简单的接入方式是把检索封装成 retrieve_context 工具。"
            "LLM 节点判断是否需要外部知识，发起 tool_call；ToolNode 执行检索后把证据作为 ToolMessage"
            "写回 state；LLM 再基于证据生成最终回答。"
        ),
    },
    {
        "id": "rag-eval",
        "source": "RAG evaluation",
        "title": "RAG evaluation",
        "text": (
            "RAG 评估至少分三层：检索层看 recall@k、MRR、命中率；生成层看 groundedness、faithfulness、"
            "citation 是否正确；端到端看用户任务成功率。调优时要先判断是没召回、召回排序差，还是 LLM"
            "没有正确使用证据。"
        ),
    },
]


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第二部分：Embedding + Chroma 索引                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class HashEmbeddingFunction(EmbeddingFunction):
    """确定性本地 embedding，用于教学与离线测试。

    生产环境应替换为真实 embedding 模型；这里用 token hashing 避免下载模型，
    让 `--self-test` 在没有网络和 API key 的环境下也能跑通 Chroma 链路。
    """

    def __init__(self, dim: int = 128):
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
    """同时覆盖英文术语、数字、中文单字和中文 bigram。"""
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.extend(a + b for a, b in zip(chinese_chars, chinese_chars[1:]))
    return tokens


def build_collection(reset: bool = False):
    """创建/加载 Chroma collection，并写入今天的 RAG 知识块。"""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    embedding_fn = HashEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [doc["id"] for doc in KNOWLEDGE_DOCS]
    collection.upsert(
        ids=ids,
        documents=[doc["text"] for doc in KNOWLEDGE_DOCS],
        metadatas=[
            {"source": doc["source"], "title": doc["title"]}
            for doc in KNOWLEDGE_DOCS
        ],
    )
    return collection


def search_knowledge(collection, query: str, top_k: int = 4) -> list[dict]:
    """查询 Chroma，并用关键词信号做轻量混合重排。

    Chroma 负责向量距离；本地 lexical score 负责精确词、英文术语、
    编号和中文关键词。这里用加权融合演示 hybrid retrieval 的思想。
    """
    top_k = max(1, min(int(top_k), 8))
    n_results = min(collection.count(), max(top_k * 3, top_k))
    vector_result = collection.query(query_texts=[query], n_results=n_results)

    candidates = []
    documents = vector_result.get("documents", [[]])[0]
    metadatas = vector_result.get("metadatas", [[]])[0]
    distances = vector_result.get("distances", [[]])[0]
    ids = vector_result.get("ids", [[]])[0]

    for doc_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        lexical = _lexical_score(query, f"{meta.get('title', '')} {text}")
        vector = max(0.0, 1.0 - float(distance))
        final_score = 0.65 * lexical + 0.35 * vector
        candidates.append((final_score, lexical, vector, doc_id, text, meta, distance))

    candidates.sort(key=lambda item: item[0], reverse=True)

    rows = []
    for rank, (score, lexical, vector, doc_id, text, meta, distance) in enumerate(
        candidates[:top_k],
        start=1,
    ):
        rows.append({
            "rank": rank,
            "id": doc_id,
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "score": round(float(score), 4),
            "lexical": round(float(lexical), 4),
            "vector": round(float(vector), 4),
            "distance": round(float(distance), 4),
            "text": text,
        })
    return rows


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    if not query_tokens:
        return 0.0

    overlap = query_tokens & text_tokens
    score = len(overlap) / math.sqrt(len(query_tokens) * max(len(text_tokens), 1))

    lowered_query = query.lower()
    lowered_text = text.lower()
    phrase_hits = 0
    for phrase in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", lowered_query):
        if phrase in lowered_text:
            phrase_hits += 1
    return min(1.0, score + 0.18 * phrase_hits)


def format_results(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"[{row['rank']}] {row['title']} ({row['source']}, "
            f"score={row['score']}, vector={row['vector']}, lexical={row['lexical']})\n"
            f"    {row['text']}"
        )
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第三部分：LangGraph Agent，把 RAG 封装成工具                    ║
# ╚══════════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = """你是一个 RAG 学习助手，负责回答 Agent、RAG、记忆系统相关问题。

你拥有 retrieve_context 工具，可从今天的 RAG 知识库中检索证据。
行为准则：
- 涉及 RAG、chunking、embedding、混合检索、reranker、长期记忆时，必须先调用 retrieve_context
- 回答必须基于工具返回的证据，避免编造
- 最终答案用中文，结构清晰，末尾列出使用的证据来源
- 如果检索结果不足，明确说“知识库证据不足”，再给出有限回答"""


def make_tools(collection):
    @tool
    def retrieve_context(query: str, top_k: int = 4) -> str:
        """从本地 Chroma 知识库检索 RAG/Agent 相关资料。

        Args:
            query: 用户问题或检索关键词
            top_k: 返回的证据块数量，默认 4，最大 8
        """
        rows = search_knowledge(collection, query=query, top_k=top_k)
        return json.dumps({"query": query, "matches": rows}, ensure_ascii=False)

    return [retrieve_context]


def _to_openai_format(messages: list) -> list[dict]:
    """将 LangChain 消息列表转换为 OpenAI API 格式，保留 reasoning_content。"""
    result = []
    for msg in messages:
        if msg.type == "system":
            result.append({"role": "system", "content": msg.content})
        elif msg.type == "human":
            result.append({"role": "user", "content": msg.content or ""})
        elif msg.type == "ai":
            d: dict = {"role": "assistant", "content": msg.content or ""}
            rc = (msg.additional_kwargs or {}).get("reasoning_content")
            if rc:
                d["reasoning_content"] = rc
            if msg.tool_calls:
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
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif msg.type == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
    return result


def _from_openai_response(msg) -> AIMessage:
    """将 OpenAI 响应消息转为 LangChain AIMessage，保留工具调用。"""
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
    """构建 LangGraph RAG Agent 图。

      START → chatbot ──[tool_calls]──→ tools → chatbot
                   └──[final answer]──→ END
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
        base_url="https://api.deepseek.com",
    )

    def chatbot(state: State) -> dict:
        messages = state["messages"]
        if not messages or messages[0].type != "system":
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = raw_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=_to_openai_format(messages),
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.0,
        )
        return {"messages": [_from_openai_response(response.choices[0].message)]}

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")
    return graph.compile(checkpointer=checkpointer)


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第四部分：演示与测试                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def chat(app, user_input: str, thread_id: str = "rag-demo") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    final = result["messages"][-1].content

    print(f"  用户: {user_input}")
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  → 工具调用: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
        elif msg.type == "tool":
            payload = json.loads(msg.content)
            print("  ← 检索结果:")
            print(format_results(payload["matches"]))
    print(f"\n  Agent:\n{final}\n")
    return final


def run_self_test(collection):
    queries = [
        "RAG pipeline 包含哪些阶段？",
        "chunk 太大和太小分别有什么问题？",
        "为什么要混合检索？",
        "LangGraph 怎么接入 RAG？",
    ]

    print("=" * 72)
    print("Chroma RAG 检索自测（离线）")
    print("=" * 72)
    print(f"知识块数量: {collection.count()}")

    for query in queries:
        print("\n" + "-" * 72)
        print(f"Query: {query}")
        rows = search_knowledge(collection, query=query, top_k=3)
        print(format_results(rows))


def run_agent_demo(collection):
    if not os.getenv("API_KEY"):
        raise RuntimeError("未找到 API_KEY。请在 .env 中配置 API_KEY，或先运行 --self-test 离线验证检索。")

    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        app = build_graph(collection, checkpointer=checkpointer)
        print("=" * 72)
        print("LangGraph + Chroma RAG Agent")
        print("=" * 72)
        chat(app, "用 5 句话解释 RAG pipeline，并说明 chunking 为什么重要。", thread_id="rag-demo-1")
        chat(app, "混合检索和 reranker 分别解决什么问题？", thread_id="rag-demo-2")


def print_explain():
    print("""
RAG 八股速记
============
1. Pipeline:
   Indexing: 文档清洗 → chunking → embedding → 向量库
   Query: 问题向量化 → top-k 检索 → context 注入 → LLM 生成

2. Chunking:
   过大：噪声多、相关信息被稀释、占 context
   过小：语义断裂、跨段因果丢失
   常用：按标题/段落/代码块切分 + overlap

3. 混合检索:
   BM25/关键词负责精确词、函数名、错误码
   Embedding 负责语义相似和同义改写
   两路召回后用 RRF/加权融合，再用 reranker 重排

4. Agent 记忆:
   短期记忆 = 当前 state/history
   长期记忆 = 向量库/数据库/文件系统
   RAG 是长期记忆最常见的落地方式
""".strip())


def main():
    reset = "--reset-index" in sys.argv
    collection = build_collection(reset=reset)

    if "--self-test" in sys.argv:
        run_self_test(collection)
        return

    if "--explain" in sys.argv:
        print_explain()
        return

    if "--graph" in sys.argv:
        app = build_graph(collection)
        print(app.get_graph(xray=True).draw_mermaid())
        return

    run_agent_demo(collection)


if __name__ == "__main__":
    main()
