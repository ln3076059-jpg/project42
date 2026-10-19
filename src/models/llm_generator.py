"""
llm_generator.py  —  v4.0 (Full Groq + Local ONNX Embeddings)
- Sử dụng 100% Groq API cho mọi tác vụ sinh văn bản (Chat, RAG, Tóm tắt, Thuật ngữ, Web Search).
- Sử dụng mô hình nhúng cục bộ all-MiniLM-L6-v2 (ChromaDB ONNX) chạy 100% offline, không giới hạn rate limit.
"""

import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq, AsyncGroq
from chromadb.utils import embedding_functions

load_dotenv()

# Kiểu alias
RetrievedChunk = Dict[str, Any]
AnswerDict     = Dict[str, Any]
CitationDict   = Dict[str, Any]

# ── GROQ CONFIGURATION ───────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("⚠️ CHƯA CÓ GROQ API KEY: Vui lòng kiểm tra lại biến GROQ_API_KEY trong file .env")

DEFAULT_LLM_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CONTEXT_CHARS   = 20000

# Singleton Groq clients
_groq_sync_client: Optional[Groq] = None
_groq_async_client: Optional[AsyncGroq] = None
_embed_function = None


def _get_groq_sync_client() -> Groq:
    global _groq_sync_client
    if _groq_sync_client is None:
        _groq_sync_client = Groq(api_key=GROQ_API_KEY)
    return _groq_sync_client


def _get_groq_async_client() -> AsyncGroq:
    global _groq_async_client
    if _groq_async_client is None:
        _groq_async_client = AsyncGroq(api_key=GROQ_API_KEY)
    return _groq_async_client


def _resolve_model(model_name: Optional[str]) -> str:
    """Nếu model_name truyền vào là gemini hoặc rỗng, tự động chuyển về mô hình Groq mặc định."""
    if not model_name or "gemini" in model_name.lower():
        return DEFAULT_LLM_MODEL
    return model_name


# ════════════════════════════════════════════════════════════════════════════
# TASK 3.1 — Sinh câu trả lời có trích dẫn (Groq LLM)
# ════════════════════════════════════════════════════════════════════════════

def generate_cited_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0
) -> AnswerDict:
    model_to_use = _resolve_model(model_name)
    context = _build_context(retrieved_chunks)
    prompt = _build_cited_prompt(question, context)

    client = _get_groq_sync_client()
    try:
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là một trợ lý hỏi đáp tài liệu học tập trung thực, chính xác và chuyên nghiệp.\n"
                        "QUY TẮC BẮT BUỘC:\n"
                        "1. ĐÁP ÁN BẮT BUỘC DỰA VÀO CÁC TÀI LIỆU CÓ SẴN TRONG NGỮ CẢNH. TUYỆT ĐỐI KHÔNG tự suy diễn hoặc bịa đặt thông tin ngoài tài liệu. Nếu trong tài liệu không có thông tin để trả lời câu hỏi, hãy nói rõ: 'Dựa trên các tài liệu đã cung cấp, không có đủ thông tin để trả lời câu hỏi này.'\n"
                        "2. Chèn nhãn trích dẫn [Nguồn X] tương ứng ở cuối mỗi luận điểm hoặc đoạn thông tin trích xuất từ tài liệu.\n"
                        "3. Trình bày bằng tiếng Việt rõ ràng, giải thích cặn kẽ, chuẩn xác theo giáo trình."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=3000
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as e:
        raw_answer = f"⚠️ Lỗi từ máy chủ Groq AI: {str(e)}"

    citations = _extract_citations(retrieved_chunks, raw_answer)
    clean_answer = re.sub(r'\s*\[Nguồn \d+\]|\s*\[\d+\]', '', raw_answer)

    # Fallback citation
    if not citations and retrieved_chunks and "⚠️ Lỗi" not in raw_answer:
        best = retrieved_chunks[0]
        meta = best.get('metadata', {})
        content = best.get('content', '')
        citations.append({
            "file_name": meta.get('file_name', 'Tài liệu không tên'),
            "page": meta.get('page', 1),
            "heading": meta.get('heading', ""),
            "snippet": content[:120] + "..." if len(content) > 120 else content
        })

    return {"answer": clean_answer, "citations": citations, "model": model_to_use}


# ════════════════════════════════════════════════════════════════════════════
# TASK 3.3 — Tóm tắt tài liệu (Groq LLM)
# ════════════════════════════════════════════════════════════════════════════

def summarize_doc(
    doc_text: str,
    model_name: str = DEFAULT_LLM_MODEL
) -> AnswerDict:
    """Tóm tắt ngắn gọn nội dung cốt lõi dựa trên toàn bộ văn bản của tài liệu."""
    model_to_use = _resolve_model(model_name)
    if not doc_text.strip():
        return {"summary": "⚠️ Tài liệu không có nội dung văn bản để tóm tắt.", "model": model_to_use}

    client = _get_groq_sync_client()
    try:
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia phân tích tài liệu cao cấp. "
                        "Hãy đọc toàn bộ văn bản được cung cấp và viết một bản tóm tắt ngắn gọn, súc tích về nội dung cốt lõi của tài liệu.\n"
                        "YÊU CẦU:\n"
                        "- Bố cục rõ ràng, mạch lạc (sử dụng dấu gạch đầu dòng).\n"
                        "- Tập trung vào mục đích chính, các luận điểm cốt lõi hoặc định nghĩa quan trọng nhất.\n"
                        "- Độ dài khoảng 3-6 câu chất lượng cao bằng tiếng Việt."
                    )
                },
                {"role": "user", "content": f"VĂN BẢN TÀI LIỆU:\n{doc_text[:MAX_CONTEXT_CHARS]}"}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        ans = response.choices[0].message.content or ""
    except Exception as e:
        ans = f"⚠️ Lỗi tóm tắt từ Groq AI: {str(e)}"

    return {"summary": ans, "citations": [], "model": model_to_use}


# ════════════════════════════════════════════════════════════════════════════
# TASK 3.3 — Giải thích thuật ngữ (Groq LLM)
# ════════════════════════════════════════════════════════════════════════════

def explain_term(
    term: str,
    retrieved_chunks: List[RetrievedChunk],
    model_name: str = DEFAULT_LLM_MODEL
) -> AnswerDict:
    model_to_use = _resolve_model(model_name)
    context = _build_context(retrieved_chunks)

    prompt = f"""Hãy giải thích thuật ngữ: "{term}" BẮT BUỘC DỰA TRÊN ngữ cảnh dưới đây.

---
CONTEXT TÀI LIỆU:
{context}
---
YÊU CẦU:
- BẮT BUỘC DỰA TRÊN ĐÚNG TÀI LIỆU ĐÃ CUNG CẤP. Nêu định nghĩa chính xác và ví dụ nếu có trong tài liệu.
- Chèn nhãn trích dẫn [Nguồn N] tương ứng.
- Trả lời bằng tiếng Việt dễ hiểu và chuẩn xác.

GIẢI THÍCH:"""

    client = _get_groq_sync_client()
    try:
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý giải thích thuật ngữ chuyên sâu bám sát tài liệu học tập."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as e:
        raw_answer = f"⚠️ Lỗi tra cứu từ Groq AI: {str(e)}"

    citations = _extract_citations(retrieved_chunks, raw_answer)
    clean_answer = re.sub(r'\s*\[Nguồn \d+\]|\s*\[\d+\]', '', raw_answer)

    return {"answer": clean_answer, "citations": citations, "model": model_to_use}


# ════════════════════════════════════════════════════════════════════════════
# MỚI — Agentic Web Search Answer (Groq Web Knowledge Search)
# ════════════════════════════════════════════════════════════════════════════

async def generate_web_search_answer(
    question: str,
    doc_chunks: List[RetrievedChunk],
    model_name: str = DEFAULT_LLM_MODEL,
    num_web_results: int = 5
) -> dict:
    """
    1. Gọi Groq Web Knowledge Search lấy thông tin mở rộng.
    2. Kết hợp thông tin mở rộng + context tài liệu thành prompt.
    3. Groq sinh câu trả lời tổng hợp, ưu tiên tài liệu nội bộ.
    """
    model_to_use = _resolve_model(model_name)

    # ── Bước 1: Tìm kiếm qua Groq (Web Knowledge Search) ─────────────────
    web_results = await _groq_web_search(question, num_web_results)

    # ── Bước 2: Xây dựng context kết hợp ────────────────────────────────
    web_context = _build_web_context(web_results)
    doc_context = _build_context(doc_chunks)

    prompt = f"""Bạn là trợ lý hỏi đáp tài liệu học tập thông minh.
QUY TẮC BẮT BUỘC:
1. ĐÁP ÁN BẮT BUỘC DỰA VÀO CÁC TÀI LIỆU CÓ SẴN (được cung cấp ở phần NGỮ CẢNH TỪ TÀI LIỆU).
2. Toàn bộ định nghĩa, lý thuyết, giải thích cốt lõi phải bám sát tài liệu giáo trình và trích dẫn [Nguồn N].
3. Chỉ sử dụng thông tin từ Tri thức mở rộng (Groq) để bổ sung nếu tài liệu nội bộ chưa có hoặc chưa đầy đủ chi tiết.
4. Trình bày rõ ràng, mạch lạc bằng tiếng Việt, chính xác theo kiến thức giáo trình.

═══ NGỮ CẢNH TỪ TÀI LIỆU (ƯU TIÊN TUYỆT ĐỐI) ═══
{doc_context if doc_context else "(Không tìm thấy đoạn nào trong tài liệu liên quan)"}

═══ TRI THỨC MỞ RỘNG TỪ WEB (GROQ) ═══
{web_context if web_context else "(Không có kết quả web)"}

═══ CÂU HỎI ═══
{question}

TRẢ LỜI:"""

    # ── Bước 3: Gọi Groq Tổng Hợp ─────────────────────────────────────────
    client = _get_groq_async_client()
    try:
        response = await client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý học tập cao cấp, luôn ưu tiên tài liệu chính thống."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as e:
        raw_answer = f"⚠️ Lỗi Groq AI: {str(e)}"

    citations = _extract_citations(doc_chunks, raw_answer)
    clean_answer = re.sub(r'\s*\[Nguồn \d+\]|\s*\[\d+\]', '', raw_answer)

    return {
        "answer": clean_answer,
        "citations": citations,
        "web_results": web_results,
        "model": model_to_use
    }


async def _groq_web_search(query: str, num: int = 5) -> List[dict]:
    """Sử dụng Groq AI để tìm kiếm và cung cấp tri thức mở rộng bên ngoài."""
    if not GROQ_API_KEY:
        return []

    client = _get_groq_async_client()
    try:
        response = await client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là công cụ tìm kiếm và cung cấp tri thức mở rộng (Web Knowledge Engine). "
                        "Hãy tìm kiếm và cung cấp thông tin cập nhật, chính xác, súc tích về chủ đề của người dùng. "
                        "BẮT BUỘC trả về đúng định dạng JSON object: {\"results\": [{\"title\": \"...\", \"snippet\": \"...\", \"url\": \"...\"}]}"
                    )
                },
                {"role": "user", "content": f"Chủ đề tìm kiếm: {query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        items = parsed.get("results", [])
        return [
            {
                "title": item.get("title", "Tri thức mở rộng (Groq)"),
                "url": item.get("url", "https://groq.com"),
                "snippet": item.get("snippet", "")
            }
            for item in items[:num]
        ]
    except Exception as e:
        print(f"Lỗi Groq search: {e}")
        return []


def _build_web_context(web_results: List[dict]) -> str:
    if not web_results:
        return ""
    parts = []
    for i, r in enumerate(web_results, 1):
        parts.append(
            f"[Web {i}] {r.get('title','')}\n"
            f"URL: {r.get('url','')}\n"
            f"Tóm tắt: {r.get('snippet','')}\n"
        )
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# GENERATE EMBEDDINGS (Local ONNX all-MiniLM-L6-v2)
# ════════════════════════════════════════════════════════════════════════════

def get_embedding(texts: list, model_name: str = None) -> list:
    """
    Sinh vector nhúng sử dụng mô hình nhúng cục bộ all-MiniLM-L6-v2 (ChromaDB ONNX)
    chạy 100% offline, không giới hạn rate limit, không phụ thuộc API bên ngoài.
    """
    global _embed_function
    if _embed_function is None:
        _embed_function = embedding_functions.DefaultEmbeddingFunction()

    if not texts:
        return []

    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Clean text
        clean_batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]
        embeddings = _embed_function(clean_batch)
        all_embeddings.extend([[float(val) for val in emb] for emb in embeddings])

    return all_embeddings


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _build_context(
    retrieved_chunks: List[RetrievedChunk],
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    context_parts = []
    current_length = 0

    for idx, chunk in enumerate(retrieved_chunks):
        source_label = f"[Nguồn {idx + 1}]"
        meta         = chunk.get('metadata', {})
        file_name    = meta.get('file_name', 'Unknown')
        page_num     = meta.get('page', 1)
        content      = chunk.get('content', '')

        chunk_text  = f"--- Khối Ngữ Cảnh {source_label} ---\n"
        chunk_text += f"Nội dung: {content}\n"
        chunk_text += f"Nguồn file: {file_name}, Trang: {page_num}\n\n"

        if current_length + len(chunk_text) > max_chars:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    return "".join(context_parts)


def _build_cited_prompt(question: str, context: str) -> str:
    return f"""Bạn là một trợ lý hỏi đáp tài liệu học tập trung thực và chính xác.
Nhiệm vụ: Hãy trả lời câu hỏi BẮT BUỘC DỰA TRÊN ĐÚNG THÔNG TIN TỪ CÁC TÀI LIỆU ĐÃ CUNG CẤP TRONG PHẦN NGỮ CẢNH.

---
HƯỚNG DẪN QUAN TRỌNG:
1. ĐÁP ÁN BẮT BUỘC DỰA VÀO CÁC TÀI LIỆU CÓ SẴN TRONG NGỮ CẢNH. TUYỆT ĐỐI KHÔNG suy diễn hoặc bịa đặt thông tin bên ngoài tài liệu. Nếu tài liệu không chứa đủ thông tin để trả lời, hãy nói rõ: "Dựa trên các tài liệu đã cung cấp, không có đủ thông tin để trả lời câu hỏi này."
2. Chèn nhãn trích dẫn [Nguồn X] tương ứng ở cuối mỗi ý hoặc mỗi đoạn thông tin lấy từ tài liệu.
3. Trả lời bằng tiếng Việt rõ ràng, giải thích cặn kẽ và bám sát nguyên văn hoặc ý nghĩa trong tài liệu học tập.

NGỮ CẢNH TÀI LIỆU:
{context if context else "(Không tìm thấy tài liệu phù hợp trong kho lưu trữ)"}

CÂU HỎI:
{question}

TRẢ LỜI:"""


def _extract_citations(retrieved_chunks: List[RetrievedChunk], raw_answer: str) -> List[CitationDict]:
    citations    = []
    seen_indices = []

    for idx in range(len(retrieved_chunks)):
        source_number = idx + 1
        if f"[Nguồn {source_number}]" in raw_answer or f"[{source_number}]" in raw_answer:
            if idx not in seen_indices:
                seen_indices.append(idx)

    for i in seen_indices:
        chunk   = retrieved_chunks[i]
        content = chunk.get('content', '')
        snippet = content[:100] + "..." if len(content) > 100 else content
        meta    = chunk.get('metadata', {})
        citations.append({
            "file_name": meta.get('file_name', 'Unknown'),
            "page":      meta.get('page', 1),
            "heading":   meta.get('heading', ""),
            "snippet":   snippet
        })

    return citations
