import os
import re
import pymupdf 
import pdfplumber

from typing import List, Dict, Any

ChunkDict = Dict[str, Any]


# =============================================================================
# TASK 1.1 + 1.2 + 1.3 — HÀM CHÍNH (Pipeline hoàn chỉnh)
# =============================================================================

def process_pdf(file_path: str, workspace_name: str) -> List[ChunkDict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'không thấy file: {file_path}')
    if not workspace_name:
        raise ValueError("workspace_name empty")
    text_chunk = _extract_text_chunks(file_path, workspace_name)
    table_chunk = _extract_table_chunks(file_path, workspace_name)

    sum_chunk = text_chunk + table_chunk
    sum_chunk.sort(key= lambda x: x["metadata"]["page"])  # xếp theo trang

    return sum_chunk


# =============================================================================
# TASK 1.1 — HELPER: Extract text với PyMuPDF
# =============================================================================

def _extract_text_chunks(file_path: str, workspace_name: str) -> List[ChunkDict]:
    """
    (Private) Dùng PyMuPDF (fitz) đọc text, nhận diện block Heading/Paragraph,
    rồi cắt thành chunk có nghĩa và gắn metadata.

    Args:
        file_path      (str): Đường dẫn file PDF
        workspace_name (str): Tên workspace

    Returns:
        List[ChunkDict]: Các chunk loại "text" và "heading"
    """
    chunks = []
    filename = os.path.basename(file_path)

    with pymupdf.open(file_path) as document:
        cur_heading = ''
        cur_chunk_text = ''
        page_start = 1

        for page_index in range(len(document)):
            page = document[page_index]

            blocks = page.get_text('dict').get('blocks', [])

            # lấy font size
            total_size = 0
            total_spans = 0
            for block in blocks:
                if block.get("type") == 0: # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get('text', " ").strip():
                                total_size += span.get('size', 0)
                                total_spans +=1
            font_size_avg = (total_size/total_spans) if total_spans > 0 else 12.0

            # phân loại block
            for block in blocks:
                block_text = ""
                max_font_block = 0.0

                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        block_text += span.get('text', "")
                        if span.get('size', 0) > max_font_block:
                            max_font_block = span.get('size', 0)
                
                block_text = block_text.strip()

                if not block_text: continue

                # heading
                if _is_heading(block_text, max_font_block, font_size_avg):
                    # đẩy chunk cũ list chunk
                    if cur_chunk_text:
                        chunks.append({
                            'content': cur_chunk_text.strip(),
                            'metadata':{
                                "file_name": filename,
                                "page": page_start,
                                "workspace_name": workspace_name,
                                "chunk_type": "text",
                                "heading": cur_heading
                            }
                        })

                        # update current chunk text
                        cur_chunk_text = ''
                    cur_heading = block_text
                    page_start = page_index + 1
                # paragraph
                else:
                    if not cur_chunk_text:  page_start = page_index + 1
                    cur_chunk_text += block_text + '\n'

                    # limit chunk size 1000
                    if len(cur_chunk_text) > 5000:
                        chunks.append({
                            'content': cur_chunk_text.strip(),
                            'metadata':{
                                "file_name": filename,
                                "page": page_start,
                                "workspace_name": workspace_name,
                                "chunk_type": "text",
                                "heading": cur_heading
                            }
                        })
                        cur_chunk_text = ""
        # last chunk
        if cur_chunk_text.strip():
            chunks.append({
                'content': cur_chunk_text.strip(),
                'metadata':{
                    "file_name": filename,
                    "page": page_start,
                    "workspace_name": workspace_name,
                    "chunk_type": "text",
                    "heading": cur_heading
                }
            })
    return chunks



# =============================================================================
# TASK 1.1 — HELPER: Extract bảng biểu với pdfplumber
# =============================================================================

def _extract_table_chunks(file_path: str, workspace_name: str) -> List[ChunkDict]:
    chunks = []
    filename = os.path.basename(file_path)

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                if not (page.curves or page.lines or page.rects):
                    continue
                tables = page.extract_tables()
                for table in tables:
                    if not table: continue
                
                    markdow_table = ""
                    for i, row in enumerate(table):
                        row_cleaning = [str(cell).replace("\n", "") if cell is not None else "" for cell in row] 
                        markdow_table += "| " + " | ".join(row_cleaning) + " |\n"

                        if i ==0:
                            markdow_table += "| " + " | ".join(["---"] * len(row_cleaning)) + " |\n"
                            
                    if markdow_table.strip():
                        chunks.append({
                            "content": markdow_table.strip(),
                            "metadata": {
                                "file_name": filename,
                                "page": page_idx,
                                "workspace_name": workspace_name,
                                "chunk_type": "text",
                                "heading": "bảng"
                            }
                        })
    except Exception as e:
        pass
    return chunks

# =============================================================================
# TASK 1.2 — HELPER: Nhận diện Heading
# =============================================================================

def _is_heading(text: str, font_size: float, avg_font_size: float) -> bool:
    text = text.strip()
    if not text:  return False

    if font_size >= avg_font_size * 1.25: return True 
    
    pattern = r"(?i)^(" \
              r"chương\s+\d+|phần\s+[IVXLCDM]+|bài\s+\d+|mục\s+\d+|" \
              r"\d+(\.\d+)+|" \
              r"\d+[\.\)]|" \
              r"[IVX]+[\.\)]|" \
              r"[A-Z][\.\)]|" \
              r"(mục lục|lời mở đầu|mở đầu|kết luận|tài liệu tham khảo|phụ lục)" \
              r")\b"
    if re.match(pattern, text): return True

    if text.isupper():  return True # tiêu đề thừn in hoa

    return False


# =============================================================================
# BACKWARD COMPATIBILITY — Giữ lại hàm cũ để không break code hiện tại
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    [DEPRECATED] Hàm cũ từ pdf_processor.py — giữ lại để không break code.
    Nên dùng process_pdf() thay thế.
    """
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list:
    """
    [DEPRECATED] Hàm cắt chunk cũ theo ký tự — giữ lại để không break code.
    Nên dùng process_pdf() thay thế.
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
