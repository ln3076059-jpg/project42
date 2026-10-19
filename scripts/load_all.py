import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from src.data.parser import process_pdf
from src.pipeline.retriever import add_chunks_to_workspace, list_files_in_workspace, get_or_create_collection, delete_workspace

WORKSPACE_NAME = "GiaoTrinh"
DOCS_DIR = PROJECT_ROOT / "tai_lieu"

def main():
    print(f"=== ĐỒNG BỘ TOÀN BỘ GIÁO TRÌNH VÀO WORKSPACE '{WORKSPACE_NAME}' ===", flush=True)
    
    # Kiểm tra dimension của collection hiện tại
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        col = client.get_collection(WORKSPACE_NAME)
        peek = col.peek(1)
        if peek['embeddings'] is not None and len(peek['embeddings']) > 0:
            dim = len(peek['embeddings'][0])
            if dim != 384:
                print(f"[Reset] Phát hiện collection cũ có dimension={dim} (Gemini). Đang đặt lại collection sang dimension=384 (Local ONNX)...", flush=True)
                delete_workspace(WORKSPACE_NAME)
    except Exception:
        pass

    get_or_create_collection(WORKSPACE_NAME)
    already_indexed = set(list_files_in_workspace(WORKSPACE_NAME))
    print(f"Các file đã có: {list(already_indexed)}", flush=True)

    pdf_files = sorted([f for f in DOCS_DIR.glob("*.pdf")])
    remaining = [f for f in pdf_files if f.name not in already_indexed]
    print(f"Tổng số file PDF tìm thấy: {len(pdf_files)}")
    print(f"Số file cần xử lý thêm: {len(remaining)}", flush=True)

    total_chunks = 0
    t_start = time.time()

    for i, pdf_path in enumerate(remaining, 1):
        filename = pdf_path.name
        print(f"\n[{i}/{len(remaining)}] Bắt đầu xử lý: {filename}...", flush=True)
        t0 = time.time()
        
        chunks = process_pdf(str(pdf_path), WORKSPACE_NAME)
        for c in chunks:
            if "metadata" in c:
                c["metadata"]["file_name"] = filename

        parse_time = time.time() - t0
        print(f"-> Đã trích xuất {len(chunks)} chunks trong {parse_time:.1f}s. Đang embedding cục bộ và lưu ChromaDB...", flush=True)
        
        t1 = time.time()
        add_chunks_to_workspace(WORKSPACE_NAME, chunks)
        embed_time = time.time() - t1
        total_chunks += len(chunks)
        print(f"-> HOÀN TẤT file {filename} trong {embed_time:.1f}s!", flush=True)

    final = list_files_in_workspace(WORKSPACE_NAME)
    print(f"\n=== ĐÃ NẠP XONG TẤT CẢ GIÁO TRÌNH ({len(final)}/{len(pdf_files)} file) ===", flush=True)
    for f in final:
        print(f" - {f}", flush=True)
    print(f"Tổng chunks: {total_chunks} | Tổng thời gian: {time.time() - t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
