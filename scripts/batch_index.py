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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parser import process_pdf
from src.pipeline.retriever import add_chunks_to_workspace, list_files_in_workspace, get_or_create_collection

WORKSPACE_NAME = "GiaoTrinh"
DOCS_DIR = PROJECT_ROOT / "tai_lieu"

def main():
    print(f"=== BẮT ĐẦU XỬ LÝ VÀ INDEX CÁC FILE PDF VÀO WORKSPACE '{WORKSPACE_NAME}' ===")
    get_or_create_collection(WORKSPACE_NAME)
    already_indexed = set(list_files_in_workspace(WORKSPACE_NAME))
    print(f"Các file đã có trong workspace: {list(already_indexed)}")

    pdf_files = sorted([f for f in DOCS_DIR.glob("*.pdf")])
    print(f"Tổng cộng tìm thấy {len(pdf_files)} file PDF trong thư mục {DOCS_DIR.name}/")

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = pdf_path.name
        print(f"\n[{i}/{len(pdf_files)}] File: {filename}")

        if filename in already_indexed:
            print(f"-> Đã được index trước đó. Bỏ qua.")
            continue

        print(f"-> Đang trích xuất (parse) và chia nhỏ (chunk)...")
        start_t = time.time()
        try:
            chunks = process_pdf(str(pdf_path), WORKSPACE_NAME)
            if not chunks:
                print(f"-> Cảnh báo: Không thể trích xuất nội dung từ {filename}.")
                continue

            for c in chunks:
                if "metadata" in c:
                    c["metadata"]["file_name"] = filename

            print(f"-> Đã tạo {len(chunks)} chunks. Đang tạo embedding và lưu vào ChromaDB...")
            add_chunks_to_workspace(WORKSPACE_NAME, chunks)
            elapsed = time.time() - start_t
            print(f"-> THÀNH CÔNG! Đã index {len(chunks)} chunks trong {elapsed:.1f}s.")
        except Exception as e:
            print(f"-> LỖI khi xử lý {filename}: {e}")

        # Nghỉ ngắn giữa các file để giữ rate limit an toàn
        time.sleep(3)

    final_files = list_files_in_workspace(WORKSPACE_NAME)
    print(f"\n=== HOÀN TẤT! Tổng số file trong workspace '{WORKSPACE_NAME}': {len(final_files)} ===")
    for f in final_files:
        print(f" - {f}")

if __name__ == "__main__":
    main()
