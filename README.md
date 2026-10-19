# 📚 PDF RAG Assistant v3.0

> Hệ thống hỏi đáp và phân tích tài liệu học tập thông minh (RAG) sử dụng **Groq LLM** siêu tốc và mô hình nhúng cục bộ **ChromaDB Local ONNX**.

🔗 **GitHub Repository:** [https://github.com/ln3076059-jpg/project42](https://github.com/ln3076059-jpg/project42)

---

## 🌟 Tính Năng Nổi Bật

- **⚡ Siêu tốc với Groq LLM (`openai/gpt-oss-120b`)**: Tốc độ sinh câu trả lời tức thì hàng trăm tokens/giây, không bị nghẽn hay giới hạn như các gói Free Tier thông thường.
- **🛡️ 100% Cục bộ cho Vector Embedding (`all-MiniLM-L6-v2`)**: Chạy trực tiếp qua ChromaDB ONNX Runtime, hoàn toàn offline, **không tốn quota, không giới hạn request**.
- **🎯 Bám sát tài liệu & Trích dẫn chính xác**: Câu trả lời được kiểm soát chặt chẽ theo ngữ cảnh tài liệu, tự động đính kèm số trang, tiêu đề mục và trích đoạn nguồn `[Nguồn X]`.
- **🔍 Hybrid Search (BM25 + Dense Semantic Vector)**: Kết hợp thuật toán so khớp từ khóa BM25 và tìm kiếm ngữ nghĩa vector qua Reciprocal Rank Fusion (RRF) để tối ưu độ chính xác tiếng Việt.
- **🌐 Web Knowledge Engine**: Bổ sung tri thức mở rộng khi người dùng cần tra cứu thêm kiến thức ngoài giáo trình.
- **📝 Tóm tắt & Giải thích thuật ngữ**: Hỗ trợ tóm tắt toàn bộ tài liệu và giải thích các thuật ngữ cốt lõi theo cấu trúc sư phạm 4 phần.
- **🎨 Giao diện Dark Mode cao cấp**: Tích hợp bộ phân tích cú pháp Markdown (`Marked.js`), đọc to văn bản (TTS), sao chép câu trả lời và quản lý nhiều Workspace độc lập.
- **🔑 Đăng nhập Google OAuth2**: Quản lý phiên làm việc và lịch sử trò chuyện theo từng tài khoản.

---

## 🛠️ Công Nghệ Sử Dụng

| Thành phần | Công nghệ | Mô tả |
|---|---|---|
| **Backend API** | FastAPI / Uvicorn | Xử lý các endpoint RESTful, streaming và xác thực OAuth2 |
| **LLM Generation** | Groq API (`openai/gpt-oss-120b`) | Trả lời RAG, tóm tắt, giải thích thuật ngữ và tra cứu web |
| **Vector Database** | ChromaDB (`PersistentClient`) | Lưu trữ và lập chỉ mục vector nhúng của các khối văn bản |
| **Embedding Model** | `all-MiniLM-L6-v2` (ONNX) | 384 dimensions, xử lý nhúng cục bộ trên CPU |
| **PDF Parser** | PyMuPDF (fitz) + pdfplumber | Trích xuất văn bản, cấu trúc bảng biểu và phân đoạn thông minh |
| **Search Engine** | BM25 + Vector Hybrid Search | Rerank kết quả tìm kiếm với độ chính xác cao |
| **Frontend** | Vanilla JS + HTML5 + CSS3 | Giao diện Dark theme, Marked.js, Web Speech API (TTS) |

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Clone mã nguồn
```bash
git clone https://github.com/ln3076059-jpg/project42.git
cd project42
```

### 2. Thiết lập môi trường ảo Python
```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate
```

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
pip install groq chromadb
```

### 4. Cấu hình file `.env`
Tạo file `.env` tại thư mục gốc với các thông số sau:
```env
# Groq API Key
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Google OAuth2 (Tùy chọn cho tính năng đăng nhập)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
FRONTEND_URL=http://localhost:8000

# Session Secret
SESSION_SECRET=rag-assistant-secret-key-for-session-32chars
```

### 5. Nạp tài liệu giáo trình vào ChromaDB
Đặt các file PDF giáo trình vào thư mục `tai_lieu/`, sau đó chạy:
```bash
python scripts/load_all.py
```

### 6. Khởi động Web Server
```bash
uvicorn src.api.api_server:app --port 8000 --host 127.0.0.1
```
Mở trình duyệt và truy cập: **`http://localhost:8000`**

---

## 📁 Cấu Trúc Dự Án

```
project42/
├── frontend/
│   └── index.html             # Giao diện người dùng đơn trang
├── src/
│   ├── api/
│   │   └── api_server.py      # FastAPI endpoints & OAuth routing
│   ├── data/
│   │   └── parser.py          # Trích xuất và chunking tài liệu PDF
│   ├── models/
│   │   └── llm_generator.py   # Tích hợp Groq API và local ONNX embeddings
│   └── pipeline/
│       └── retriever.py       # Quản lý ChromaDB & Hybrid Search (BM25 + Vector)
├── scripts/
│   └── load_all.py            # Đồng bộ toàn bộ tài liệu PDF vào vector DB
├── tai_lieu/                  # Thư mục chứa các tệp PDF giáo trình
├── chroma_db/                 # Cơ sở dữ liệu vector ChromaDB cục bộ
├── requirements.txt           # Danh sách thư viện phụ thuộc
└── README.md                  # Hướng dẫn dự án
```

---

## 📄 Bản Quyền
Dự án được phát triển phục vụ mục đích nghiên cứu và học tập.
Mã nguồn: [ln3076059-jpg/project42](https://github.com/ln3076059-jpg/project42)
