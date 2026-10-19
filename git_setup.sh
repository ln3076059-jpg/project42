#!/bin/bash
# =============================================================================
# SCRIPT: git_setup.sh
# MỤC ĐÍCH: Khởi tạo Git repo và tạo branch cho từng thành viên
#
# CÁCH DÙNG:
#   1. Vào thư mục gốc dự án
#   2. Chạy: bash git_setup.sh
# =============================================================================

set -e  # Dừng ngay nếu có lỗi

echo "========================================"
echo "  PDF RAG Chatbot — Git Branch Setup"
echo "========================================"

# --- Bước 1: Init repo (bỏ qua nếu đã có) ---
if [ ! -d ".git" ]; then
    echo "[1/5] Khởi tạo Git repository..."
    git init
    git add .
    git commit -m "chore: initial project skeleton with API contracts"
else
    echo "[1/5] Git repo đã tồn tại, bỏ qua init."
fi

# --- Bước 2: Tạo branch develop ---
echo "[2/5] Tạo branch develop..."
git checkout -b develop 2>/dev/null || git checkout develop
echo "  ✓ develop"

# --- Bước 3: Tạo branch cho từng thành viên từ develop ---
echo "[3/5] Tạo branch cho từng thành viên..."

# Thùy — Data Engineer
git checkout -b feature/thuy-data-parser develop 2>/dev/null || echo "  ! branch feature/thuy-data-parser đã tồn tại"
echo "  ✓ feature/thuy-data-parser"

# Phi — Pipeline Engineer
git checkout -b feature/phi-pipeline-retriever develop 2>/dev/null || echo "  ! branch feature/phi-pipeline-retriever đã tồn tại"
echo "  ✓ feature/phi-pipeline-retriever"

# Tiến — Model Engineer
git checkout -b feature/tien-model-generator develop 2>/dev/null || echo "  ! branch feature/tien-model-generator đã tồn tại"
echo "  ✓ feature/tien-model-generator"

# Mạnh — Backend + UI
git checkout -b feature/manh-api-ui develop 2>/dev/null || echo "  ! branch feature/manh-api-ui đã tồn tại"
echo "  ✓ feature/manh-api-ui"

# --- Bước 4: Quay về develop ---
echo "[4/5] Quay về branch develop..."
git checkout develop

# --- Bước 5: Tóm tắt ---
echo ""
echo "[5/5] Hoàn thành! Cấu trúc branch:"
git branch -a
echo ""
echo "========================================"
echo "  HƯỚNG DẪN LÀM VIỆC CHO TỪNG NGƯỜI"
echo "========================================"
echo ""
echo ""
echo "  Khi xong 1 task, tạo Pull Request vào develop"
echo "========================================"
