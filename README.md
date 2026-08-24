# Lab Preference Alignment: Bộ khởi đầu DPO \& ORPO

Đây là bài lab 2 giờ theo phong cách production về căn chỉnh theo preference. Repository đã hoàn thiện các phần chính: nạp dữ liệu, validation, loss DPO/ORPO, đánh giá, test và báo cáo.

## Mục tiêu học tập

- Kiểm tra hợp lệ và nạp các cặp preference (`prompt`, `chosen`, `rejected`).
- Triển khai hoặc bọc logic huấn luyện DPO/ORPO.
- Xây dựng metric đánh giá cho preference theo cặp và prompt kiểm thử hồi quy.
- Luyện thói quen production: code có type, config, test, Makefile, CI, tài liệu.

## Chạy nhanh trên Windows

Mở PowerShell tại thư mục project rồi chạy:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

Nếu PowerShell chặn script kích hoạt môi trường ảo, chạy lệnh sau trong phiên PowerShell hiện tại rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Hoặc dùng Command Prompt:

```cmd
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -e ".[dev]"
python -m pytest -q
```

Dependency huấn luyện tùy chọn:

```powershell
python -m pip install -e ".[dev,train]"
```

Nếu bạn đã cài `make` trên Windows, vẫn có thể chạy các lệnh trong Makefile. Nếu chưa cài, hãy dùng trực tiếp các lệnh `python -m pytest`, `python -m ruff`, `python -m mypy` như bảng bên dưới.

## Kiểm tra trước khi nộp trên Windows

```powershell
python -m ruff check src tests
python -m mypy src
python -m pytest -q
.\scripts\smoke_test.ps1
git status --short
git diff --stat
```

`outputs/` được ignore, nên không dùng `git add -f outputs/`. Nội dung metric cần nộp đã được ghi lại trong `docs/REPORT_TEMPLATE.md`.

## Quy tắc lab

1. Không viết lại toàn bộ repository.
2. Giữ thay đổi tập trung vào các module của lab, không refactor lan rộng khi không cần thiết.
3. Giữ test luôn pass sau mỗi mốc.
4. Không commit secret, trọng số mô hình hoặc dataset riêng tư.

## Các mốc thực hiện

| Thời gian | Mục tiêu | Lệnh |
|---|---|---|
| 0-30 phút | Thiết lập môi trường và xem dữ liệu mẫu | `python -m pytest -q` |
| 30-50 phút | Triển khai validation/collator cho dataset | `python -m pytest tests/test_data.py` |
| 50-70 phút | (Tùy chọn) Sinh dữ liệu tổng hợp | `python scripts/generate_data.py` |
| 70-100 phút | Triển khai TODO cho DPO hoặc ORPO | `python -m pytest tests/test_losses.py` |
| 100-115 phút | Triển khai đánh giá và báo cáo | `pref-lab evaluate --config configs/local.yaml` |
| 115-120 phút | Demo một phút | `Get-Content outputs\metrics.json` |

## Cấu trúc repository

```text
src/preference_lab/     Gói Python
data/                   Dataset preference mẫu cỡ nhỏ
configs/                Config YAML cho thí nghiệm local
docs/                   Hướng dẫn lab, rubric, template thẻ dữ liệu
scripts/                Entrypoint tiện ích
tests/                  Unit test cho phần sinh viên triển khai
```

## Checklist production

- [x] Schema của dataset đã được kiểm tra hợp lệ.
- [x] Chia train/eval theo prompt, không chia theo từng dòng.
- [x] Config đã được commit; artifact sinh ra đã được ignore.
- [x] Metric được lưu dưới dạng JSON.
- [x] Prompt kiểm thử hồi quy về safety đã có trong tài liệu để chạy trước/sau huấn luyện.
- [x] Thẻ dữ liệu đã được cập nhật.
