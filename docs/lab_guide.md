# Hướng dẫn lab

## Nhiệm vụ 1: Data loader
Triển khai quá trình nạp JSONL một cách chắc chắn. Thêm thông báo lỗi có số dòng và kiểm tra bản ghi trùng lặp.

## Nhiệm vụ 1.5: (Tùy chọn) Sinh dữ liệu tổng hợp
Dùng LLM để mở rộng dataset. Phần này giúp bạn học cách tạo dữ liệu căn chỉnh chất lượng cao ở quy mô lớn.
```powershell
$env:OPENAI_API_KEY="your_key"
python scripts/generate_data.py --count 10 --domain "python coding"
```

## Nhiệm vụ 2: Hàm loss
Chọn DPO hoặc ORPO. Triển khai phần TODO trong `src/preference_lab/losses.py`.

## Nhiệm vụ 3: Đánh giá
Thay điểm giả lập bằng điểm lấy từ mô hình hoặc một bộ chấm điểm tất định cho chế độ CPU.

## Nhiệm vụ 4: Báo cáo
Viết báo cáo ngắn gồm ghi chú về dataset, config, metric và các lỗi thường gặp.
