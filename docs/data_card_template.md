# Thẻ dữ liệu

- Tên dataset: `sample_preferences.jsonl`
- Nguồn: Dataset mẫu đi kèm repository, gồm các câu hỏi giáo dục về machine learning/deep learning.
- Giấy phép/quyền sử dụng: Dữ liệu lab nội bộ trong repository; không chứa dữ liệu riêng tư hoặc trọng số mô hình.
- Schema: Mỗi dòng là một JSON object có `prompt`, `chosen`, `rejected`, `metadata`. `metadata` hiện có `domain` và `rubric`.
- Rubric gán nhãn: `chosen` là câu trả lời chính xác, hữu ích hơn; `rejected` là câu trả lời hợp lý bề mặt nhưng có lỗi factual, thiếu chính xác hoặc diễn giải kém.
- Bias đã biết: Chủ yếu thuộc domain giáo dục ML, tiếng Anh, rubric thiên về độ chính xác kỹ thuật. Dataset nhỏ nên chưa đại diện cho nhiều miền, ngôn ngữ hoặc phong cách người dùng.
- Kiểm tra safety/PII: Loader có tùy chọn `reject_pii=True` để phát hiện email/số điện thoại cơ bản. Dataset mẫu không có PII rõ ràng.
- Phương pháp chia train/validation/test: Chia train/validation theo prompt đã chuẩn hóa để tránh rò rỉ; với config mặc định `validation_ratio=0.2`, `seed=42`, dataset 24 example được chia thành 19 train và 5 validation. Chưa tạo test split riêng.
