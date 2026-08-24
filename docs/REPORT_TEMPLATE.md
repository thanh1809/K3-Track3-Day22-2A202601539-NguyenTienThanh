# Báo cáo thí nghiệm căn chỉnh theo preference

*Báo cáo đã được điền sau khi hoàn thành các mốc chính của lab trên dataset mẫu local.*

## 1. Phân tích và làm sạch dataset

### Tóm tắt quá trình nạp dữ liệu
- **Tổng số example đã nạp**: `24`
- **Vấn đề validation phát hiện được**: Dòng 1 của `data/sample_preferences.jsonl` bị sai JSON vì cụm `"self-attention"` chưa được escape trong chuỗi `prompt`.
- **Các bước làm sạch đã thực hiện**: Đã sửa dòng 1 thành `\"self-attention\"`; triển khai loader JSONL có thông báo lỗi kèm số dòng, validation schema bằng Pydantic, kiểm tra prompt trùng lặp theo dạng đã chuẩn hóa và tùy chọn chặn PII cơ bản.

### Chiến lược chia dữ liệu
- **Tỷ lệ train/validation**: Khoảng `80/20`, cụ thể `19` example train và `5` example validation với `validation_ratio=0.2`, `seed=42`.
- **Ngăn rò rỉ dữ liệu**: Split được thực hiện theo prompt đã chuẩn hóa thay vì theo từng dòng. Các example có cùng prompt luôn nằm trong cùng một split, giúp tránh việc cùng một câu hỏi xuất hiện ở cả train và validation.

## 2. Triển khai: DPO và ORPO

### Lựa chọn hàm mục tiêu
- **Vì sao chọn phương pháp này?**: DPO được dùng làm phương pháp chính vì config local đặt `training.method: dpo` và DPO phù hợp với dữ liệu preference dạng `chosen/rejected` mà không cần reward model riêng. ORPO cũng được triển khai để hoàn thiện TODO và có thể chuyển đổi khi cần.
- **Hyperparameter chính**:
    - `beta`: `0.1`
    - `lambda_orpo` (nếu áp dụng): `0.1`

### Ổn định số học
- **Thách thức**: DPO/ORPO làm việc trực tiếp với log probability nên dễ gặp underflow/overflow nếu tính sigmoid hoặc `log(1 - exp(logp))` một cách ngây thơ.
- **Giải pháp**: DPO dùng `np.logaddexp(0, -logits)` để tính `-log(sigmoid(logits))` ổn định. ORPO dùng nhánh ổn định cho `log1mexp`, clamp log probability sát `0`, kiểm tra shape và kiểm tra giá trị hữu hạn trước khi tính loss.

## 3. Kết quả đánh giá

### Chỉ số
| Chỉ số | Giá trị |
|---|---|
| Độ chính xác pairwise | `95.83%` |
| Mean score margin | `0.2215` |
| DPO loss sanity check | `0.6636` |
| ORPO loss sanity check | `1.0171` |

### Đánh giá định tính
- **Prompt**: `Explain the concept of "self-attention" in Transformers.`
- **Câu trả lời được chọn**: `Self-attention allows the model to weigh the importance of different words in the input sequence when processing each word, capturing long-range dependencies.`
- **Câu trả lời bị loại**: `Self-attention is a simpler version of RNNs that uses less memory and is faster to train.`
- **Preference của mô hình**: Đúng. Bộ chấm điểm tất định cho CPU cho điểm câu `chosen` cao hơn câu `rejected` nhờ độ đặc hiệu, độ dài hợp lý và mức khớp thuật ngữ tốt hơn.

## 4. Thảo luận và lỗi thường gặp

- **Điều gì hoạt động tốt?**: Loader phát hiện lỗi JSON có số dòng rõ ràng, dữ liệu mẫu nạp đủ 24 example, split không rò rỉ prompt, loss DPO/ORPO có test sanity check và CLI evaluate ghi metric JSON vào `outputs/metrics.json`.
- **Bias quan sát được**: Bộ chấm điểm CPU hiện tại là heuristic, nên có xu hướng ưu tiên câu trả lời dài hơn, nhiều thuật ngữ hơn và có nhiều từ trùng với prompt. Đây là proxy để chạy local, không thay thế được logprob từ mô hình thật.
- **Safety**: `docs/regression_prompts.md` đã liệt kê 4 prompt kiểm thử hồi quy về lời khuyên y tế rủi ro cao, tóm tắt có giới hạn từ, thừa nhận không chắc chắn và troubleshooting thiếu ngữ cảnh. Trong bản CPU local, các prompt này được giữ làm bộ regression trước/sau huấn luyện; chưa gọi mô hình thật nên chưa đánh giá nội dung sinh ra.
