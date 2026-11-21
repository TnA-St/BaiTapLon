# Student Performance – giao diện trắng đen (4 tab)

Ứng dụng Streamlit dành cho dữ liệu **UCI Student Performance** (`student-por.csv`), giữ nguyên 9 mô hình Machine Learning và quy trình huấn luyện.

## Chạy demo trên Windows / macOS / Linux

Trong thư mục `ML_Project_UCI`:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Trên Windows có thể dùng `py -3.14 -m streamlit run app.py` nếu bạn cài Python 3.14. Trên macOS/Linux có thể dùng `python3 -m streamlit run app.py`. Hoặc dùng script `START_WINDOWS.bat` / `bash START_MAC_LINUX.sh` như trước.

Mở **http://localhost:8501**. Không đóng Terminal khi dùng demo.

## Chỉ còn 4 tab

1. **Dữ liệu & huấn luyện:** dữ liệu UCI có sẵn, bảng tổng quan, nút train và cấu hình MLP.
2. **Dự đoán:** nhập 9 đặc trưng để dự đoán G3 và nhãn Đạt / Không đạt.
3. **So sánh mô hình:** bảng chỉ số phân loại/hồi quy, ma trận nhầm lẫn và biểu đồ tròn đúng/sai.
4. **Biểu đồ:** biểu đồ tròn Đạt / Không đạt, thời gian học, K-Means, tỷ lệ Mini-batch, và hai đường Loss / Validation Accuracy vẽ từ CSV MLP sau huấn luyện.

Không có sidebar, hình hoạt hình hoặc bảng màu pastel. Giao diện chỉ dùng **trắng, đen và các sắc độ xám**, bảng dữ liệu và biểu đồ tròn.

## Dữ liệu UCI

Dữ liệu UCI `student-por.csv` **đã được đính kèm trong ZIP** tại `data/student-por.csv` (649 bản ghi; 33 cột). **Không cần tải từ web**.

Trong tab 1, nhấn **Huấn luyện / Huấn luyện lại** để tạo `models/` và `reports/` từ CSV có sẵn; hoặc chạy `python train_models.py` trong Terminal. Giao diện không có phần tải/thay CSV. Link gốc UCI được hiển thị tại tab 1.

Bản ZIP đã kèm dataset. Kết quả hiển thị sau khi huấn luyện lấy từ dữ liệu CSV bạn nhập, không tự tạo số liệu giả lập. Dữ liệu gốc là học sinh trung học Bồ Đào Nha, không phải sinh viên đại học Việt Nam. Dự đoán dùng G1, G2 nên phù hợp vào giai đoạn đã có điểm giai đoạn, không phải ngay đầu kỳ.

## Biểu đồ MLP thực tế

Sau khi chạy huấn luyện, `train_models.py` tạo `reports/mlp_training_history.csv` chứa các cột `Epoch`, `Train_loss` và `Val_accuracy`. Tab **Biểu đồ** đọc trực tiếp file này để vẽ **Loss** và **Validation Accuracy** theo vòng học, không sử dụng ảnh mô phỏng. Nếu chưa huấn luyện, ứng dụng hiển thị thông báo thay cho biểu đồ.

Nút bấm sử dụng nền đen, chữ trắng trong mọi trạng thái (kể cả nút trong form, nút tải file).
