# Dự án Energy-based Out-of-distribution (OOD) Detection

Dự án này là mã nguồn thử nghiệm và báo cáo phân tích cho phương pháp phát hiện dữ liệu Out-of-distribution (OOD) dựa trên Energy Score (tham khảo từ bài báo của Liu et al., 2020), so sánh với phương pháp Baseline sử dụng Softmax Confidence Score (MSP).

Phương pháp này khai thác độ lớn của vector logit đầu ra (Energy Score) để khắc phục hiện tượng overconfidence của Softmax, giúp cải thiện độ nhạy khi phát hiện mẫu lạ (OOD) so với mẫu trong phân phối (In-distribution - ID).

## Cấu trúc thư mục và mô tả các tệp tin

- **Thư mục dữ liệu và kết quả**:
  - `data/`: Thư mục chứa các tập dữ liệu được tải về tự động (CIFAR-10, SVHN, LSUN, Tiny-ImageNet).
  - `results/`: Thư mục lưu trữ kết quả phân tích và các hình ảnh biểu đồ được sinh ra.
    - `results/charts/`: Lưu các biểu đồ từ kịch bản `export_charts.py`.
    - `results/charts_combined/`: Lưu các biểu đồ từ kịch bản `export_combined_figures.py`.
    - `results/individual/`: Lưu các hình ảnh minh họa từ kịch bản `export_individual_figures.py`.

- **Mã nguồn thực thi và phân tích**:
  - `run_ood_analysis.py`: Kịch bản chính dùng để tải dữ liệu, đánh giá mô hình bằng MSP và Energy Score.
  - `eval_ood_all.py`: Kịch bản đánh giá mở rộng trên nhiều tập dữ liệu OOD khác nhau.
  - `evaluate_temperature.py`: Chạy khảo sát và so sánh hiệu suất với các mức giá trị Temperature khác nhau (T từ 1 đến 10).
  - `deep_analysis.py`: Mã nguồn thực hiện phân tích chuyên sâu các chỉ số đánh giá.

- **Các kịch bản xuất biểu đồ (Hình ảnh kết quả)**:
  - `export_charts.py`: Xuất các biểu đồ phân tích cơ bản (ROC, Precision-Recall, phân phối điểm số) khi so sánh In-Distribution (CIFAR-10) và OOD (SVHN). Hình ảnh lưu tại `results/charts/`.
  - `export_combined_figures.py`: Chạy đánh giá và xuất các biểu đồ so sánh kết hợp (Combined) trên nhiều tập dữ liệu OOD (SVHN, LSUN, Tiny-ImageNet) với các ngưỡng Temperature (T) khác nhau. Hình ảnh lưu tại `results/charts_combined/`.
  - `export_individual_figures.py`: Trích xuất từng figure (A, B, C, D...) và các hình ảnh mẫu minh họa dự đoán đúng/sai cho In-Distribution và OOD. Hình ảnh lưu tại `results/individual/`.
  - `section5b_sample_visualization.py`: Kịch bản minh họa mẫu dữ liệu cụ thể cho mục đích báo cáo.

- **Tài liệu và tệp cấu hình**:
  - `wrn28_10_cifar10.pth`: Trọng số mô hình WideResNet-28-10 đã được huấn luyện sẵn trên CIFAR-10 (Cần có để chạy các kịch bản).
  - `main.tex`, `main.pdf`, v.v...: Tệp mã nguồn LaTeX và bản dịch PDF của báo cáo khoa học.
  - `TrustWorthyAI.ipynb`: File Jupyter Notebook chứa nội dung tương tự cho mục đích chạy thử nghiệm từng bước.
  - `download_lsun.py`: Kịch bản phụ hỗ trợ tải tập dữ liệu LSUN.
  - `run_check.sh`: Shell script hỗ trợ kiểm tra nhanh.
  - `README.md`: Tệp hướng dẫn này.

## Yêu cầu môi trường

Hệ thống cần cài đặt:
- Python 3.8+
- PyTorch và torchvision (Khuyến nghị có CUDA để chạy trên GPU)
- Các thư viện Python khác: `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `scipy`.

Cài đặt bằng pip:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm scipy
```

## Hướng dẫn chạy và tạo các file hình ảnh kết quả

### Bước 1: Chuẩn bị
Đảm bảo bạn đã tải file trọng số `wrn28_10_cifar10.pth` và đặt nó ở thư mục gốc của dự án. Khi chạy các kịch bản, các tập dữ liệu sẽ tự động được tải xuống thư mục `data/` (nếu chưa có).

### Bước 2: Sinh các hình ảnh biểu đồ
Để tạo ra các hình ảnh, bạn chỉ cần chạy lần lượt các kịch bản xuất (export). Mỗi kịch bản sẽ thực hiện tính toán và lưu file hình ảnh `.png` vào thư mục tương ứng trong `results/`.

1. **Sinh biểu đồ cơ bản (CIFAR-10 vs SVHN)**:
   ```bash
   python export_charts.py
   ```
   *Kết quả: 8 hình ảnh (chart1 tới chart8) sẽ được tạo và lưu trong thư mục `results/charts/`.*

2. **Sinh biểu đồ kết hợp (CIFAR-10 vs SVHN, LSUN, Tiny-ImageNet)**:
   ```bash
   python export_combined_figures.py
   ```
   *Kết quả: 6 hình ảnh tổng hợp (fig1 tới fig6) sẽ được lưu trong thư mục `results/charts_combined/`.*

3. **Sinh biểu đồ chi tiết lẻ và hình ảnh minh họa mẫu (ID và OOD samples)**:
   ```bash
   python export_individual_figures.py
   ```
   *Kết quả: Các hình ảnh từ figA tới figH (gồm các biểu đồ phân phối, ROC, FPR-TPR, PR, và 4 ảnh mẫu cho từng loại) sẽ được tạo ra tại `results/individual/`.*

### Bước 3: Khảo sát bổ sung (Tùy chọn)
- Chạy khảo sát chi tiết với siêu tham số Temperature:
  ```bash
  python evaluate_temperature.py
  ```
- Chạy phân tích tổng quát:
  ```bash
  python run_ood_analysis.py
  ```

### Bước 4: Biên dịch báo cáo LaTeX
Sau khi các biểu đồ đã được sinh ra, bạn có thể biên dịch lại file báo cáo (sẽ nhúng các biểu đồ mới tạo ra nếu file tex được cấu hình đúng đường dẫn):
```bash
pdflatex main.tex
```
File `main.pdf` sẽ được cập nhật nội dung.
