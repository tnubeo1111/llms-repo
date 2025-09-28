# Hiểu Biết Kiến Trúc Transformer

## Transformer là gì?
Transformer là một **kiến trúc mạng nơ-ron** được giới thiệu trong bài báo “**Attention is All You Need**” (2017), trở thành nền tảng cho các mô hình ngôn ngữ lớn (LLMs) như GPT và BERT. Transformer xử lý dữ liệu chuỗi (như văn bản) hiệu quả, đặc biệt trong các nhiệm vụ xử lý ngôn ngữ tự nhiên (NLP) như dịch máy, tạo văn bản, và phân tích cảm xúc. Ngoài ra, nó còn được áp dụng trong thị giác máy tính (Vision Transformers - ViT).

**Điểm mạnh**:
- **Xử lý song song**: Xử lý toàn bộ câu cùng lúc, nhanh hơn RNN hay LSTM.
- **Self-Attention**: Tập trung vào các từ quan trọng, nắm bắt mối quan hệ ngữ cảnh dài hạn.

## Các thành phần chính của Transformer

### 1. Bộ mã hóa (Encoder)
- **Chức năng**: Chuyển câu đầu vào thành **vector embedding** biểu diễn ý nghĩa ngữ cảnh.
- **Cấu trúc**:
  - Gồm **N lớp encoder** (thường 6 hoặc 12 lớp).
  - Mỗi lớp có:
    - **Multi-Head Self-Attention**: Xác định mức độ quan trọng của các từ.
    - **Feed-Forward Neural Network**: Tinh chỉnh vector embedding.
    - **Layer Normalization và Residual Connections**: Ổn định huấn luyện.
- **Quy trình**:
  1. Token hóa câu (ví dụ: “Tôi ăn cơm” → ["Tôi", "ăn", "cơm"]).
  2. Ánh xạ token thành vector embedding (ví dụ: “Tôi” → [0.5, -0.2, 0.1, …]).
  3. Self-attention xác định mối quan hệ (ví dụ: “ăn” liên quan đến “cơm”).
  4. Feed-forward tạo vector biểu diễn ngữ cảnh đầy đủ.
- **Ví dụ**: Trong “Con mèo ăn cá trên bàn”, encoder xác định “cá” quan trọng hơn “bàn” khi hiểu hành động của “mèo”.

### 2. Bộ giải mã (Decoder)
- **Chức năng**: Dự đoán từ tiếp theo trong chuỗi đầu ra, sử dụng vector từ encoder.
- **Cấu trúc**:
  - Gồm **N lớp decoder**.
  - Mỗi lớp có:
    - **Masked Multi-Head Self-Attention**: Chỉ xem từ trước đó.
    - **Encoder-Decoder Attention**: Liên kết với đầu vào từ encoder.
    - **Feed-Forward Neural Network**: Tinh chỉnh vector đầu ra.
    - **Layer Normalization và Residual Connections**.
- **Quy trình**:
  1. Nhận văn bản đầu ra tạm thời (ví dụ: “Tôi ăn…”).
  2. Dự đoán từ tiếp theo (ví dụ: “cơm”).
  3. Lặp lại để hoàn thành chuỗi.
- **Ví dụ**: Dịch “I eat rice” → “Tôi ăn cơm”, decoder sinh từng từ dựa trên encoder.

![Parameters](/images/transformers-lec4.png)

### 3. Cơ chế tự chú ý (Self-Attention)
- **Khái niệm**: Xác định mức độ quan trọng của từng từ đối với các từ khác.
- **Cách hoạt động**:
  - Mỗi token có ba vector: **Query (Q)**, **Key (K)**, **Value (V)**.
  - Tính **điểm chú ý**: `Score = Q * K^T / sqrt(d_k)`, chuẩn hóa bằng softmax.
  - Nhân xác suất với Value để tạo đầu ra.
  - **Multi-Head Attention**: Thực hiện nhiều lần self-attention để nắm bắt nhiều khía cạnh.
- **Ví dụ**: Trong “Con mèo ăn cá trên bàn”, “ăn” chú ý nhiều đến “cá” hơn “bàn”.
- **Ý nghĩa**: Hiểu mối quan hệ dài hạn, nhanh hơn RNN/LSTM.

### 4. Token hóa và Vector Embedding
- **Token hóa**: Chia câu thành token (ví dụ: “Tôi ăn cơm” → ["Tôi", "ăn", "cơm"]).
- **Vector Embedding**: Ánh xạ token thành vector số trong không gian đa chiều (ví dụ: “king” và “queen” gần nhau hơn “king” và “apple”).
- **Positional Encoding**: Thêm thông tin vị trí (dùng hàm sin/cos) để giữ thứ tự từ.

## Cách Transformer hoạt động
1. **Đầu vào**: Token hóa câu, chuyển thành vector embedding, thêm positional encoding.
2. **Encoder**: Tạo vector biểu diễn ngữ cảnh qua self-attention và feed-forward.
3. **Decoder**: Sinh chuỗi đầu ra từng từ, sử dụng masked self-attention và encoder-decoder attention.
4. **Đầu ra**: Chuỗi văn bản mới (bản dịch, câu trả lời).

**Ví dụ dịch máy**:
- Đầu vào: “I eat rice”.
- Encoder: Tạo vector cho “I”, “eat”, “rice”.
- Decoder: Sinh “Tôi ăn cơm” từng từ.

## Các biến thể của Transformer
- **BERT**: Dùng encoder, ngữ cảnh hai chiều, tốt cho phân tích cảm xúc, phân loại văn bản.
- **GPT**: Dùng decoder, sinh văn bản tuần tự, tốt cho tạo văn bản, trả lời câu hỏi.

## Tại sao Transformer quan trọng?
- **Hiệu quả**: Xử lý song song, nhanh hơn RNN/LSTM.
- **Ngữ cảnh dài hạn**: Self-attention nắm bắt mối quan hệ xa.
- **Ứng dụng đa dạng**: NLP (dịch máy, tạo văn bản) và thị giác máy tính (phân loại ảnh).

## Chi tiết bổ sung
### Cách tính Self-Attention
- Mỗi token có vector **Q**, **K**, **V**.
- Tính điểm chú ý: `Score = Q * K^T / sqrt(d_k)`, chuẩn hóa bằng softmax.
- Đầu ra: Nhân xác suất với Value.
- **Ví dụ**: Trong “Con mèo ăn cá”, “ăn” chú ý nhiều đến “cá” qua điểm chú ý cao.

### Multi-Head vs. Single-Head Attention
- **Single-Head**: Một lần self-attention, nắm bắt một khía cạnh.
- **Multi-Head**: Nhiều lần self-attention, mỗi lần tập trung vào khía cạnh khác (cú pháp, ngữ nghĩa).
- **Ví dụ**: “Con mèo ăn cá trên bàn” – đầu 1 chú ý “ăn” và “cá”, đầu 2 chú ý “mèo” và “cá”.

### Positional Encoding
- Thêm vector vị trí (sin/cos) vào embedding để giữ thứ tự từ.
- **Ví dụ**: “Tôi ăn cơm” – “Tôi” có vị trí 1, “ăn” có vị trí 2.

### Ứng dụng cụ thể
- **Dịch máy**: “I eat rice” → “Tôi ăn cơm”.
- **Tạo văn bản (GPT)**: Sinh câu trả lời tự nhiên.
- **Phân tích cảm xúc (BERT)**: Phân loại văn bản tích cực/tiêu cực.
- **Vision Transformers**: Phân loại ảnh, phát hiện khối u.