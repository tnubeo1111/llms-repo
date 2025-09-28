### Mục lục nội dung bài giảng

| Phần | Chủ đề chính                                          |
|-------|-----------------------------------------------------|
| 1     | Mô hình ngôn ngữ lớn (Large Language Model - LLM) là gì?            |
| 2     | Ý nghĩa của từ “Large” trong LLM                        |
| 3     | Sự khác biệt giữa LLM hiện đại và các mô hình NLP trước đây      |
| 4     | Bí quyết tạo nên sức mạnh của LLM – kiến trúc Transformer         |
| 5     | Phân biệt các thuật ngữ AI, Machine Learning, Deep Learning, LLM, Generative AI |
| 6     | Ứng dụng thực tiễn của LLM                              |

---

### 1. Mô hình ngôn ngữ lớn (LLM) là gì?

- **Định nghĩa cơ bản:** LLM là một loại **mạng nơ-ron sâu (deep neural network)** được thiết kế  dễ hiểu, tạo ra và phản hồi văn bản giống như con người.
- Mạng nơ-ron mô phỏng cách hoạt động của các tế bào thần kinh trong não, gồm nhiều lớp neuron xử lý đầu vào và tạo ra đầu ra.
- LLM chuyên xử lý các tác vụ liên quan đến ngôn ngữ tự nhiên như: hiểu văn bản, tạo văn bản, trả lời câu hỏi, dịch thuật, phân tích cảm xúc.
- Ví dụ minh họa: ChatGPT – một LLM có thể hỏi về sở thích thư giãn, trả lời và lên kế hoạch như một người thật.
- **Tóm lại:** LLM là mạng nơ-ron sâu được huấn luyện trên lượng dữ liệu khổng lồ, giúp thực hiện các tác vụ ngôn ngữ một cách linh hoạt và ngày càng giống con người.

---

### 2. Tại sao gọi là “Large” Language Model?

- “Large” ám chỉ **kích thước mô hình, tính bằng số lượng tham số (parameters)**.
- Các mô hình LLM hiện đại có đến hàng tỷ hoặc thậm chí hàng nghìn tỷ tham số. Chẳng hạn:
  
| Mô hình    | Số tham số (Parameters)          |
|------------|---------------------------------|
| GPT-3 nhỏ  | 125 triệu                      |
| GPT-3 trung bình | 350 triệu                   |
| GPT-3 lớn  | 760 triệu                      |
| GPT-3 13B  | 13 tỷ                         |
| GPT-3 175B | 175 tỷ                        |
| GPT-4      | *Không được công bố chính xác, nhưng lớn hơn 175 tỷ* |

- Số lượng tham số tăng theo cấp số nhân qua các thế hệ GPT 1 → 2 → 3.
- Sự tăng trưởng tham số này là một phần quan trọng giúp LLM trở nên mạnh mẽ hơn.
- Biểu đồ lịch sử số lượng tham số trong mô hình AI từ năm 1950 đến 2022 cho thấy sự bùng nổ kích thước trong thập kỷ gần đây, đặc biệt ở LLM.
- “Language Model” vì LLM chỉ xử lý ngôn ngữ, không phải các dạng dữ liệu khác như hình ảnh hay video.

![Parameters](/images/parameters-lec2.png)

---

### 3. Sự khác biệt giữa LLM hiện đại và các mô hình NLP trước đây

- **Các mô hình NLP cũ** thường được thiết kế cho từng tác vụ cụ thể như dịch máy, phân tích cảm xúc, nhận diện chủ đề, v.v.
- **LLM hiện đại** có khả năng thực hiện đa dạng tác vụ NLP chỉ với một kiến trúc chung – rất linh hoạt.
- Ví dụ: Cùng một mô hình GPT có thể làm tốt việc hoàn thành câu, dịch thuật, viết email, trả lời câu hỏi…
- Các tác vụ “dễ dàng” với LLM như soạn email theo yêu cầu gần như không thể với các mô hình NLP cũ.
- LLM có khả năng ứng dụng rộng lớn và hiệu quả hơn nhiều so với các mô hình trước.

---

### 4. Bí quyết tạo nên sức mạnh của LLM: Kiến trúc Transformer

- **Không phải ngẫu nhiên LLM lại xuất sắc đến vậy**, bí quyết nằm ở kiến trúc mạng gọi là **Transformer**.
- Transformer được giới thiệu trong bài báo nổi tiếng năm 2017 “Attention is All You Need” của nhóm Google Brain.
- Bài báo này đã có hơn 100,000 trích dẫn chỉ trong 5 năm, minh chứng cho tầm ảnh hưởng cực lớn.
- Cấu trúc Transformer sử dụng cơ chế “attention” cho phép mô hình tập trung vào các phần quan trọng của văn bản khi xử lý.
- Các thành phần chính trong Transformer gồm: embedding đầu vào, multi-head attention, feed forward layers, positional encoding, normalization layers, v.v.
- Transformer là nền tảng giúp LLM có thể xử lý dữ liệu chuỗi văn bản lớn, đa dạng và tạo ra đầu ra chất lượng cao.
- Bài giảng sẽ đi sâu vào từng phần của Transformer trong các bài học tiếp theo.

---

### 5. Phân biệt các thuật ngữ AI, Machine Learning, Deep Learning, LLM, Generative AI

| Thuật ngữ          | Định nghĩa và Phạm vi                                          | Ví dụ minh họa                                |
|--------------------|---------------------------------------------------------------|-----------------------------------------------|
| **Artificial Intelligence (AI)** | Mọi hệ thống máy tính thể hiện hành vi có trí tuệ (rule-based hoặc học được). | Chatbot dựa trên quy tắc (rule-based chatbot). |
| **Machine Learning (ML)**        | Hệ thống máy tính có khả năng học và thích nghi dựa trên dữ liệu.               | Hệ thống dự đoán bệnh tim dựa trên dữ liệu bệnh nhân. |
| **Deep Learning (DL)**           | Một nhánh của ML sử dụng mạng nơ-ron sâu, chủ yếu là mạng nơ-ron nhân tạo.    | Mạng CNN nhận dạng hình ảnh, mạng nơ-ron phân loại chữ viết tay. |
| **Large Language Models (LLM)**  | Mạng nơ-ron sâu chuyên xử lý và tạo ra văn bản ngôn ngữ tự nhiên, thuộc DL.    | GPT-3, GPT-4, ChatGPT.                          |
| **Generative AI**                | AI tạo ra nội dung mới (văn bản, hình ảnh, âm thanh, video), kết hợp DL và nhiều loại dữ liệu. | DALL·E (tạo ảnh từ mô tả), ChatGPT (tạo văn bản). |

- **AI là phạm vi rộng nhất**, bao gồm cả hệ thống rule-based không học được.
- **ML là tập con của AI**, tập trung vào máy học từ dữ liệu.
- **DL là tập con của ML**, chuyên sâu về mạng nơ-ron sâu.
- **LLM là tập con của DL**, chuyên về xử lý ngôn ngữ tự nhiên.
- **Generative AI** là sự kết hợp của DL và các mô hình tạo nội dung đa phương tiện, không chỉ giới hạn ở văn bản.

---

### 6. Ứng dụng của Large Language Models

LLM có thể được ứng dụng rộng rãi trong nhiều lĩnh vực, tiêu biểu gồm:

- **Tạo nội dung mới:** Viết thơ, truyện, sách, bài báo, nội dung sáng tạo độc đáo chưa từng tồn tại.
- **Chatbots và trợ lý ảo:** Giao tiếp tự nhiên, hỗ trợ khách hàng, tư vấn, tự động hóa dịch vụ, ứng dụng trong ngân hàng, hãng hàng không, nhà hàng, giáo dục.
- **Dịch máy:** Chuyển đổi văn bản giữa các ngôn ngữ một cách trôi chảy và chính xác, hỗ trợ cả một số ngôn ngữ địa phương.
- **Phân tích cảm xúc:** Phân tích thái độ, cảm xúc trong văn bản, phát hiện ngôn từ thù địch trên mạng xã hội.
- **Tạo câu hỏi trắc nghiệm, tóm tắt văn bản:** Hỗ trợ giáo viên và nhà nghiên cứu trong việc chuẩn bị tài liệu, bài giảng.

**Ví dụ thực tế**: Một cổng thông tin dành cho giáo viên được thiết kế dựa trên LLM có thể:
- Tạo kế hoạch bài học theo chương trình học cụ thể.
- Tạo câu hỏi trắc nghiệm với các mức độ khó khác nhau.
- Tóm tắt và viết lại văn bản nhanh chóng, giúp tiết kiệm thời gian.

---

### Từ khóa chính

- Mô hình ngôn ngữ lớn (Large Language Model - LLM)
- Transformer
- Neural Network (Mạng nơ-ron)
- Tham số mô hình (Parameters)
- Trí tuệ nhân tạo (Artificial Intelligence - AI)
- Máy học (Machine Learning - ML)
- Học sâu (Deep Learning - DL)
- Generative AI (AI tạo nội dung)
- Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)
- Chatbot, dịch máy, tạo nội dung, phân tích cảm xúc
