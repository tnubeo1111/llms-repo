### Tóm tắt  
Bài giảng này giới thiệu tổng quan về kiến trúc Transformer – nền tảng chính của các mô hình ngôn ngữ lớn (LLM) hiện đại như GPT. Transformer được giới thiệu lần đầu trong bài báo “Attention is All You Need” năm 2017 và đã tạo ra bước đột phá lớn trong các nhiệm vụ dịch máy, chuyển đổi ngôn ngữ. Kiến trúc này bao gồm hai thành phần chính: bộ mã hóa (encoder) và bộ giải mã (decoder). Quá trình hoạt động của Transformer bắt đầu bằng việc tách câu thành các token, ánh xạ các token này thành các vector embedding trong không gian đa chiều để biểu diễn ý nghĩa ngữ cảnh. Bộ decoder sau đó sử dụng các vector này cùng với phần văn bản đầu ra tạm thời để dự đoán từng từ tiếp theo trong câu dịch.  

![Parameters](/images/transformers-lec4.png)

Bài giảng cũng giải thích khái niệm cơ chế tự chú ý (self-attention), cho phép mô hình xác định mức độ quan trọng của từng từ trong câu, giúp nắm bắt các mối quan hệ dài hạn giữa các từ trong ngữ cảnh.  

Ngoài ra, bài giảng phân biệt rõ ràng giữa Transformer và LLM, cũng như các biến thể sau này của Transformer như BERT và GPT. BERT sử dụng bộ mã hóa và tập trung vào việc dự đoán các từ bị ẩn trong câu bằng cách xem xét cả ngữ cảnh hai chiều (trái và phải), rất phù hợp cho các tác vụ như phân tích cảm xúc. Ngược lại, GPT chỉ sử dụng bộ giải mã và dự đoán từ tiếp theo theo hướng từ trái sang phải, phù hợp cho việc tạo văn bản tuần tự.  

Cuối cùng, bài giảng nhấn mạnh rằng không phải tất cả các Transformer đều là LLM và cũng không phải tất cả LLM đều dựa trên Transformer, vì trước đây còn có các mô hình mạng nơ-ron hồi tiếp (RNN), mạng bộ nhớ dài-ngắn hạn (LSTM) và các kiến trúc CNN cũng có thể thực hiện các tác vụ mô hình ngôn ngữ.  

### Điểm nhấn quan trọng  
- 🔑 Transformer là kiến trúc nền tảng của hầu hết các mô hình ngôn ngữ lớn hiện đại.  
- 🧩 Quá trình tiền xử lý bao gồm token hóa và ánh xạ token thành vector embedding để biểu diễn ngữ nghĩa.  
- 🔍 Cơ chế tự chú ý (self-attention) là điểm mấu chốt giúp mô hình hiểu được mối quan hệ dài hạn giữa các từ trong câu.  
- 🔄 Transformer bao gồm hai thành phần chính: encoder và decoder, nhưng GPT chỉ có decoder, BERT chỉ có encoder.  
- 🧠 BERT là mô hình hai chiều, hiệu quả trong các nhiệm vụ như phân tích cảm xúc nhờ khả năng dự đoán từ bị ẩn trong ngữ cảnh.  
- ⚙️ GPT dự đoán từ tiếp theo theo trình tự từ trái sang phải, áp dụng cho việc sinh văn bản.  
- 🌐 Không phải tất cả Transformers đều là LLM và không phải tất cả LLM đều dựa trên Transformer; có các mô hình khác như RNN, LSTM và CNN cũng có thể là LLM.  

### Những hiểu biết quan trọng  
- 📜 **Sự phát triển của Transformer từ bài báo “Attention is All You Need”**: Bài báo năm 2017 đã mở ra kỷ nguyên mới cho các mô hình ngôn ngữ, thay thế các mô hình RNN truyền thống bằng kiến trúc dựa trên attention, giúp xử lý song song hiệu quả và nắm bắt được các mối quan hệ ngữ cảnh dài hạn. Điều này làm cho các mô hình như GPT và BERT có thể đạt hiệu suất cực kỳ cao trong nhiều nhiệm vụ NLP.  

- 🧮 **Token hóa và vector embedding giải quyết bài toán biểu diễn ngữ cảnh**: Việc chuyển đổi câu thành các token và ánh xạ token thành vector trong không gian đa chiều cho phép mô hình hiểu được mối liên hệ ngữ nghĩa giữa các từ, ví dụ như “king”, “man”, “woman” có vector gần nhau ngữ nghĩa hơn so với “king” và một loại trái cây. Đây là bước nền tảng để mô hình có thể xử lý ngôn ngữ tự nhiên hiệu quả.  

- 🔄 **Phân biệt rõ ràng giữa encoder và decoder trong Transformer**: Encoder chịu trách nhiệm mã hóa thông tin đầu vào thành embedding, còn decoder dựa vào embedding và thông tin đầu ra tạm thời để sinh kết quả. Điều này tạo nên sự linh hoạt trong các ứng dụng khác nhau, ví dụ như dịch máy, tạo văn bản, hay hoàn thiện câu.  

- 👁️‍🗨️ **Cơ chế self-attention cho phép mô hình xử lý phụ thuộc dài hạn**: Một trong những điểm mạnh lớn nhất của Transformer chính là khả năng cân nhắc tất cả các từ trong câu (hoặc đoạn văn) khi dự đoán từ tiếp theo, không bị giới hạn bởi khoảng cách từ, giúp mô hình duy trì ngữ cảnh một cách toàn diện và chính xác hơn.  

- ⚖️ **Sự khác biệt trong cách hoạt động của BERT và GPT**: BERT dự đoán các từ bị ẩn trong câu dựa trên ngữ cảnh hai chiều nên phù hợp với các tác vụ như phân tích cảm xúc, hiểu ngữ nghĩa sâu sắc. Còn GPT sinh từ theo cách tuần tự, rất thích hợp cho việc tạo nội dung hoặc trả lời câu hỏi. Việc hiểu rõ sự khác biệt này giúp lựa chọn mô hình phù hợp cho từng nhiệm vụ cụ thể.  

- 🌍 **Transformer không chỉ dành cho ngôn ngữ**: Transformer còn được áp dụng cho các lĩnh vực khác như thị giác máy tính với Vision Transformers (ViT), giúp phân loại ảnh, phát hiện dị vật trên đường, phân loại khối u,... Điều này cho thấy tính đa dụng và mạnh mẽ của kiến trúc Transformer vượt ra ngoài giới hạn xử lý ngôn ngữ.  

- 🔄 **Không đồng nhất giữa Transformer và LLM**: Hiểu rằng Transformer là một kiến trúc mạng nơ-ron trong khi LLM là một khái niệm rộng hơn về mô hình xử lý ngôn ngữ, có thể dựa trên nhiều kiến trúc khác nhau như RNN, LSTM, CNN. Việc không nhầm lẫn hai khái niệm này rất quan trọng để có cái nhìn đúng đắn về công nghệ và lựa chọn phương pháp phù hợp khi phát triển hoặc đánh giá mô hình.  
