# Reflection — Lab 19

- **Tên:** Nguyễn Ngọc Minh
- **Cohort:** 4
- **Path đã chạy:** _lite_

---

## Câu hỏi (≤ 200 chữ)

Trên 50 queries, Hybrid thắng tổng thể với 78,6% Precision@10, so với BM25 77,8% và Vector 73,2%. Theo lát cắt, `exact`: BM25 và Hybrid cùng đạt 96,7%; `paraphrase`: BM25 đạt 33,3%, Hybrid 32,0%, Vector 24,0%; `mixed`: Hybrid thắng với 100%, Vector 98,5% và BM25 97,0%. Kết quả paraphrase cho thấy BGE-small được huấn luyện chủ yếu bằng tiếng Anh chưa mạnh với cách diễn đạt tiếng Việt; một số từ khóa trùng vẫn giúp BM25. Hybrid hữu ích nhất khi query vừa có từ khóa vừa diễn đạt ý nghĩa. Tôi sẽ chọn BM25 thuần cho mã, tên riêng hoặc thuật ngữ khớp chính xác khi lexical search đã đủ tốt và cần giảm chi phí. Tôi chỉ chọn Vector thuần cho paraphrase khi đã xác nhận embedding đa ngôn ngữ đạt chất lượng và không cần khớp chính xác identifier.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid không dẫn đầu mọi lát cắt: BM25 vẫn nhỉnh hơn ở paraphrase tiếng Việt với embedding model mặc định. Kết quả phụ thuộc vào ngôn ngữ của model và đặc tính của golden set.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
