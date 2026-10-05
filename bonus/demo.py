"""Run a five-query HybridMemoryAgent smoke demo from the repository root."""
from __future__ import annotations

from agent import HybridMemoryAgent


def main() -> None:
    agent = HybridMemoryAgent()
    user_id = "u_001"
    memories = [
        "User muốn tự động mở rộng cloud khi lưu lượng traffic tăng; đang xây autoscaling.",
        "User thích câu trả lời tiếng Việt ngắn gọn, giữ nguyên thuật ngữ API và vector store.",
        "User theo dõi số lượng truy vấn trong một giờ; queries_last_hour hiện là 11.",
        "User có tốc độ đọc 187 từ mỗi phút và chủ đề cloud yêu thích là cloud.",
        "User quan tâm bảo mật dữ liệu cá nhân và quyền truy cập trong hệ thống.",
    ]
    for text in memories:
        agent.remember(text, user_id)

    queries = [
        "tự động mở rộng cloud khi lưu lượng tăng",
        "cách trả lời bằng tiếng Việt và giữ nguyên thuật ngữ API",
        "số lượng truy vấn trong một giờ",
        "tốc độ đọc và chủ đề cloud yêu thích",
        "bảo mật dữ liệu cá nhân và quyền truy cập",
    ]
    for number, query in enumerate(queries, start=1):
        context = agent.recall(query, user_id, top_k=2)
        print()
        print(f"[{number}] Query: {query}")
        print("  Stable profile:", context["profile"])
        for memory in context["memories"]:
            print(f"  Episodic context ({memory['rrf_score']:.4f}): {memory['text']}")


if __name__ == "__main__":
    main()
