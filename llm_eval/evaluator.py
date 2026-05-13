from __future__ import annotations

import json
from pathlib import Path

from ai.chatbot import PhysioChatbot
from llm_eval.session_to_question import session_log_to_question


def simple_answer_score(answer: str) -> dict | None:
    if not answer or answer.startswith("❌"):
        return None

    lower = answer.lower()
    score = 0

    if any(k in lower for k in ["açı", "hareket", "tekrar", "simetri", "kalite"]):
        score += 1
    if any(k in lower for k in ["eksik", "yetersiz", "düşük", "kontrol", "tam"]):
        score += 1
    if any(k in lower for k in ["öner", "dikkat", "deneyebilirsin", "yapabilirsin", "koru"]):
        score += 1
    if len(answer) <= 700:
        score += 1

    return {
        "score": score,
        "max_score": 4
    }


class LLMEvaluator:
    def __init__(self, model_configs: list[dict]):
        self.model_configs = model_configs
        self.chatbots = [self._build_chatbot(cfg) for cfg in self.model_configs]

    def _build_chatbot(self, cfg: dict) -> PhysioChatbot:
        return PhysioChatbot(
            backend_type=cfg["backend_type"],
            api_key=cfg.get("api_key"),
            model_name=cfg.get("model_name"),
            ollama_base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
        )

    def evaluate_logs(self, logs: list[dict], output_path: str = "experiment_logs/llm_eval_results.json") -> list[dict]:
        all_results = []

        for log_item in logs:
            if "error" in log_item:
                all_results.append({
                    "source_file": log_item["file_name"],
                    "status": "log_read_error",
                    "error": log_item["error"],
                })
                continue
            print(f"İşleniyor: {log_item['file_name']}")

            for chatbot in self.chatbots:
                print(f"  -> {chatbot.backend_type} / {chatbot.model_name}")
                chatbot.reset()

            log_data = log_item["data"]
            question = session_log_to_question(log_data)

            case_result = {
                "source_file": log_item["file_name"],
                "question": question,
                "raw_summary": log_data.get("summary", {}),
                "model_outputs": [],
            }

            for chatbot in self.chatbots:
                chatbot.reset()

                result = chatbot.analyze_message(question)
                result["success"] = not result.get("answer", "").startswith("❌")
                result["auto_score"] = simple_answer_score(result.get("answer", ""))

                case_result["model_outputs"].append(result)

            all_results.append(case_result)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        return all_results