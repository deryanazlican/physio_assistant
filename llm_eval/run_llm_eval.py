from __future__ import annotations

import os
from pathlib import Path

from llm_eval.log_loader import load_experiment_logs
from llm_eval.evaluator import LLMEvaluator


def main():
    logs = load_experiment_logs("experiment_logs")

    if not logs:
        print("❌ experiment_logs içinde JSON log bulunamadı.")
        return

    model_configs = [
        {
            "backend_type": "gemini",
            "model_name": "gemini-2.5-flash",
            "api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        },
        {
            "backend_type": "together",
            "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "api_key": os.getenv("TOGETHER_API_KEY"),
        },
        {
            "backend_type": "ollama",
            "model_name": "llama3.1",
            "ollama_base_url": "http://localhost:11434",
        },
    ]

    Path("llm_outputs").mkdir(exist_ok=True)

    evaluator = LLMEvaluator(model_configs=model_configs)
    results = evaluator.evaluate_logs(
        logs,
        output_path="llm_outputs/llm_eval_results.json"
    )

    print(f"✅ Tamamlandı. {len(results)} log değerlendirildi.")
    print("📁 Sonuç: llm_outputs/llm_eval_results.json")


if __name__ == "__main__":
    main()