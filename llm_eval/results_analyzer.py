from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_results(path="llm_outputs/llm_eval_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(results):
    rows = []

    for item in results:
        source_file = item.get("source_file")
        question = item.get("question")

        for output in item.get("model_outputs", []):
            auto_score = output.get("auto_score") or {}
            rows.append({
                "source_file": source_file,
                "question": question,
                "backend_type": output.get("backend_type"),
                "model_name": output.get("model_name"),
                "success": output.get("success", False),
                "latency": output.get("latency", 0.0),
                "score": auto_score.get("score"),
                "max_score": auto_score.get("max_score"),
                "answer_length": len(output.get("answer", "")),
            })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    summary = (
        df.groupby(["backend_type", "model_name"], dropna=False)
        .agg(
            total_cases=("source_file", "count"),
            success_count=("success", "sum"),
            avg_latency=("latency", "mean"),
            avg_score=("score", "mean"),
            avg_answer_length=("answer_length", "mean"),
        )
        .reset_index()
    )

    summary["success_rate"] = summary["success_count"] / summary["total_cases"]
    return summary


def save_summary(summary: pd.DataFrame, out_path="experiment_logs/llm_eval_summary.csv"):
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")


def plot_success_rate(summary: pd.DataFrame, out_path="experiment_logs/success_rate.png"):
    plt.figure(figsize=(8, 5))
    labels = summary["backend_type"] + "\n" + summary["model_name"]
    plt.bar(labels, summary["success_rate"])
    plt.ylabel("Success Rate")
    plt.title("LLM Success Rate Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_avg_latency(summary: pd.DataFrame, out_path="experiment_logs/avg_latency.png"):
    plt.figure(figsize=(8, 5))
    labels = summary["backend_type"] + "\n" + summary["model_name"]
    plt.bar(labels, summary["avg_latency"])
    plt.ylabel("Average Latency (s)")
    plt.title("Average Latency Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_avg_score(summary: pd.DataFrame, out_path="experiment_logs/avg_score.png"):
    plt.figure(figsize=(8, 5))
    labels = summary["backend_type"] + "\n" + summary["model_name"]
    plt.bar(labels, summary["avg_score"].fillna(0))
    plt.ylabel("Average Score")
    plt.title("Average Feedback Score Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    results = load_results()
    df = build_dataframe(results)
    summary = summarize(df)

    Path("experiment_logs").mkdir(exist_ok=True)

    save_summary(summary)
    plot_success_rate(summary)
    plot_avg_latency(summary)
    plot_avg_score(summary)

    print(summary)
    print("✅ Özet CSV ve grafikler kaydedildi.")


if __name__ == "__main__":
    main()