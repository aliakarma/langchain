"""Evaluation script for prompt injection detection."""

import json

from langchain.agents.middleware.prompt_injection_guard import PromptInjectionGuardMiddleware
from langchain.agents.preprocessing.multimodal_input_processor import MultiModalInputProcessor
from langchain_core.messages import HumanMessage

DATASET_PATH = "libs/langchain_v1/evaluation/prompt_injection_dataset.json"


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _run_experiment(dataset: list[dict[str, str]], *, title: str) -> None:
    middleware = PromptInjectionGuardMiddleware(
        strategy="block",
        use_intent_agent=True,
    )
    processor = MultiModalInputProcessor()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in dataset:
        text = sample["text"]
        label = sample["label"]

        processed_text = processor.process(
            {
                "type": "text",
                "data": text,
            }
        )

        try:
            middleware.before_model(
                {"messages": [HumanMessage(content=processed_text)]},
                runtime=None,
            )
            prediction = "benign"
        except ValueError:
            prediction = "attack"

        if label == "attack" and prediction == "attack":
            tp += 1
        elif label == "benign" and prediction == "benign":
            tn += 1
        elif label == "benign" and prediction == "attack":
            fp += 1
        elif label == "attack" and prediction == "benign":
            fn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1_score = _safe_divide(2 * (precision * recall), precision + recall)

    print(title)
    print("-------------------")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print()


def main() -> None:
    with open(DATASET_PATH, encoding="utf-8") as file:
        dataset = json.load(file)

    _run_experiment(
        dataset,
        title="Evaluation Results (Full AgentGuard)",
    )


if __name__ == "__main__":
    main()
