"""Evaluation script for prompt injection detection."""

import json
import re
from collections.abc import Callable

from langchain.agents.middleware.prompt_injection_guard import PromptInjectionGuardMiddleware
from langchain.agents.preprocessing.multimodal_input_processor import MultiModalInputProcessor
from langchain_core.messages import HumanMessage

DATASET_PATH = "libs/langchain_v1/evaluation/prompt_injection_dataset.json"


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _predict_regex_only(
    middleware: PromptInjectionGuardMiddleware,
    processed_text: str,
) -> str:
    matches = middleware._detect_prompt_injection(processed_text)
    return "attack" if matches else "benign"


def _predict_hybrid(
    middleware: PromptInjectionGuardMiddleware,
    processed_text: str,
) -> str:
    segments = re.split(r"[.!?]\s+", processed_text)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        if middleware._detect_prompt_injection(segment):
            return "attack"

        try:
            result = middleware.intent_agent.analyze(segment)
        except Exception:
            continue

        is_attack = result.get("is_attack") is True
        confidence = result.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0

        if is_attack and confidence_value >= 0.6:
            return "attack"

    return "benign"


def _predict_full_agentguard(
    middleware: PromptInjectionGuardMiddleware,
    processed_text: str,
) -> str:
    try:
        middleware.before_model(
            {"messages": [HumanMessage(content=processed_text)]},
            runtime=None,
        )
        return "benign"
    except ValueError:
        return "attack"
    except Exception:
        return "benign"


def _run_experiment(
    dataset: list[dict[str, str]],
    *,
    title: str,
    middleware: PromptInjectionGuardMiddleware,
    processor: MultiModalInputProcessor,
    predictor: Callable[[PromptInjectionGuardMiddleware, str], str],
) -> None:

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = len(dataset)
    print(f"\n=== {title} ===")
    print(f"Starting evaluation on {total} samples...")

    for i, sample in enumerate(dataset):
        text = sample["text"]
        label = sample["label"]
        print(f"[{i + 1}/{total}] Processing...")
        print(f"[{i + 1}/{total}] Label={label} -> Running detection...")

        processed_text = processor.process(
            {
                "type": "text",
                "data": text,
            }
        )

        prediction = predictor(middleware, processed_text)
        progress = int((i + 1) / total * 20)
        bar = "#" * progress + "-" * (20 - progress)
        print(f"[{bar}] {i + 1}/{total}")
        print(f"[{i + 1}/{total}] {label} -> {prediction}")
        print("-" * 40)

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

    processor = MultiModalInputProcessor()
    regex_middleware = PromptInjectionGuardMiddleware(
        strategy="block",
        use_intent_agent=False,
    )
    hybrid_middleware = PromptInjectionGuardMiddleware(
        strategy="block",
        use_intent_agent=True,
    )
    full_middleware = PromptInjectionGuardMiddleware(
        strategy="block",
        use_intent_agent=True,
    )

    _run_experiment(
        dataset,
        title="Evaluation Results (Regex Only)",
        middleware=regex_middleware,
        processor=processor,
        predictor=_predict_regex_only,
    )
    _run_experiment(
        dataset,
        title="Evaluation Results (Hybrid)",
        middleware=hybrid_middleware,
        processor=processor,
        predictor=_predict_hybrid,
    )
    _run_experiment(
        dataset,
        title="Evaluation Results (Full AgentGuard)",
        middleware=full_middleware,
        processor=processor,
        predictor=_predict_full_agentguard,
    )


if __name__ == "__main__":
    main()
