import os
import json
import time
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.evaluate import evaluate

# Import metrics
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.models.llms.ollama_model import OllamaModel

# -----------------------------
# Latency & Token Metric
# -----------------------------
class LatencyAndTokenMetric:
    def __init__(self, model):
        self.model = model
        self.name = "LatencyAndTokenEfficiency"

    def score(self, query, answer, context):
        start_time = time.time()
        ctx = ""
        if isinstance(context, list):
            ctx = "\n\n".join(str(c) for c in context)
        input_tokens = len(query.split()) + len(ctx.split())
        output_tokens = len(answer.split())
        elapsed = time.time() - start_time

        return {
            "latency_seconds": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_total": input_tokens + output_tokens,
            "efficiency_ratio": output_tokens / (input_tokens + 1e-6)
        }

# -----------------------------
# Resolve JSON path robustly
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "10_startup_q_and_a.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# Prepare goldens and dataset
# -----------------------------
goldens = []
for item in data:
    context_texts = []
    for chunk in item.get("RetrievedChunks", []):
        if isinstance(chunk, dict) and "chunk_text" in chunk:
            context_texts.append(chunk["chunk_text"])
        elif isinstance(chunk, str):
            context_texts.append(chunk)
    goldens.append(
        Golden(
            input=item["Question"],
            actual_output=item["Predicted answer"],
            expected_output=item["Answer"],
            retrieval_context=context_texts
        )
    )

dataset = EvaluationDataset(goldens=goldens)

# -----------------------------
# Convert to LLM test cases
# -----------------------------
test_cases = [
    LLMTestCase(
        input=g.input,
        actual_output=g.actual_output,
        expected_output=g.expected_output,
        retrieval_context=g.retrieval_context
    ) for g in dataset.goldens
]

# -----------------------------
# Initialize Gemma 3 judge model
# -----------------------------
ollama_model = OllamaModel(model="gemma3")

# -----------------------------
# Define evaluation metrics
# -----------------------------
precision_metric = ContextualPrecisionMetric(model=ollama_model)
recall_metric = ContextualRecallMetric(model=ollama_model)
faithfulness_metric = FaithfulnessMetric(model=ollama_model)

metrics = [precision_metric, recall_metric, faithfulness_metric]

# -----------------------------
# Run evaluation
# -----------------------------
eval_results = evaluate(test_cases=test_cases, metrics=metrics)

# -----------------------------
# Latency and token stats
# -----------------------------
latency_metric = LatencyAndTokenMetric(model=ollama_model)
latency_results = [
    latency_metric.score(
        query=g.input,
        answer=g.actual_output,
        context=g.retrieval_context
    )
    for g in dataset.goldens
]

# -----------------------------
# Print results summary
# -----------------------------
print("\nEvaluation Summary (Gemma 3 as Judge):\n")
for i, testRes in enumerate(eval_results.test_results, 1):
    print(f"Test Case {i}: {testRes.input[:80]}...")
    for m in testRes.metrics_data:
        print(f"   {m.name}: {m.score:.2f}")

print("\nLatency & Token Efficiency:")
for i, res in enumerate(latency_results, 1):
    print(f"Q{i}: Latency={res['latency_seconds']:.3f}s, Tokens={res['tokens_total']}, Efficiency={res['efficiency_ratio']:.2f}")

# -----------------------------
# Save full results
# -----------------------------
results_to_save = []
for i, testRes in enumerate(eval_results.test_results):
    latency_info = latency_results[i] if i < len(latency_results) else {}
    result_entry = {
        "Question": testRes.input,
        "Predicted_Answer": testRes.actual_output,
        "Expected_Answer": testRes.expected_output
    }
    for m in testRes.metrics_data:
        result_entry[m.name] = m.score
    result_entry.update({
        "Latency_Seconds": latency_info.get("latency_seconds"),
        "Input_Tokens": latency_info.get("input_tokens"),
        "Output_Tokens": latency_info.get("output_tokens"),
        "Total_Tokens": latency_info.get("tokens_total"),
        "Efficiency_Ratio": latency_info.get("efficiency_ratio")
    })
    results_to_save.append(result_entry)

output_path = os.path.join(BASE_DIR, "evaluation_results_precision_recall_faithfulness.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=2)

print(f"\n✅ Full evaluation results saved to: {output_path}")
