import time
from transformers import pipeline, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain what a large language model is in two sentences."
NUM_RUNS = 3
MAX_NEW_TOKENS = 200

def build_chat_prompt(user_message: str) -> str:
    return (
        "<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{user_message}</s>\n"
        "<|assistant|>\n"
    )

def main():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompt = build_chat_prompt(PROMPT)
    print(f"Prompt: {PROMPT}")
    print(f"\nRunning {NUM_RUNS} inference passes...\n")

    col = "{:<6} {:>12} {:>10} {:>12}"
    print(col.format("Run", "Latency (s)", "Tokens", "Tok/s"))
    print("-" * 44)

    latencies = []
    tps_values = []

    for i in range(1, NUM_RUNS + 1):
        start = time.perf_counter()
        result = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        elapsed = time.perf_counter() - start

        reply = result[0]["generated_text"][len(prompt):]
        token_count = len(tokenizer.encode(reply))
        tps = token_count / elapsed

        latencies.append(elapsed)
        tps_values.append(tps)

        print(col.format(i, f"{elapsed:.2f}", token_count, f"{tps:.1f}"))

    print("=" * 44)
    print(col.format("Min",  f"{min(latencies):.2f}",  "-", f"{min(tps_values):.1f}"))
    print(col.format("Max",  f"{max(latencies):.2f}",  "-", f"{max(tps_values):.1f}"))
    avg_lat = sum(latencies) / NUM_RUNS
    avg_tps = sum(tps_values) / NUM_RUNS
    print(col.format("Avg",  f"{avg_lat:.2f}",          "-", f"{avg_tps:.1f}"))
    print("=" * 44)

if __name__ == "__main__":
    main()
