import time
from transformers import pipeline

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain what a large language model is in two sentences."

def build_chat_prompt(user_message: str) -> str:
    return (
        "<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{user_message}</s>\n"
        "<|assistant|>\n"
    )

def main():
    print(f"Loading model: {MODEL_ID}")
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        device="cpu",
    )

    prompt = build_chat_prompt(PROMPT)
    print(f"\nPrompt: {PROMPT}\n")

    start = time.perf_counter()
    result = pipe(
        prompt,
        max_new_tokens=200,
        do_sample=False,
    )
    elapsed = time.perf_counter() - start

    generated = result[0]["generated_text"]
    # Strip the prompt prefix so we only print the model's reply
    reply = generated[len(prompt):]
    print(f"Response:\n{reply.strip()}")
    print(f"\nInference time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
