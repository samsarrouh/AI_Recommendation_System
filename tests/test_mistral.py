import subprocess

OLLAMA_PATH = "C:\\Users\\samer\\AppData\\Local\\Programs\\Ollama\\ollama.exe"

def query_mistral(prompt):
    result = subprocess.run(
        [OLLAMA_PATH, "run", "mistral", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

response = query_mistral("Summarize the benefits of AI-driven product recommendations.")
print("🧠 Mistral 7B Response:\n", response)
