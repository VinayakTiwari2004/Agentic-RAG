import requests
from pathlib import Path

#  Load prompts from external files
SYSTEM_PROMPT = Path("llm/system_prompt.txt").read_text().strip()
USER_PROMPT_TEMPLATE = Path("llm/user_prompt_template.txt").read_text().strip()

def query_local_llm(context, user_query):
    model_name = "granite3.1-moe"

    #  Insert actual values into the user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=user_query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = requests.post(
        url="http://localhost:11434/api/chat",
        json={
            "model": model_name,
            "messages": messages,
            "stream": False
        }
    )

    if response.ok:
        return response.json()["message"]["content"]
    else:
        return f"Ollama API error: {response.status_code} {response.text}"
