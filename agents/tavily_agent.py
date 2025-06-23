import requests

# Directly put your API key here (not recommended for production)
TAVILY_API_KEY = "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"

def tavily_search(query):
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
        json={"query": query, "num_results": 3}
    )

    data = response.json()

    if "results" not in data:
        return []

    #  Return structured list
    return [
        {"url": res.get("url", ""), "content": res.get("content", "")}
        for res in data["results"]
    ]

