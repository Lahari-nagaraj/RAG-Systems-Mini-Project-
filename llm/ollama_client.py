import requests

class OllamaLLM:
    def __init__(self, model="mistral"):
        self.model = model

    def generate(self, prompt):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]