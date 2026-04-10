import os
import json
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq(api_key="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")


EXTRACTION_PROMPT = """You are a metadata extractor for a research paper database.

Your job is to extract filtering information FROM THE QUESTION ITSELF.
Only extract what is explicitly stated in the question.
Never guess or infer information not present in the question.

Fields you can extract:
- "source": paper title or journal name if explicitly mentioned
- "authors": author last names if explicitly mentioned
- "published_at": year if explicitly mentioned
- "arxiv_id": arxiv ID if explicitly mentioned
- "categories": research domain if explicitly mentioned (e.g. cs.CL, cs.AI)

Rules:
- Return ONLY a valid JSON object
- Use {{"$in": [...]}} for each field
- If nothing specific is mentioned in the question, return {{}}
- Never add fields that are not mentioned in the question
- Author names: extract last name only

Examples of how to extract:

Question: "What did Smith et al. say about neural networks?"
Answer: {{"authors": {{"$in": ["Smith"]}}}}

Question: "Summarize the paper titled Attention Is All You Need"
Answer: {{"source": {{"$in": ["Attention Is All You Need"]}}}}

Question: "What papers from 2023 discuss large language models?"
Answer: {{"published_at": {{"$in": ["2023"]}}}}

Question: "What did Johnson and Lee publish in 2022 about computer vision?"
Answer: {{"authors": {{"$in": ["Johnson", "Lee"]}}, "published_at": {{"$in": ["2022"]}}}}

Question: "Find the arxiv paper 2406.13213 about RAG"
Answer: {{"arxiv_id": {{"$in": ["2406.13213"]}}}}

Question: "What is retrieval augmented generation?"
Answer: {{}}

Question: "Explain how transformers work"
Answer: {{}}

Question: "What are the latest findings in NLP research?"
Answer: {{}}

Now extract from this question. Return ONLY the JSON, nothing else:
Question: {query}
Answer:"""


def extract_metadata_filter(query: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(query=query)
                }
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Clean up in case model adds extra text
        if "{" in raw and "}" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]

        result = json.loads(raw)
        return result

    except (json.JSONDecodeError, Exception):
        return {}