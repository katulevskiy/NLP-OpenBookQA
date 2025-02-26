#!/usr/bin/env python3
import os
import json
import re
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file (ensure JWT_TOKEN is set)
load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

# API endpoint for DeepSeek
API_ENDPOINT = "http://localhost:3000/api/chat/completions"


def call_deepseek(prompt, model_name="deepseek-r1:32b"):
    """
    Sends a prompt to the DeepSeek API using the specified model and returns the answer text.
    """
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None

    data = response.json()
    try:
        answer_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print("Error parsing response:", e)
        return None
    return answer_text


def get_candidate_facts(question_text, facts, top_n=10):
    """
    Tokenizes the list of fact strings and the question text, then uses BM25 to retrieve
    the top_n candidate facts.
    """
    tokenized_facts = [fact.lower().split() for fact in facts]
    bm25 = BM25Okapi(tokenized_facts)
    query_tokens = question_text.lower().split()
    candidates = bm25.get_top_n(query_tokens, facts, n=top_n)
    return candidates


def rerank_facts_with_llm(question_text, candidate_facts):
    """
    Uses DeepSeek to rerank the candidate facts.
    Constructs a prompt that lists the candidate facts and asks DeepSeek to return the three most relevant facts.
    Expects DeepSeek's output to be a JSON array.
    """
    prompt = f"Question: {question_text}\n\nCandidate facts:\n"
    for i, fact in enumerate(candidate_facts, 1):
        prompt += f"{i}. {fact}\n"
    prompt += (
        "\nPlease select the three most relevant facts to help answer the question. "
        'Return your answer as a JSON array of the exact fact texts, e.g. ["fact 1", "fact 2", "fact 3"].'
    )
    response = call_deepseek(prompt, model_name="deepseek-r1:32b")
    if response is None:
        return None
    # Try to parse the response as JSON.
    try:
        selected_facts = json.loads(response)
        return selected_facts[:3]
    except json.JSONDecodeError:
        # Fallback: try to extract three facts by matching numbered lines.
        pattern = re.compile(r"\d+\.\s*(.*)")
        lines = response.strip().splitlines()
        facts_extracted = [
            pattern.match(line).group(1).strip()
            for line in lines
            if pattern.match(line)
        ]
        return facts_extracted[:3]


def process_question(question, facts):
    """
    Processes a single question:
      - Uses BM25 to retrieve candidate facts.
      - Uses DeepSeek to rerank and select the top 3 facts.
    Returns a tuple: (question_id, question_text, top_facts)
    """
    qid = question.get("id", "unknown")
    question_text = question["question"]["stem"]
    candidate_facts = get_candidate_facts(question_text, facts, top_n=10)
    top_facts = rerank_facts_with_llm(question_text, candidate_facts)
    return qid, question_text, top_facts


def main():
    # Path to openbook.txt (facts file)
    openbook_path = os.path.join(
        "OpenBookQA-V1-Sep2018", "Data", "Main", "openbook.txt"
    )
    if not os.path.exists(openbook_path):
        print(f"File not found: {openbook_path}")
        return

    with open(openbook_path, "r") as f:
        # Assume each nonempty line is a fact; remove surrounding quotes if present.
        facts = [line.strip().strip('"') for line in f if line.strip()]

    # Path to dev.jsonl containing questions (first 500 questions)
    dev_path = os.path.join("OpenBookQA-V1-Sep2018", "Data", "Main", "dev.jsonl")
    if not os.path.exists(dev_path):
        print(f"File not found: {dev_path}")
        return

    # Read the first 500 questions from dev.jsonl
    questions = []
    with open(dev_path, "r") as f:
        for line in islice(f, 500):
            if line.strip():
                questions.append(json.loads(line))

    # Dictionary to store results keyed by question id.
    results = {}

    print("Processing questions to retrieve top 3 relevant facts for each...")
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(process_question, question, facts): question
            for question in questions
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing questions"
        ):
            qid, question_text, top_facts = future.result()
            results[qid] = {"question": question_text, "selected_facts": top_facts}

    # Save the results to a JSON file.
    output_filename = "reranked_facts.json"
    with open(output_filename, "w") as out_file:
        json.dump(results, out_file, indent=2)
    print(f"\nReranked facts for {len(questions)} questions saved to {output_filename}")


if __name__ == "__main__":
    main()
