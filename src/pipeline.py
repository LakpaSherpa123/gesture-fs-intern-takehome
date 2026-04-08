"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""
import os
import argparse
from typing import Dict, List, Any  # For Type Hinting
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate

# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""

def print_response(result: dict) -> None:
    print(f"\nAnswer: {result['answer']}")
    print("\nSources used:")
    for i, source in enumerate(result["sources"], 1):
        # Clean up newlines and truncate for a readable terminal output
        snippet = source.replace('\n', ' ')[:140]
        print(f"{i}. {snippet}...")

def single_question_CLI(vector_store: Any, llm: Any) -> bool:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Gesture AI Support Bot")
    parser.add_argument("-query", type=str, help="Single query and exit")
    args: argparse.Namespace = parser.parse_args()
    
    if args.query:
        result: Dict[str, Any] = ask_question(vector_store, llm, args.query)
        print_response(result)
        return True
        
    return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm, question: str) -> dict[str, Any]:
    docs: List[Any] = vector_store.similarity_search(question, k=3)
    context:str = "\n\n".join([doc.page_content for doc in docs])
    final_prompt: str = PROMPT_TEMPLATE.format(context=context, question=question)
    result: List[Dict[str, str]] = llm(final_prompt)
    answer: str = result[0]["generated_text"]
    return {
        "sources": [doc.page_content for doc in docs],
        "answer": answer,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")

    try:
        vector_store = build_knowledge_base(data_dir)
        llm = get_llm()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    if(single_question_CLI(vector_store, llm)):
        return

    print("(Type 'quit' or 'exit' to end the session)")
    while True:
        query: str = input("\nYour Question: ").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        try:
            result = ask_question(vector_store, llm, query)
            print_response(result)       
        except Exception as e:
            print(f"An error occurred while generating an answer: {e}")
            

if __name__ == "__main__":
    main()