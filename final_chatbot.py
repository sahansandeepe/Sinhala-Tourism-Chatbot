
import argparse

from tourism_graph_retrieval import (
    identify_destination,
    retrieve_graph_context_from_question,
    validate_retrieved_context
)

from sinllama_integration import generate_sinllama_response


def answer_question(question):
    destination = identify_destination(question)

    print("Question:", question)
    print("Destination:", destination)

    if not destination:
        print("Destination not identified.")
        return

    results = retrieve_graph_context_from_question(
        destination, question, limit=3
    )

    if not validate_retrieved_context(results):
        print("ප්‍රමාණවත් තොරතුරු නොමැත.")
        return

    if results[0]["score"] <= 0:
        print("ප්‍රමාණවත් තොරතුරු නොමැත.")
        return

    best_score = results[0]["score"]
    selected = [
        r for r in results
        if r["score"] == best_score
    ][:2]

    context = "\n\n".join(
        r["context"][:1800] for r in selected
    )

    print(
        "Sources:",
        ", ".join(r["section_id"] for r in selected)
    )

    answer = generate_sinllama_response(
        question, context
    )

    print("Final Sinhala Answer:", answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    answer_question(args.question)
