
# -*- coding: utf-8 -*-
"""Read-only Neuro Graph retrieval for Kandy, Ella and Sigiriya."""

from __future__ import annotations

import os
import re
import argparse
from pathlib import Path

from dotenv import load_dotenv
import truststore
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent

# Destination names and famous place names
ALIASES = {
    "Kandy": (
        "මහනුවර",
        "නුවර",
        "kandy",
        "දළදා මාලිගාව",
        "දළදා මාලිගාවේ",
        "ශ්‍රී දළදා",
        "දන්ත ධාතුව",
        "temple of the tooth",
        "tooth relic",
    ),

    "Ella": (
        "ඇල්ල",
        "රාවණා ඇල්ල",
        "රාවණා ඇල්ලේ",
        "ella",
        "ravana falls",
        "ravana waterfall",
        "nine arch bridge",
        "ආරුක්කු නවයේ පාලම",
    ),

    "Sigiriya": (
        "සීගිරිය",
        "සීගිරි",
        "sigiriya",
        "lion rock",
        "සීගිරි බිතුසිතුවම්",
    ),
}

STOPWORDS = {
    "කුමක්ද", "කොහේද", "කොපමණද",
    "කියන්න", "පිළිබඳ", "ගැන", "සහ",
    "විසින්", "කරන", "ලද්දේ", "ඇත්තේ",
    "ඇති", "එහි", "ඒ", "ද", "වේ",
    "the", "what", "where", "who",
    "was", "is", "of", "in",
}


# -------------------------------------------------
# 1. Identify destination
# -------------------------------------------------

def identify_destination(question: str):
    q = question.casefold()

    found = [
        (len(alias), destination)
        for destination, aliases in ALIASES.items()
        for alias in aliases
        if alias.casefold() in q
    ]

    return max(found)[1] if found else None


# -------------------------------------------------
# 2. Extract question keywords
# -------------------------------------------------

def tokens(text):
    return {
        word
        for word in re.findall(
            r"[\u0D80-\u0DFF]+|[a-zA-Z]{3,}",
            (text or "").casefold()
        )
        if len(word) > 2 and word not in STOPWORDS
    }


# -------------------------------------------------
# 3. Calculate relevance
# -------------------------------------------------

def relevance(question, text):
    words = tokens(question)

    if not words:
        return 0

    return len(words & tokens(text)) / len(words)


# -------------------------------------------------
# 4. Connect to Neuro Graph
# -------------------------------------------------

def get_driver():
    # Use Windows certificate trust store
    truststore.inject_into_ssl()

    load_dotenv(ROOT / ".env", override=True)

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all((uri, username, password)):
        raise ValueError(
            "Missing graph credentials in .env"
        )

    return GraphDatabase.driver(
        uri,
        auth=(username, password),
        connection_timeout=30,
        connection_acquisition_timeout=60,
        max_transaction_retry_time=30,
        liveness_check_timeout=10
    )


# -------------------------------------------------
# 5. Retrieve relevant Neuro Graph context
# -------------------------------------------------

def retrieve_graph_context_from_question(
    destination,
    question,
    limit=5
):
    """
    Retrieve tourism information from existing
    Destination, Place and Section nodes.

    Read-only: no database changes.
    """

    driver = get_driver()

    try:
        driver.verify_connectivity()

        database = os.getenv(
            "NEO4J_DATABASE", "neo4j"
        )

        with driver.session(database=database) as session:

            graph_query = """
            MATCH (d:Destination {name_en:$destination})
                  -[:HAS_SECTION]->(s:Section)

            OPTIONAL MATCH (p:Place)-[:SUPPORTED_BY]->(s)
            WHERE EXISTS {
                MATCH (d)-[:HAS_PLACE]->(p)
            }

            RETURN
                d.name_en AS destination,
                s.id AS section_id,
                properties(s) AS section,
                collect(DISTINCT p.name_si) AS places
            """

            rows = [
                dict(record)
                for record in session.run(
                    graph_query,
                    destination=destination
                )
            ]

    finally:
        driver.close()

    # Rank sections by question relevance
    scored = []

    for row in rows:

        section = row["section"] or {}

        content = str(
            section.get("text_si")
            or section.get("content_si")
            or section.get("content")
            or ""  
        ).strip()
            

        place_names = ", ".join(
            place
            for place in row["places"]
            if place
        )

        evidence = (
            f"Place: {place_names}\n{content}"
            if place_names
            else content
        )

        score = relevance(question, evidence)

        # Give priority to exact place-name matches
        if any(
            place and place in question
            for place in row["places"]
        ):
            score += 1.0

        scored.append((
            score,
            row["section_id"] or "",
            evidence
        ))

    scored.sort(
        key=lambda item: (-item[0], item[1])
    )

    return [
        {
            "section_id": section_id,
            "score": round(score, 3),
            "context": evidence
        }
        for score, section_id, evidence
        in scored[:limit]
        if evidence.strip()
    ]


# -------------------------------------------------
# 6. Validate retrieved context
# -------------------------------------------------

def validate_retrieved_context(results):
    """
    Check whether retrieved context contains
    usable text.

    This does not guarantee factual accuracy.
    """

    return bool(
        results
        and any(
            result["context"].strip()
            for result in results
        )
    )


# -------------------------------------------------
# 7. Main testing function
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        required=True,
        help="Sinhala tourism question"
    )

    args = parser.parse_args()

    destination = identify_destination(
        args.question
    )

    print("Question:", args.question)
    print("Destination:", destination)

    if not destination:
        print(
            "Destination not identified. "
            "Please mention Kandy, Ella or Sigiriya "
            "or a known place name."
        )
        return

    results = retrieve_graph_context_from_question(
        destination,
        args.question
    )

    print(
        "Context Retrieved:",
        validate_retrieved_context(results)
    )

    for index, item in enumerate(results, 1):

        print(
            f"\n--- Source {index}: "
            f"{item['section_id']} "
            f"(score {item['score']}) ---"
        )

        print(item["context"][:1800])

    if not results:
        print(
            "No graph context found. "
            "Do not generate an unsupported answer."
        )


if __name__ == "__main__":
    main()
