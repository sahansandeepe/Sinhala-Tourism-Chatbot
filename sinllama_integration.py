
# -*- coding: utf-8 -*-
"""Sinhala Tourism Chatbot: SinLlama + Neuro Graph."""

import re
from pathlib import Path
from llama_cpp import Llama

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "SinLlama.Q2_K.gguf"

_llm = None
NO_INFO = "ප්‍රමාණවත් තොරතුරු නොමැත."


def get_model():
    """Load SinLlama only once."""
    global _llm

    if _llm is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"SinLlama model not found: {MODEL_PATH}"
            )

        print("Loading SinLlama model...")

        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            chat_format="llama-3",
            verbose=False
        )

    return _llm


def clean_context(context):
    """Remove graph metadata and normalize whitespace."""

    context = re.sub(
        r"(?m)^Place:\s*.*$",
        "",
        context
    )

    return re.sub(r"\s+", " ", context).strip()


def split_sentences(text):
    """Split Sinhala source context into sentences."""

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?।])\s+|\n+",
            text
        )
        if sentence.strip()
    ]


def identify_question_intent(question):
    """Identify the Sinhala question type."""

    if "උස" in question and (
        "කොපමණ" in question or "කීයද" in question
    ):
        return "height"

    if "ඇයි" in question or "හේතුව" in question:
        return "why"

    if any(word in question for word in [
        "රජු", "කවුද", "කවුරුන්ද"
    ]):
        return "who"

    if any(word in question for word in [
        "කවදාද", "වර්ෂයේද", "කවර වසරේද"
    ]):
        return "when"

    if "කොහේද" in question or "පිහිටා" in question:
        return "where"

    if "යනු කුමක්ද" in question:
        return "definition"

    return "general"


def extract_grounded_answer(question, context):
    """Select relevant Neuro Graph facts based on question intent."""

    text = clean_context(context)
    sentences = split_sentences(text)

    if not sentences:
        return NO_INFO

    intent = identify_question_intent(question)

    question_words = set(
        re.findall(r"[\u0D80-\u0DFF]+", question)
    )

    stopwords = {
        "කුමක්ද", "කොපමණද", "කොහේද",
        "කියන්න", "ඇති", "කරන", "ලද්දේ",
        "විසින්", "තැන්පත්", "ඇත්තේ",
        "වන්නේ", "ඇයි", "යනු",
        "කවුද", "කවදාද"
    }

    question_words -= stopwords

    scored = []

    for i, sentence in enumerate(sentences):

        words = set(
            re.findall(r"[\u0D80-\u0DFF]+", sentence)
        )

        score = len(question_words & words)

        # HEIGHT: prioritize height over distance
        if intent == "height":
            if "උස" in sentence and "මීටර්" in sentence:
                score += 10

            if "කිලෝමීටර්" in sentence:
                score -= 10

        # WHY: prioritize sentences explaining reasons
        elif intent == "why":
            if any(
                word in sentence
                for word in ["නිසා", "හේතුවෙන්", "බැවින්"]
            ):
                score += 8

        # WHO: prioritize sentences identifying people
        elif intent == "who":
            if "රජු" in sentence:
                score += 8

            if any(
                word in sentence
                for word in [
                    "ඉදිකළ", "ඉදිකරන",
                    "නිර්මාණය"
                ]
            ):
                score += 3

        # WHEN: prioritize historical dates
        elif intent == "when":
            if re.search(
                r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
                sentence
            ):
                score += 8

        # WHERE: prioritize location information
        elif intent == "where":
            if any(
                word in sentence
                for word in [
                    "පිහිටි", "පිහිටා",
                    "ආසන්නයේ", "දුරින්"
                ]
            ):
                score += 8

        # DEFINITION: prioritize descriptions
        elif intent == "definition":
            if any(
                phrase in sentence
                for phrase in [
                    "යනු", "ලෙස",
                    "ඇති වන පරිදි",
                    "සමන්විත", "හැඳින්වේ"
                ]
            ):
                score += 6

            if "වැදගත් ඓතිහාසික අංගයකි" in sentence:
                score -= 5

        scored.append((score, i, sentence))

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_index, best_sentence = scored[0]

    if best_score <= 0:
        return NO_INFO

    # WHY: include the preceding reason sentence
    if intent == "why":
        if best_index > 0:
            previous = sentences[best_index - 1]
            return previous + " " + best_sentence

    # WHO: extract complete Sinhala name
    # Fixes truncated names such as "යප රජු"
    if intent == "who":
        match = re.search(
            r"([^\s,.!?।]+)\s+රජු",
            best_sentence
        )

        if match:
            return match.group(1) + " රජු"

        return best_sentence

    # DEFINITION: include a supporting sentence
    if intent == "definition":
        if best_index + 1 < len(sentences):
            return (
                best_sentence + " " +
                sentences[best_index + 1]
            )

    return best_sentence


def is_valid_answer(answer, question, context):
    """Reject obviously malformed SinLlama responses."""

    if not answer or not answer.strip():
        return False

    answer = answer.strip()

    if len(answer) > 250:
        return False

    if question in answer:
        return False

    invalid_markers = [
        "ප්‍රශ්නය:",
        "ප්‍රශ්නය :",
        "මූලාශ්‍ර",
        "තොරතුරු:",
        "Place:",
        "මල:"
    ]

    if any(
        marker in answer
        for marker in invalid_markers
    ):
        return False

    if "?" in answer:
        return False

    return True


def generate_sinllama_response(question, context):
    """Generate Sinhala answers using SinLlama and graph context."""

    if not context or not context.strip():
        return NO_INFO

    cleaned = clean_context(context)

    # Retrieve source-grounded candidate answer
    grounded_answer = extract_grounded_answer(
        question,
        cleaned
    )

    llm = get_model()

    result = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "ඔබ ශ්‍රී ලංකා සංචාරක සහායකයෙකි. "
                    "ලබා දී ඇති මූලාශ්‍රයෙන් පමණක් "
                    "පිළිතුරු දෙන්න. "
                    "නිවැරදි සහ ස්වභාවික සිංහල "
                    "භාෂාව භාවිතා කරන්න. "
                    "ප්‍රශ්නයට සෘජු පිළිතුරක් දෙන්න. "
                    "කෙටි වාක්‍ය එකක් හෝ දෙකක් ලියන්න. "
                    "ප්‍රශ්නය නැවත ලියන්න එපා. "
                    "තොරතුරු නොමැති නම් "
                    "'ප්‍රමාණවත් තොරතුරු නොමැත.' "
                    "ලෙස පිළිතුරු දෙන්න."
                )
            },
            {
                "role": "user",
                "content": (
                    f"මූලාශ්‍රය:\n{cleaned[:1200]}\n\n"
                    f"ප්‍රශ්නය: {question}\n\n"
                    "පිළිතුර:"
                )
            }
        ],
        max_tokens=100,
        temperature=0.0,
        repeat_penalty=1.2
    )

    answer = (
        result["choices"][0]["message"]["content"]
        or ""
    ).strip()

    # Accept model output if formatting is valid
    if is_valid_answer(answer, question, cleaned):
        return answer

    # Use Neuro Graph fallback
    print("Using grounded Neuro Graph fallback...")

    return grounded_answer
