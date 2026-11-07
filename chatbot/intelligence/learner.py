"""
Supervised question-pattern learner.

Training data: official Annexure A questions + curated paraphrases + reviewed feedback.
Features: word and character TF-IDF.
Model: LogisticRegression for intent classification, cosine similarity for nearest
question-template retrieval.

This is a learned model, not a chain of phrase-specific if/else rules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from .catalog import QUESTION_CATALOG, INTENT_DESCRIPTIONS

DEFAULT_FEEDBACK_PATH = Path(__file__).with_name("feedback.jsonl")

# Ranking/leaderboard examples are supplemental product intents. They do not add
# to or modify the 40 official Annexure A catalogue questions.
RANKING_TRAINING_EXAMPLES = [
    "Who is ranked first in Engineering in 2025?",
    "Who holds the number one position in Overall for 2025?",
    "Show the top 10 institutions in Engineering in 2025.",
    "Give me the top ten colleges in Pharmacy for 2024.",
    "List the best ranked institutions in Management in 2025.",
    "Show institutions ranked in the top 20 for Engineering in 2023.",
    "What is the NIRF rank of IIT Dharwad in Engineering in 2025?",
    "Find the institution at rank 25 in Engineering in 2025.",
    "Display a leaderboard for the top 15 Overall institutions in 2025.",
    "Which university has the highest NIRF rank in 2025?",
]

class QuestionLearner:
    def __init__(self, feedback_path: str | Path | None = None,
                 confidence_threshold: float = 0.30,
                 similarity_threshold: float = 0.12):
        self.feedback_path = Path(feedback_path) if feedback_path else DEFAULT_FEEDBACK_PATH
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                     strip_accents="unicode", lowercase=True,
                                     min_df=1, max_features=30000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     sublinear_tf=True, lowercase=True,
                                     min_df=1, max_features=40000)),
        ])
        self.classifier = LogisticRegression(max_iter=2000, class_weight="balanced",
                                             C=2.0, random_state=42)
        self._train()

    @staticmethod
    def _normalise(text: str) -> str:
        text = str(text or "").lower().replace("–", "-").replace("—", "-")
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s%.-]", " ", text)).strip()

    def _training_rows(self):
        rows = []
        for item in QUESTION_CATALOG:
            examples = [item["question"], *item.get("examples", [])]
            for example in examples:
                rows.append({
                    "text": self._normalise(example),
                    "intent": item["intent"],
                    "question_id": item["id"],
                    "chart": item["chart"],
                    "operation": item["operation"],
                    "required_slots": item["required_slots"],
                    "source": "catalog",
                })
        for example in RANKING_TRAINING_EXAMPLES:
            rows.append({
                "text": self._normalise(example),
                "intent": "ranking",
                "question_id": "RANK01",
                "chart": "ranking",
                "operation": "ranking",
                "required_slots": ["category", "year"],
                "source": "ranking_examples",
            })
        if self.feedback_path.exists():
            for line in self.feedback_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("question") and obj.get("corrected_intent"):
                        rows.append({
                            "text": self._normalise(obj["question"]),
                            "intent": obj["corrected_intent"],
                            "question_id": obj.get("question_id"),
                            "chart": obj.get("chart"),
                            "operation": obj.get("operation"),
                            "required_slots": obj.get("required_slots", []),
                            "source": "feedback",
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
        return rows

    def _train(self):
        self.rows = self._training_rows()
        self.texts = [r["text"] for r in self.rows]
        self.labels = [r["intent"] for r in self.rows]
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.classifier.fit(self.matrix, self.labels)

    def retrain(self):
        """Reload feedback and retrain the model."""
        self._train()
        return {"training_examples": len(self.rows),
                "intents": sorted(set(self.labels))}

    @staticmethod
    def _extract_slots(question: str) -> dict[str, Any]:
        q = question.lower()
        years = sorted({int(y) for y in re.findall(r"\b(?:201[7-9]|202[0-5])\b", q)})
        slots: dict[str, Any] = {}
        if len(years) == 1:
            slots["year"] = years[0]
        elif len(years) >= 2:
            slots["start_year"], slots["end_year"] = years[0], years[-1]

        rank_match = re.search(r"\btop\s*(\d{1,3})\b", q)
        if rank_match:
            slots["rank_limit"] = int(rank_match.group(1))
        exact = re.search(r"\b(?:rank|position)\s*(?:number\s*)?(?:#\s*)?(\d{1,3})\b", q)
        if exact:
            slots["exact_rank"] = int(exact.group(1))

        metric_aliases = {
            "TLR": ["tlr", "teaching learning resources", "teaching"],
            "RPC": ["rpc", "research and professional practice", "research"],
            "GO": ["go", "graduation outcomes"],
            "OI": ["oi", "outreach and inclusivity"],
            "PR": ["pr", "perception"],
            "QNR": ["qnr", "quantitative research"],
            "QLR": ["qlr", "qualitative research"],
            "SFC": ["sfc", "student and faculty contribution"],
            "score": ["overall score", "total score", "nirf score"],
            "rank": ["rank", "ranking"],
        }
        # Extract metrics in the order they appear in the question.
        metric_matches = []
        for metric, aliases in metric_aliases.items():
            for alias in aliases:
                pattern = (
                    r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
                )
                for match in re.finditer(
                    pattern,
                    question,
                    flags=re.IGNORECASE,
                ):
                    metric_matches.append(
                        (match.start(), -(match.end() - match.start()), metric)
                    )
        metric_matches.sort()
        found = []
        seen_metrics = set()
        for _, _, metric in metric_matches:
            if metric not in seen_metrics:
                found.append(metric)
                seen_metrics.add(metric)
        if found:
            slots["metrics"] = found

        state_aliases = {
            "Karnataka": ["karnataka"],
            "Tamil Nadu": ["tamil nadu", "tamilnadu"],
            "Maharashtra": ["maharashtra"],
            "Uttar Pradesh": ["uttar pradesh"],
            "Delhi": ["delhi"],
            "Kerala": ["kerala"],
            "Telangana": ["telangana"],
            "Andhra Pradesh": ["andhra pradesh"],
        }
        states = [state for state, aliases in state_aliases.items()
                  if any(alias in q for alias in aliases)]
        if states:
            slots["states"] = states
        return slots

    def predict(self, question: str) -> dict[str, Any]:
        if not str(question or "").strip():
            return {"status": "needs_clarification", "reason": "empty_question"}
        clean = self._normalise(question)
        query_matrix = self.vectorizer.transform([clean])
        probabilities = self.classifier.predict_proba(query_matrix)[0]
        best_idx = int(np.argmax(probabilities))
        predicted_intent = self.classifier.classes_[best_idx]
        confidence = float(probabilities[best_idx])

        similarities = cosine_similarity(query_matrix, self.matrix)[0]
        nearest_idx = int(np.argmax(similarities))
        similarity = float(similarities[nearest_idx])
        nearest = self.rows[nearest_idx]

        # Resolve disagreements between the classifier and nearest reviewed template.
        classifier_intent = str(predicted_intent)
        nearest_intent = str(nearest["intent"])
        intent = classifier_intent
        accepted = (
            confidence >= self.confidence_threshold
            and similarity >= self.similarity_threshold
        )
        intent_source = "classifier"

        # When classifier confidence is weak but a sufficiently similar
        # reviewed template supports another intent, resolve the conflict.
        if (
            nearest_intent != classifier_intent
            and confidence < 0.50
            and similarity >= max(0.40, self.similarity_threshold)
        ):
            intent = nearest_intent
            accepted = True
            intent_source = "nearest_template_conflict_resolution"

        status = "success" if accepted else "needs_clarification"
        intent = intent if accepted else None
        matched = (
            nearest
            if accepted
            and nearest_intent == intent
            and similarity >= self.similarity_threshold
            else None
        )
        question_id = matched["question_id"] if matched else None
        return {
            "status": status,
            "question": question,
            "intent": intent,
            "intent_source": intent_source if accepted else "uncertain",
            "intent_description": INTENT_DESCRIPTIONS.get(intent) if intent else None,
            "question_id": question_id,
            "nearest_question_id": nearest["question_id"],
            "nearest_question": nearest["text"],
            "confidence": round(confidence, 4),
            "similarity": round(similarity, 4),
            "operation": matched["operation"] if matched else None,
            "catalog_chart": matched["chart"] if matched else None,
            "required_slots": matched["required_slots"] if matched else [],
            "slots": self._extract_slots(question),
            "training_examples": len(self.rows),
            "clarification": None if accepted else
                "I am not confident about the analytical intent. Please specify whether you want a trend, comparison, distribution, correlation, geographic analysis, transition, or outlier analysis.",
        }

    def learn(self, question: str, corrected_intent: str,
              question_id: str | None = None, chart: str | None = None,
              operation: str | None = None, required_slots: list[str] | None = None):
        """Add a reviewed correction. Retraining is explicit, not automatic."""
        allowed = set(INTENT_DESCRIPTIONS)
        if corrected_intent not in allowed:
            raise ValueError(f"Unknown intent: {corrected_intent}")
        record = {
            "question": question,
            "corrected_intent": corrected_intent,
            "question_id": question_id,
            "chart": chart,
            "operation": operation,
            "required_slots": required_slots or [],
        }
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.retrain()
