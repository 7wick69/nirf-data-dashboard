"""Train and inspect the NIRF question-pattern model."""
from chatbot.intelligence import QuestionLearner

if __name__ == "__main__":
    learner = QuestionLearner()
    print("Training examples:", len(learner.rows))
    print("Intents:", sorted(set(learner.labels)))
    while True:
        try:
            q = input("\nQuestion (blank to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        result = learner.predict(q)
        for key, value in result.items():
            print(f"{key}: {value}")
