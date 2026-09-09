import csv

from chatbot.rag.pipeline.pipeline import RAGPipeline


QUESTIONS = [
    ("יש לי פצוע בן 19 שלא עובד, מה יכול להגיע לו?", "ASK_FOLLOWUP"),
    ("אלמנה בת 28 בפשיטת רגל", "ASK_FOLLOWUP"),
    ('משפחה שכולה דרגה ראשונה - פנתה אלינה עו"סים לעזרה ונעלמה?', "ASK_FOLLOWUP"),
    ("איזה סיוע כלכלי יוצא לי כאלמנה שבעלה נרצח?", "ANSWER"),
    ("הבן שלי נרצח - ויש לו רכב במימון מה עושים?", "ANSWER"),
    ("אני לא מצליחה לגשת לחשבון הבנק של בעלי המנוח", "ANSWER"),
    ("היה ירי ליד בית הספר של הבן שלי ומאז הוא מפחד לצאת החוצה - למי אני יכולה לפנות", "ANSWER"),
    ("מקרי סחיטה", "ASK_FOLLOWUP"),
    ("אנחנו רוצים לעבור דירה בגלל האיומים. יש מי שיכול לעזור בזה?", "ANSWER"),
    ("מאז הפציעה אני לא עובד, אני לא מצליח לשלם חובות ומשכנתאות", "ASK_FOLLOWUP"),
    ("אף אחד לא מסביר לנו מה קורה בתיק? למי אפשר לפנות?", "ASK_FOLLOWUP"),
    ('אני מתקשר למשטרה לבדוק מה קורה? אומרים שאני יכול להיכנס למנ"ע, מה זה? איך נכנסים?', "ANSWER"),

    # Negative controls
    ("איך מכינים פיצה?", "OUT_OF_SCOPE"),
    ("מה מזג האוויר מחר?", "OUT_OF_SCOPE"),
    ("איך מחליפים סיסמה באינסטגרם?", "OUT_OF_SCOPE"),
    ("מי זכה בליגת האלופות?", "OUT_OF_SCOPE"),
]


def main():
    print("Loading RAG pipeline...")
    pipeline = RAGPipeline.from_static_dir()

    rows = []

    for number, (query, expected) in enumerate(QUESTIONS, start=1):
        diagnostics = pipeline.retrieval_diagnostics(query)

        row = {
            "question_number": number,
            "query": query,
            "expected": expected,
            "raw_cosine": diagnostics["top_raw_cosine"],
            "adjusted_score": diagnostics["top_adjusted_score"],
            "second_adjusted_score": diagnostics["second_adjusted_score"],
            "adjusted_gap": diagnostics["adjusted_gap"],
            "top_bm25": diagnostics["top_bm25"],
        }

        rows.append(row)

        print(
            f"{number:02d} | {expected:12} | "
            f"cos={row['raw_cosine']:.4f} | "
            f"adj={row['adjusted_score']:.4f} | "
            f"gap={row['adjusted_gap']:.4f} | "
            f"bm25={row['top_bm25']:.4f} | "
            f"{query}"
        )

    output_file = "threshold_results.csv"

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Finished. Results saved to {output_file}")


if __name__ == "__main__":
    main()