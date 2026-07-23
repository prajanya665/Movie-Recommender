"""
cli.py

The user-facing interface for the hybrid recommender system.

Architecture (this is exactly what should go in your Part C architecture
diagram):

    User Input (genre, mood, era, runtime limit)
            |
            v
    Rule Engine (rule_engine.py)  --  hard filters, knowledge-based
            |
            v
    Probability Engine (probability_engine.py)  --  soft scoring, uncertainty
            |
            v
    Ranked, explained recommendations (printed to console)

Run with:  python3 cli.py
"""

import csv

from rule_engine import apply_rules, explain_rule_pass
from probability_engine import build_likelihood_tables, score_movies


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ask_preferences(available_genres, available_eras):
    print("\n--- Tell me what you're in the mood for ---")
    print(f"Available genres: {', '.join(available_genres)}")
    genre = input("Genre (press Enter to skip): ").strip()
    genre = genre if genre in available_genres else None

    print(f"Available eras: {', '.join(available_eras)}")
    era = input("Era (press Enter to skip): ").strip()
    era = era if era in available_eras else None

    runtime_input = input("Max runtime in minutes (press Enter to skip): ").strip()
    max_runtime = int(runtime_input) if runtime_input.isdigit() else None

    mood = input("Mood -- 'light', 'intense', or Enter for neutral: ").strip().lower()
    if mood not in ("light", "intense"):
        mood = "neutral"

    return {"genre": genre, "era": era, "max_runtime": max_runtime}, mood


def print_recommendations(scored, preferences, top_n=5):
    print(f"\n=== Top {top_n} Recommendations ===\n")
    if not scored:
        print("No movies matched your hard constraints. Try loosening genre/era/runtime.")
        return

    for rank, (movie, score, breakdown) in enumerate(scored[:top_n], start=1):
        rule_reason = explain_rule_pass(movie, preferences)
        print(f"{rank}. {movie['title']}  ({movie['genre']}, {movie['era']}, "
              f"{movie['runtime_minutes']} min, popularity: {movie['popularity']})")
        print(f"   Confidence score: {breakdown['final_score']}")
        print(f"   Rule stage: {rule_reason}")
        print(f"   Probability breakdown: P(genre)={breakdown['P(like|genre)']}, "
              f"P(era)={breakdown['P(like|era)']}, P(popularity)={breakdown['P(like|popularity)']}, "
              f"mood_weight={breakdown['mood_weight']}")
        print()


def main():
    movies = load_csv("data/movies.csv")
    ratings = load_csv("data/ratings.csv")

    available_genres = sorted(set(m["genre"] for m in movies))
    available_eras = sorted(set(m["era"] for m in movies))

    print("=== Hybrid Knowledge-Based + Probabilistic Movie Recommender ===")

    # Build the probability lookup tables ONCE from the full ratings history.
    # This is the "learning from data" step -- it only needs to happen once,
    # not per-user, since it represents general crowd-level tendencies.
    likelihood_tables = build_likelihood_tables(movies, ratings)

    preferences, mood = ask_preferences(available_genres, available_eras)

    # Stage 1: knowledge-based hard filtering
    candidates = apply_rules(movies, preferences)

    # Stage 2: probabilistic reasoning under uncertainty
    scored = score_movies(candidates, likelihood_tables, mood=mood)

    print_recommendations(scored, preferences, top_n=5)


if __name__ == "__main__":
    main()
