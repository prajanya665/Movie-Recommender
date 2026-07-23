"""
rule_engine.py

The KNOWLEDGE-BASED component of the recommender.

This represents domain knowledge as explicit IF-THEN rules, rather than
learning them from data. This is the "expert system" style part of the
project: a human (you) has encoded reasoning like:

    IF the user gave a genre preference AND the movie doesn't match it
    THEN exclude the movie.

Keeping this separate from the probability engine (probability_engine.py)
is deliberate: it keeps the "hard constraints" (genre, runtime) distinct
from the "soft, uncertain scoring" (probabilistic reasoning), which is
exactly the kind of hybrid architecture the assignment brief asks for.
"""


def apply_rules(movies, preferences):
    """
    Filters the full movie list down to only movies that satisfy the user's
    hard constraints.

    movies: list of dicts, each representing one movie (from movies.csv)
    preferences: dict with keys like:
        - "genre": str or None      e.g. "Action"
        - "max_runtime": int or None  e.g. 120
        - "era": str or None        e.g. "1990s"

    Returns: filtered list of movies that pass every rule.
    """
    candidates = []

    for movie in movies:
        # Rule 1: if the user specified a genre, the movie must match it
        if preferences.get("genre") and movie["genre"] != preferences["genre"]:
            continue

        # Rule 2: if the user specified a max runtime, respect it
        if preferences.get("max_runtime"):
            if int(movie["runtime_minutes"]) > preferences["max_runtime"]:
                continue

        # Rule 3: if the user specified an era, the movie must match it
        if preferences.get("era") and movie["era"] != preferences["era"]:
            continue

        candidates.append(movie)

    return candidates


def explain_rule_pass(movie, preferences):
    """
    Produces a short, human-readable explanation of why a movie passed the
    rule stage. Used later by the CLI to make recommendations explainable.
    """
    reasons = []
    if preferences.get("genre"):
        reasons.append(f"matches requested genre '{preferences['genre']}'")
    if preferences.get("era"):
        reasons.append(f"matches requested era '{preferences['era']}'")
    if preferences.get("max_runtime"):
        reasons.append(f"runtime {movie['runtime_minutes']} min is within your limit")
    if not reasons:
        reasons.append("no hard constraints given, so passed by default")
    return "; ".join(reasons)
