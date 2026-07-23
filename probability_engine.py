"""
probability_engine.py

The PROBABILISTIC REASONING component of the recommender.

This is where "reasoning under uncertainty" happens (the module requirement).
Instead of hard rules, we estimate probabilities from historical rating data:

    P(a movie is "liked" | its genre)
    P(a movie is "liked" | its era)
    P(a movie is "liked" | its popularity tier)

"Liked" is defined as rating >= 4 (out of 5).

We then combine these under a NAIVE BAYES style independence assumption
(i.e., we assume genre, era, and popularity contribute independently to
the chance of being liked -- a simplification, but a standard and
defensible one, and worth mentioning as a limitation in Part D):

    Score(movie) = P(like|genre) * P(like|era) * P(like|popularity) * mood_weight

We also fold in the user's stated MOOD as a soft weighting factor. This
represents an additional, human-authored piece of uncertain reasoning:
mood is fuzzy/subjective, so instead of a hard filter, it nudges the score
up or down depending on how well the movie's genre fits the requested mood.
"""

from collections import defaultdict

LIKE_THRESHOLD = 4  # rating >= 4 counts as "liked"

# Soft mapping of mood -> genre affinity weights.
# This encodes designer knowledge about which genres suit which moods,
# used as a soft (probabilistic-style) weight, not a hard rule.
MOOD_GENRE_WEIGHTS = {
    "light": {"Comedy": 1.3, "Romance": 1.2, "Drama": 1.0, "SciFi": 0.9, "Action": 0.8, "Horror": 0.6},
    "intense": {"Horror": 1.3, "Action": 1.3, "SciFi": 1.1, "Drama": 1.0, "Romance": 0.8, "Comedy": 0.7},
    "neutral": {g: 1.0 for g in ["Comedy", "Romance", "Drama", "SciFi", "Action", "Horror"]},
}


def _laplace_smoothed_rate(liked_count, total_count, smoothing=1, prior=0.5):
    """
    Estimates a probability from counts, using Laplace (additive) smoothing.
    This avoids the problem of a probability being exactly 0 or 1 just
    because we've seen very few examples -- an important detail for a
    small dataset like ours. `prior` softly pulls sparse estimates toward
    a neutral 0.5 rather than an extreme value.
    """
    return (liked_count + smoothing * prior) / (total_count + smoothing)


def build_likelihood_tables(movies, ratings):
    """
    Scans the ratings data and builds probability lookup tables:
        P(liked | genre), P(liked | era), P(liked | popularity)

    Returns a dict of three dicts, e.g.:
        {
          "genre":      {"Action": 0.62, "Comedy": 0.48, ...},
          "era":        {"1990s": 0.55, ...},
          "popularity": {"high": 0.70, "medium": 0.5, "low": 0.4},
        }
    """
    movie_lookup = {m["movie_id"]: m for m in movies}

    counts = {
        "genre": defaultdict(lambda: [0, 0]),       # attribute -> [liked_count, total_count]
        "era": defaultdict(lambda: [0, 0]),
        "popularity": defaultdict(lambda: [0, 0]),
    }

    for r in ratings:
        movie = movie_lookup.get(r["movie_id"])
        if not movie:
            continue
        liked = 1 if int(r["rating"]) >= LIKE_THRESHOLD else 0

        counts["genre"][movie["genre"]][0] += liked
        counts["genre"][movie["genre"]][1] += 1

        counts["era"][movie["era"]][0] += liked
        counts["era"][movie["era"]][1] += 1

        counts["popularity"][movie["popularity"]][0] += liked
        counts["popularity"][movie["popularity"]][1] += 1

    tables = {}
    for attribute, attr_counts in counts.items():
        tables[attribute] = {
            key: round(_laplace_smoothed_rate(liked, total), 3)
            for key, (liked, total) in attr_counts.items()
        }

    return tables


def score_movies(candidate_movies, likelihood_tables, mood="neutral"):
    """
    Scores each candidate movie using the probabilistic formula described
    at the top of this file, and returns a list of
    (movie, score, breakdown_dict) tuples sorted by score descending.
    """
    mood_weights = MOOD_GENRE_WEIGHTS.get(mood, MOOD_GENRE_WEIGHTS["neutral"])

    scored = []
    for movie in candidate_movies:
        p_genre = likelihood_tables["genre"].get(movie["genre"], 0.5)
        p_era = likelihood_tables["era"].get(movie["era"], 0.5)
        p_pop = likelihood_tables["popularity"].get(movie["popularity"], 0.5)
        mood_weight = mood_weights.get(movie["genre"], 1.0)

        score = p_genre * p_era * p_pop * mood_weight

        breakdown = {
            "P(like|genre)": p_genre,
            "P(like|era)": p_era,
            "P(like|popularity)": p_pop,
            "mood_weight": mood_weight,
            "final_score": round(score, 4),
        }
        scored.append((movie, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
