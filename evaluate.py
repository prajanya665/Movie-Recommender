"""
evaluate.py

Part D: Evaluation and Experimental Results.

METHODOLOGY
-----------
For each simulated user:
  1. Split their ratings into TRAIN (80%) and TEST (20%).
  2. Infer their "stated preferences" from the TRAIN ratings only
     (their favourite genre / era = the one with the highest average
     rating in their training data). This simulates a user telling the
     system what they like.
  3. Rebuild the probability engine's likelihood tables using only the
     TRAIN ratings of ALL users (never the held-out test ratings --
     this avoids data leakage, i.e. "cheating" by learning from data
     the system is later evaluated on).
  4. Generate top-5 recommendations under each configuration.
  5. Check how many of those 5 recommended movies appear in the user's
     TEST set with a rating >= 4 ("liked"). This is PRECISION@5, a
     standard recommender systems metric: of the movies we recommended,
     what fraction did the user actually like?

We run this for:
  - CONFIG 1: Hybrid system, hard-filtered by genre only
  - CONFIG 2: Hybrid system, hard-filtered by genre + era
  - CONFIG 3: Rules only, no probabilistic ranking (just genre+era filter,
              sorted by popularity tier) -- shows what the probability
              engine actually adds
  - BASELINE: Non-personalized "most popular movies" (baseline.py)

Run with:  python3 evaluate.py
"""

import csv
import random
from collections import defaultdict

from rule_engine import apply_rules
from probability_engine import build_likelihood_tables, score_movies
from baseline import most_popular_recommendations

random.seed(7)
LIKE_THRESHOLD = 4
TOP_N = 5
TEST_FRACTION = 0.2


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_train_test(ratings):
    """Splits each user's ratings into train/test sets, per-user."""
    by_user = defaultdict(list)
    for r in ratings:
        by_user[r["user_id"]].append(r)

    train, test = [], []
    for user_id, user_ratings in by_user.items():
        random.shuffle(user_ratings)
        n_test = max(1, int(len(user_ratings) * TEST_FRACTION))
        test.extend(user_ratings[:n_test])
        train.extend(user_ratings[n_test:])
    return train, test


def infer_preferences(user_ratings, movies):
    """
    Simulates a user 'stating' their favourite genre and era, inferred
    from their own training ratings (highest average rating).
    """
    movie_lookup = {m["movie_id"]: m for m in movies}
    genre_scores = defaultdict(list)
    era_scores = defaultdict(list)

    for r in user_ratings:
        movie = movie_lookup.get(r["movie_id"])
        if not movie:
            continue
        genre_scores[movie["genre"]].append(int(r["rating"]))
        era_scores[movie["era"]].append(int(r["rating"]))

    if not genre_scores:
        return None, None

    fav_genre = max(genre_scores, key=lambda g: sum(genre_scores[g]) / len(genre_scores[g]))
    fav_era = max(era_scores, key=lambda e: sum(era_scores[e]) / len(era_scores[e]))
    return fav_genre, fav_era


def precision_at_k(recommended_movie_ids, test_ratings_for_user):
    """Fraction of recommended movies the user rated >= LIKE_THRESHOLD in test data."""
    if not recommended_movie_ids:
        return 0.0
    liked_test_ids = {r["movie_id"] for r in test_ratings_for_user if int(r["rating"]) >= LIKE_THRESHOLD}
    hits = sum(1 for mid in recommended_movie_ids if mid in liked_test_ids)
    return hits / len(recommended_movie_ids)


def run_config(config_name, movies, train_by_user, test_by_user, likelihood_tables, use_era, use_probability):
    precisions = []

    for user_id, train_ratings in train_by_user.items():
        test_ratings = test_by_user.get(user_id, [])
        if not test_ratings:
            continue

        fav_genre, fav_era = infer_preferences(train_ratings, movies)
        if fav_genre is None:
            continue

        preferences = {"genre": fav_genre}
        if use_era:
            preferences["era"] = fav_era

        candidates = apply_rules(movies, preferences)

        if use_probability:
            scored = score_movies(candidates, likelihood_tables, mood="neutral")
            top_movies = [m for m, s, b in scored[:TOP_N]]
        else:
            # Rules-only baseline within our own system: sort by popularity
            # tier instead of using the probability engine, to isolate what
            # the probability engine contributes.
            popularity_rank = {"high": 0, "medium": 1, "low": 2}
            candidates_sorted = sorted(candidates, key=lambda m: popularity_rank.get(m["popularity"], 3))
            top_movies = candidates_sorted[:TOP_N]

        recommended_ids = [m["movie_id"] for m in top_movies]
        precisions.append(precision_at_k(recommended_ids, test_ratings))

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    print(f"{config_name:45s} precision@{TOP_N} = {avg_precision:.3f}   (evaluated on {len(precisions)} users)")
    return avg_precision


def run_baseline(movies, train_ratings_all, test_by_user):
    recs = most_popular_recommendations(movies, train_ratings_all, top_n=TOP_N)
    recommended_ids = [m["movie_id"] for m, avg in recs]

    precisions = []
    for user_id, test_ratings in test_by_user.items():
        if not test_ratings:
            continue
        precisions.append(precision_at_k(recommended_ids, test_ratings))

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    print(f"{'BASELINE: non-personalized popularity':45s} precision@{TOP_N} = {avg_precision:.3f}   (evaluated on {len(precisions)} users)")
    return avg_precision


def main():
    movies = load_csv("data/movies.csv")
    ratings = load_csv("data/ratings.csv")

    train, test = split_train_test(ratings)

    train_by_user = defaultdict(list)
    for r in train:
        train_by_user[r["user_id"]].append(r)
    test_by_user = defaultdict(list)
    for r in test:
        test_by_user[r["user_id"]].append(r)

    # Likelihood tables built ONLY from training data -- no leakage from test set
    likelihood_tables = build_likelihood_tables(movies, train)

    print("=== Part D: Experimental Results ===\n")
    print(f"Train ratings: {len(train)}   Test ratings: {len(test)}\n")

    results = {}
    results["Config 1: Hybrid, genre-only filter"] = run_config(
        "Config 1: Hybrid, genre-only filter", movies, train_by_user, test_by_user,
        likelihood_tables, use_era=False, use_probability=True)

    results["Config 2: Hybrid, genre+era filter"] = run_config(
        "Config 2: Hybrid, genre+era filter", movies, train_by_user, test_by_user,
        likelihood_tables, use_era=True, use_probability=True)

    results["Config 3: Rules-only (no probability engine)"] = run_config(
        "Config 3: Rules-only (no probability engine)", movies, train_by_user, test_by_user,
        likelihood_tables, use_era=True, use_probability=False)

    results["Baseline: non-personalized popularity"] = run_baseline(movies, train, test_by_user)

    print("\n=== Summary ===")
    for name, score in results.items():
        print(f"{name:45s} {score:.3f}")


if __name__ == "__main__":
    main()
