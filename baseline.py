"""
baseline.py

A simple BASELINE recommender, required by Part D of the assignment to
give something to compare the hybrid system against.

This baseline is deliberately "dumb": it ignores the user's stated
preferences entirely and just recommends the most popular movies overall
(highest average rating among movies with a reasonable number of ratings).
This is a standard, well-known baseline in recommender systems research --
if your smarter system can't beat this, it isn't earning its complexity.
"""

from collections import defaultdict


def most_popular_recommendations(movies, ratings, top_n=5):
    """
    Returns the top_n movies with the highest average rating, ignoring any
    user preferences. Requires at least MIN_RATINGS ratings to avoid a
    single 5-star rating on an obscure movie topping the list.
    """
    MIN_RATINGS = 5

    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in ratings:
        mid = r["movie_id"]
        sums[mid] += int(r["rating"])
        counts[mid] += 1

    movie_lookup = {m["movie_id"]: m for m in movies}

    averages = []
    for mid, total in sums.items():
        if counts[mid] >= MIN_RATINGS:
            avg = total / counts[mid]
            averages.append((movie_lookup[mid], avg))

    averages.sort(key=lambda x: x[1], reverse=True)
    return averages[:top_n]
