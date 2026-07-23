"""
generate_data.py

Creates a small, self-contained movie dataset for the recommender project.
No internet download needed -- this generates:
  data/movies.csv   -> the movie catalogue (id, title, genre, era, popularity)
  data/ratings.csv  -> simulated user ratings (user_id, movie_id, rating 1-5)

Why simulated data? Real datasets like MovieLens require a download from an
external site. For a reliable, reproducible student project, a small synthetic
dataset with realistic patterns built in works just as well for demonstrating
the AI techniques, and removes any risk of the demo breaking on the day.
"""

import csv
import random

random.seed(42)  # reproducible results every run

GENRES = ["Action", "Comedy", "Drama", "Horror", "SciFi", "Romance"]
ERAS = ["1980s", "1990s", "2000s", "2010s", "2020s"]

# --- 1. Build the movie catalogue ---
MOVIE_TITLES = [
    "Neon Skyline", "Last Signal", "Quiet Harbor", "The Long Drive", "Static Bloom",
    "Northbound", "Glass City", "Midnight Circuit", "Paper Moonlight", "Iron Season",
    "The Wrong Exit", "Salt & Ash", "Falling Upward", "Echo Chamber", "The Slow Fire",
    "Borrowed Time", "Winter Arcade", "Red Tide Rising", "The Cartographer", "Second Sunrise",
    "Hollow Orbit", "Velvet Static", "Ghost Frequency", "The Last Ember", "Copper Rain",
    "Blue Hour", "Nightshift Diner", "The Silence Engine", "Marble Sky", "Fractured Light",
    "The Wandering Signal", "Gravity's Edge", "Painted Static", "The Departure Gate", "Coyote Moon",
    "Static & Snow", "The Forgotten Frequency", "Amber Skyline", "The Last Broadcast", "Hollow Pines",
    "Wildfire Season", "The Quiet Coup", "Neon Requiem", "Paper Satellites", "The Drift",
    "Aftershock City", "The Long Goodbye Tour", "Static Horizon", "The Midnight Ledger", "Ember & Frost",
]

movies = []
for i, title in enumerate(MOVIE_TITLES, start=1):
    genre = random.choice(GENRES)
    era = random.choice(ERAS)
    # popularity tier: weighted so most movies are "medium", few are "high"
    popularity = random.choices(["low", "medium", "high"], weights=[0.35, 0.45, 0.20])[0]
    runtime = random.randint(85, 165)
    movies.append({
        "movie_id": i,
        "title": title,
        "genre": genre,
        "era": era,
        "popularity": popularity,
        "runtime_minutes": runtime,
    })

with open("data/movies.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["movie_id", "title", "genre", "era", "popularity", "runtime_minutes"])
    writer.writeheader()
    writer.writerows(movies)

# --- 2. Simulate users with hidden preferences, then generate ratings ---
NUM_USERS = 60
ratings = []

for user_id in range(1, NUM_USERS + 1):
    # each simulated user has a "true" favourite genre and era -- this creates
    # the realistic patterns the probabilistic scorer will later detect
    fav_genre = random.choice(GENRES)
    fav_era = random.choice(ERAS)

    # each user rates a random subset of movies (simulating incomplete data,
    # which is realistic -- not everyone has seen every movie)
    seen_movies = random.sample(movies, k=random.randint(15, 30))

    for movie in seen_movies:
        base_score = 3.0  # neutral starting point

        if movie["genre"] == fav_genre:
            base_score += 1.2
        if movie["era"] == fav_era:
            base_score += 0.6
        if movie["popularity"] == "high":
            base_score += 0.3

        # add random noise so it's not perfectly predictable
        noise = random.gauss(0, 0.6)
        final_score = base_score + noise

        # clip to valid 1-5 rating range and round to nearest integer
        rating = max(1, min(5, round(final_score)))

        ratings.append({
            "user_id": user_id,
            "movie_id": movie["movie_id"],
            "rating": rating,
        })

with open("data/ratings.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["user_id", "movie_id", "rating"])
    writer.writeheader()
    writer.writerows(ratings)

print(f"Generated {len(movies)} movies and {len(ratings)} ratings.")
print("Saved to data/movies.csv and data/ratings.csv")
