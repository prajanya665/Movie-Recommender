"""
gui.py

A simple GUI interface for the hybrid recommender, built with Tkinter
(part of the Python standard library -- no extra installs needed).

This is a thin presentation layer only: it calls the exact same
rule_engine.py and probability_engine.py used by the CLI version
(cli.py). The underlying AI logic is unchanged -- only the way the user
interacts with it is different.

Run with:  python3 gui.py
"""

import csv
import tkinter as tk
from tkinter import ttk, scrolledtext

from rule_engine import apply_rules, explain_rule_pass
from probability_engine import build_likelihood_tables, score_movies


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class RecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Movie Recommender")
        self.root.geometry("720x600")

        # Load data and build the probability tables once at startup,
        # exactly as cli.py does.
        self.movies = load_csv("data/movies.csv")
        self.ratings = load_csv("data/ratings.csv")
        self.likelihood_tables = build_likelihood_tables(self.movies, self.ratings)

        self.genres = sorted(set(m["genre"] for m in self.movies))
        self.eras = sorted(set(m["era"] for m in self.movies))

        self._build_widgets()

    def _build_widgets(self):
        padding = {"padx": 10, "pady": 6}

        title = tk.Label(self.root, text="Hybrid Knowledge-Based + Probabilistic Movie Recommender",
                          font=("Segoe UI", 13, "bold"))
        title.pack(pady=(14, 4))

        subtitle = tk.Label(self.root, text="Rule engine (hard filters) + probability engine (uncertainty scoring)",
                             font=("Segoe UI", 9), fg="#555")
        subtitle.pack(pady=(0, 10))

        form = tk.Frame(self.root)
        form.pack(**padding)

        # Genre dropdown
        tk.Label(form, text="Genre:").grid(row=0, column=0, sticky="e", **padding)
        self.genre_var = tk.StringVar(value="Any")
        genre_dropdown = ttk.Combobox(form, textvariable=self.genre_var,
                                       values=["Any"] + self.genres, state="readonly", width=20)
        genre_dropdown.grid(row=0, column=1, **padding)

        # Era dropdown
        tk.Label(form, text="Era:").grid(row=0, column=2, sticky="e", **padding)
        self.era_var = tk.StringVar(value="Any")
        era_dropdown = ttk.Combobox(form, textvariable=self.era_var,
                                     values=["Any"] + self.eras, state="readonly", width=20)
        era_dropdown.grid(row=0, column=3, **padding)

        # Max runtime
        tk.Label(form, text="Max runtime (min):").grid(row=1, column=0, sticky="e", **padding)
        self.runtime_var = tk.StringVar(value="")
        runtime_entry = tk.Entry(form, textvariable=self.runtime_var, width=22)
        runtime_entry.grid(row=1, column=1, **padding)

        # Mood dropdown
        tk.Label(form, text="Mood:").grid(row=1, column=2, sticky="e", **padding)
        self.mood_var = tk.StringVar(value="neutral")
        mood_dropdown = ttk.Combobox(form, textvariable=self.mood_var,
                                      values=["neutral", "light", "intense"], state="readonly", width=20)
        mood_dropdown.grid(row=1, column=3, **padding)

        # Button
        recommend_btn = tk.Button(self.root, text="Get Recommendations",
                                   command=self.get_recommendations,
                                   bg="#2d6cdf", fg="white", font=("Segoe UI", 10, "bold"),
                                   padx=12, pady=6, relief="flat", cursor="hand2")
        recommend_btn.pack(pady=(6, 12))

        # Results box
        self.results_box = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Consolas", 10))
        self.results_box.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.results_box.insert(tk.END, "Set your preferences above and click 'Get Recommendations'.")
        self.results_box.configure(state="disabled")

    def get_recommendations(self):
        genre = self.genre_var.get()
        era = self.era_var.get()
        mood = self.mood_var.get()
        runtime_text = self.runtime_var.get().strip()

        preferences = {}
        if genre != "Any":
            preferences["genre"] = genre
        if era != "Any":
            preferences["era"] = era
        if runtime_text.isdigit():
            preferences["max_runtime"] = int(runtime_text)

        # Same two-stage pipeline as the CLI: rule engine, then probability engine.
        candidates = apply_rules(self.movies, preferences)
        scored = score_movies(candidates, self.likelihood_tables, mood=mood)

        self._display_results(scored, preferences)

    def _display_results(self, scored, preferences):
        self.results_box.configure(state="normal")
        self.results_box.delete("1.0", tk.END)

        if not scored:
            self.results_box.insert(tk.END, "No movies matched your constraints. Try loosening genre/era/runtime.")
            self.results_box.configure(state="disabled")
            return

        self.results_box.insert(tk.END, "=== Top 5 Recommendations ===\n\n")
        for rank, (movie, score, breakdown) in enumerate(scored[:5], start=1):
            rule_reason = explain_rule_pass(movie, preferences)
            self.results_box.insert(
                tk.END,
                f"{rank}. {movie['title']}  ({movie['genre']}, {movie['era']}, "
                f"{movie['runtime_minutes']} min, popularity: {movie['popularity']})\n"
                f"   Confidence score: {breakdown['final_score']}\n"
                f"   Rule stage: {rule_reason}\n"
                f"   Probability breakdown: P(genre)={breakdown['P(like|genre)']}, "
                f"P(era)={breakdown['P(like|era)']}, P(popularity)={breakdown['P(like|popularity)']}, "
                f"mood_weight={breakdown['mood_weight']}\n\n"
            )

        self.results_box.configure(state="disabled")


def main():
    root = tk.Tk()
    app = RecommenderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
