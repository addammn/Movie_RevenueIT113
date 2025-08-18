# src/features/build_features.py
from __future__ import annotations
import os, ast, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from collections import Counter

RAW_MOVIES = Path("data/raw/tmdb_5000_movies.csv")
RAW_CREDITS = Path("data/raw/tmdb_5000_credits.csv")
OUT_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

TOP_K_GENRES = 15         # keep top K genres as one-hot columns
TOP_K_COMPANIES = 10      # keep top K production companies as indicators
TOP_K_CAST = 50           # build a simple "star power" from top K frequent cast names

def _safe_json_list(s: str) -> List[dict]:
    """
    TMDB CSV stores list of dicts as strings. Convert robustly.
    Returns [] on failure.
    """
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
        return []
    except Exception:
        return []

def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(RAW_MOVIES, low_memory=False)
    credits = pd.read_csv(RAW_CREDITS, low_memory=False)
    return movies, credits

def merge_raw(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    df = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="left",
        suffixes=("", "_credits"),
        validate="one_to_one",
    )
    # ensure we have a 'title' column (fallback to original_title)
    if "title" not in df.columns and "original_title" in df.columns:
        df["title"] = df["original_title"]
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # parse dates
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_quarter"] = df["release_date"].dt.quarter
    df["release_dow"] = df["release_date"].dt.dayofweek

    # season (rough)
    def season(m):
        if pd.isna(m): return np.nan
        if m in [12,1,2]: return "winter"
        if m in [3,4,5]: return "spring"
        if m in [6,7,8]: return "summer"
        return "fall"
    df["release_season"] = df["release_month"].apply(season)

    # numeric coercions
    for col in ["budget", "revenue", "popularity", "vote_average", "vote_count", "runtime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill simple numerics
    if "runtime" in df.columns:
        df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    if "popularity" in df.columns:
        df["popularity"] = df["popularity"].fillna(df["popularity"].median())
    for col in ["vote_average", "vote_count"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # optional filter: drop clearly unusable rows for training target
    # keep rows where revenue and budget are >= 0 (you may later filter >0 only for training)
    df = df[(df["revenue"].notna()) & (df["budget"].notna())]

    # cap extreme outliers to reasonable quantiles (winsorize-like) but keep original distributions
    for col in ["budget", "revenue", "popularity", "vote_count"]:
        if col in df.columns:
            hi = df[col].quantile(0.995)
            df[col] = np.where(df[col] > hi, hi, df[col])

    # log transforms for skewed numeric features (keep originals, add log_)
    for col in ["budget", "popularity", "vote_count"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # target log (useful for linear models)
    if "revenue" in df.columns:
        df["log_revenue"] = np.log1p(df["revenue"].clip(lower=0))

    return df

def genres_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # extract list of genre names per row
    df["genres_list"] = df["genres"].apply(lambda s: [d.get("name") for d in _safe_json_list(s)] if isinstance(s, str) else [])
    # find top K genres
    all_genres = Counter(g for row in df["genres_list"] for g in row)
    top_genres = [g for g, _ in all_genres.most_common(TOP_K_GENRES)]
    # one-hot encode for top genres
    for g in top_genres:
        df[f"genre_{g}"] = df["genres_list"].apply(lambda lst: 1 if g in lst else 0)
    df["genre_count"] = df["genres_list"].apply(len)
    return df, top_genres

def companies_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # production_companies is a list of dicts with 'name'
    df["prod_companies_list"] = df["production_companies"].apply(
        lambda s: [d.get("name") for d in _safe_json_list(s)] if isinstance(s, str) else []
    )
    all_companies = Counter(c for row in df["prod_companies_list"] for c in row)
    top_companies = [c for c, _ in all_companies.most_common(TOP_K_COMPANIES)]
    for c in top_companies:
        safe = c.replace(" ", "_").replace("/", "_")
        df[f"pc_{safe}"] = df["prod_companies_list"].apply(lambda lst: 1 if c in lst else 0)
    df["prod_company_count"] = df["prod_companies_list"].apply(len)
    return df, top_companies

def cast_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # credits 'cast' is list of dicts with 'name' and 'order'
    df["cast_list"] = df["cast"].apply(
        lambda s: [d.get("name") for d in _safe_json_list(s)] if isinstance(s, str) else []
    )
    df["cast_size"] = df["cast_list"].apply(len)
    # Top K frequent cast names (star power proxy)
    all_cast = Counter(c for row in df["cast_list"] for c in row)
    top_cast = [c for c, _ in all_cast.most_common(TOP_K_CAST)]
    df["star_power_count"] = df["cast_list"].apply(lambda lst: sum(1 for n in lst if n in top_cast))
    return df, top_cast

def assemble_features(df: pd.DataFrame,
                      top_genres: List[str],
                      top_companies: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the final feature matrix X and target y.
    """
    # core numeric + time features
    base_cols = [
        "budget","log_budget","popularity","log_popularity","vote_average","vote_count","log_vote_count",
        "runtime","release_year","release_month","release_quarter","release_dow","genre_count",
        "prod_company_count","cast_size","star_power_count"
    ]
    # some columns might not exist in dataset; keep only those present
    base_cols = [c for c in base_cols if c in df.columns]

    # season one-hot
    season_cols = []
    if "release_season" in df.columns:
        seasons = ["winter","spring","summer","fall"]
        for s in seasons:
            col = f"season_{s}"
            df[col] = (df["release_season"] == s).astype(int)
            season_cols.append(col)

    # genre one-hots
    genre_cols = [f"genre_{g}" for g in top_genres]

    # company indicators
    company_cols = [f"pc_{c.replace(' ','_').replace('/','_')}" for c in top_companies]

    # target
    y = df["revenue"].copy()

    X_cols = base_cols + season_cols + genre_cols + company_cols
    X = df[X_cols].copy().fillna(0)

    return X, y

def save_outputs(df_full: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                 top_genres: List[str], top_companies: List[str], top_cast: List[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save combined processed table (with target & a few id columns for traceability)
    keep_cols = ["id","title","revenue","log_revenue"] if "log_revenue" in df_full.columns else ["id","title","revenue"]
    for c in keep_cols:
        if c not in df_full.columns:
            # tolerate missing title/id in some dumps
            pass
    processed = pd.concat([df_full[keep_cols], X], axis=1, join="inner") if set(keep_cols).issubset(df_full.columns) else pd.concat([df_full, X], axis=1, join="inner")
    processed.to_csv(OUT_DIR / "movies_processed.csv", index=False)

    # Save pure features/target (handy for Step 4)
    X.to_csv(OUT_DIR / "X_features.csv", index=False)
    y.to_frame("revenue").to_csv(OUT_DIR / "y_target.csv", index=False)

    # brief dictionary / spec
    spec = {
        "n_rows": int(len(df_full)),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "top_genres": top_genres,
        "top_companies": top_companies,
        "top_cast_names_used_for_star_power": top_cast,
    }
    with open(REPORTS_DIR / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    # quick profile for sanity
    profile = pd.DataFrame({
        "column": X.columns,
        "non_null": X.notna().sum().values,
        "nulls": X.isna().sum().values,
        "mean": X.mean(numeric_only=True).values,
        "std": X.std(numeric_only=True).values,
    })
    profile.to_csv(REPORTS_DIR / "processed_features_profile.csv", index=False)

def main():
    movies, credits = load_raw()
    df = merge_raw(movies, credits)
    df = basic_clean(df)
    df, top_genres = genres_features(df)
    df, top_companies = companies_features(df)
    df, top_cast = cast_features(df)
    # Ensure log_vote_count exists if vote_count exists
    if "vote_count" in df.columns and "log_vote_count" not in df.columns:
        df["log_vote_count"] = np.log1p(df["vote_count"].clip(lower=0))

    X, y = assemble_features(df, top_genres, top_companies)
    save_outputs(df, X, y, top_genres, top_companies, top_cast)

    print(f"✅ Done. Saved to {OUT_DIR} and specs to {REPORTS_DIR}.")
    print(f"   Features shape: {X.shape}; Target length: {len(y)}")

if __name__ == "__main__":
    main()
