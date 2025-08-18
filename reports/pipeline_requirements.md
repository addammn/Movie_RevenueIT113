# Pipeline Requirements (from Quick Scan)

- Inputs: data/raw/tmdb_5000_movies.csv, data/raw/tmdb_5000_credits.csv
- Join key: movies.id == credits.movie_id (1–1 expected)
- Columns with high nulls: (refer to movies_columns_profile.csv / credits_columns_profile.csv)
- Cleans:
  - Consider removing rows with revenue == 0 for training target validity
  - Handle extreme outliers in budget/revenue (winsorize or log-transform)
- Feature ideas:
  - release_year, release_quarter, release_season
  - one-hot encode genres
  - cast_size (parse JSON in credits)
  - production_company_count