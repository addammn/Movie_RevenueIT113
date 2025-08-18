import subprocess, sys, os

RAW_DIR = os.path.join("data", "raw")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", "tmdb/tmdb-movie-metadata", "-p", RAW_DIR, "--unzip"]
    print("Downloading TMDB dataset...")
    code = subprocess.call(cmd)
    if code != 0:
        sys.exit("Kaggle download failed. Ensure kaggle.json is configured and Kaggle CLI is installed.")
    print("Done. Files are in data/raw")

if __name__ == "__main__":
    main()
