import pandas as pd
from datasets import load_dataset
from pathlib import Path

# Define paths assuming folders are already in place
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "nlp" / "raw_extracted_captions.csv"

print("Fetching caption dataset from Hugging Face...")

# Load dataset and drop image column
ds = load_dataset("kkcosmos/instagram-images-with-captions", split="train")
if "image" in ds.column_names:
    ds = ds.remove_columns(["image"])

# Convert to DataFrame and save
df = pd.DataFrame(ds)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Successfully saved captions to: {OUTPUT_FILE}")