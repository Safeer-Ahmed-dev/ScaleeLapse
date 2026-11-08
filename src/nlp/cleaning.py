import re
import pandas as pd
from pathlib import Path

# Define paths relative to root
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = BASE_DIR / "data" / "processed" / "nlp" / "raw_extracted_captions.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "nlp"
OUTPUT_FILE = PROCESSED_DIR / "cleaned_instagram_text.csv"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove mentions (@username) and hashtags symbol (keep the word or strip entirely? Let's strip symbols)
    text = re.sub(r'@\w+', '', text)
    
    # Keep alphanumeric characters and basic punctuation/spaces, remove emojis and special symbols
    text = re.sub(r'[^a-z0-9\s#]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    print("--> Starting text cleaning process...")
    
    if not RAW_FILE.exists():
        print(f"Error: Raw captions file not found at {RAW_FILE}. Run extraction first.")
        return

    # Load raw dataset
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded raw dataset with {len(df)} rows.")

    # Apply cleaning function to caption column
    print("Cleaning captions...")
    df['clean_text'] = df['caption'].apply(clean_text)
    
    # Drop rows where clean text is empty or blank
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    
    # Save cleaned dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Cleaning complete! Saved {len(df)} cleaned records to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()