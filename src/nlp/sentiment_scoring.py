import pandas as pd
from pathlib import Path
from textblob import TextBlob

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "nlp"
INPUT_FILE = PROCESSED_DIR / "cleaned_instagram_text.csv"
OUTPUT_FILE = PROCESSED_DIR / "labeled_instagram_nlp.csv"

def get_sentiment_label(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    print("--> Starting Sentiment Scoring...")
    
    if not INPUT_FILE.exists():
        print(f"Error: Cleaned file not found at {INPUT_FILE}. Run cleaning and vectorization first.")
        return

    # Load data
    df = pd.read_csv(INPUT_FILE)
    df['clean_text'] = df['clean_text'].fillna('')
    
    # Calculate polarity using TextBlob
    print("Calculating sentiment polarities...")
    df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Assign discrete sentiment labels
    df['sentiment'] = df['polarity'].apply(get_sentiment_label)
    
    # Save labeled dataset
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Sentiment scoring complete! Distribution:")
    print(df['sentiment'].value_counts())
    print(f"Saved labeled data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()