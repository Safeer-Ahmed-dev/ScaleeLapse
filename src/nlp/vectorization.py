import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "nlp"
INPUT_FILE = PROCESSED_DIR / "cleaned_instagram_text.csv"
OUTPUT_FILE = PROCESSED_DIR / "vectorized_features.csv"

def main():
    print("--> Starting TF-IDF Text Vectorization...")
    
    if not INPUT_FILE.exists():
        print(f"Error: Cleaned text file not found at {INPUT_FILE}. Run cleaning first.")
        return

    # Load cleaned data
    df = pd.read_csv(INPUT_FILE)
    
    # Handle any missing values in clean_text
    df['clean_text'] = df['clean_text'].fillna('')
    
    # Initialize TF-IDF Vectorizer (limiting to top 500 features for optimal SME feature extraction)
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    
    # Fit and transform the clean text
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    # Convert matrix to a DataFrame for export
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    # Combine original data columns with the new TF-IDF features
    combined_df = pd.concat([df, tfidf_df], axis=1)
    
    # Save vectorized features
    combined_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Vectorization complete! Extracted {len(feature_names)} features.")
    print(f"Saved vectorized features to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()