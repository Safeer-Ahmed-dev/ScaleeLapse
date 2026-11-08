import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "nlp"
INPUT_FILE = PROCESSED_DIR / "labeled_instagram_nlp.csv"
TRAIN_OUTPUT = PROCESSED_DIR / "train_nlp.csv"
VAL_OUTPUT = PROCESSED_DIR / "val_nlp.csv"

def main():
    print("--> Starting Train/Validation Data Split...")
    
    if not INPUT_FILE.exists():
        print(f"Error: Labeled file not found at {INPUT_FILE}. Run sentiment scoring first.")
        return

    # Load labeled dataset
    df = pd.read_csv(INPUT_FILE)
    
    # Split into 80% Training and 20% Validation sets (stratified to maintain sentiment balance)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment'] if 'sentiment' in df.columns else None
    )
    
    # Save split datasets
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    val_df.to_csv(VAL_OUTPUT, index=False)
    
    print(f"Split complete!")
    print(f"Training set saved to: {TRAIN_OUTPUT} ({len(train_df)} rows)")
    print(f"Validation set saved to: {VAL_OUTPUT} ({len(val_df)} rows)")

if __name__ == "__main__":
    main()