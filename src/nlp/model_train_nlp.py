import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "nlp"
TRAIN_FILE = PROCESSED_DIR / "train_nlp.csv"
VAL_FILE = PROCESSED_DIR / "val_nlp.csv"

def main():
    print("--> Starting Model Training & Evaluation...")
    
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        print("Error: Train or Validation files not found. Run previous pipeline steps first.")
        return

    # Load train and validation sets
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    
    # Fill missing values
    train_df['clean_text'] = train_df['clean_text'].fillna('')
    val_df['clean_text'] = val_df['clean_text'].fillna('')
    
    X_train_text = train_df['clean_text']
    y_train = train_df['sentiment']
    
    X_val_text = val_df['clean_text']
    y_val = val_df['sentiment']
    
    # Vectorize text using TF-IDF (fit on train, transform both train and val)
    print("Vectorizing text features...")
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    
    # Train Logistic Regression model
    print("Training classifier...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model on validation set
    print("Evaluating model performance...")
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

if __name__ == "__main__":
    main()