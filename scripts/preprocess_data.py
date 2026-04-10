import os               
import pandas as pd         
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def preprocess_text(text):
    """Text normalization and cleaning."""
    # Convert to lowercase and strip leading/trailing whitespace
    return str(text).lower().strip()

def main():
    print("1. Loading the BANKING77 dataset")
    # Loading the dataset from Hugging Face's dataset hub
    dataset = load_dataset("banking77")

    # Convert to pandas DataFrames for easier manipulation
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    
    print("2. Sampling the dataset")
    # The original dataset has over 10,000 training rows. 
    # We will sample 15% of the data to ensure training completes with available resources.
    # We use 'stratify' to ensure all 77 intent classes are proportionally represented in our subset.
    sampled_train, _ = train_test_split(
        df_train, 
        train_size=0.15, 
        stratify=df_train['label'], 
        random_state=42
    )
    
    sampled_test, _ = train_test_split(
        df_test, 
        train_size=0.15, 
        stratify=df_test['label'], 
        random_state=42
    )

    print("3. Preprocessing text and verifying labels")
    # Apply basic cleaning
    sampled_train['text'] = sampled_train['text'].apply(preprocess_text)
    sampled_test['text'] = sampled_test['text'].apply(preprocess_text)

    # The labels in BANKING77 are already integer IDs (0 to 76), 
    # which is exactly the format needed for Sequence Classification.
    # We will keep 'text' and 'label' columns.
    sampled_train = sampled_train[['text', 'label']]
    sampled_test = sampled_test[['text', 'label']]

    print("4. Saving splits to the 'sample_data' directory")
    # Create the directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)

    # Save to CSV format as expected by the directory structure
    sampled_train.to_csv("sample_data/train.csv", index=False)
    sampled_test.to_csv("sample_data/test.csv", index=False)
    
    print(f"   -> Train set size: {len(sampled_train)} rows")
    print(f"   -> Test set size: {len(sampled_test)} rows")
    print("Data preparation complete! ✨")

if __name__ == "__main__":
    main()