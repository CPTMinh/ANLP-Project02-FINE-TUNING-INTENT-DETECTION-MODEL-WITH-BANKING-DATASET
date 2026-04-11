import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import time

# Import your new inference class
from inference import IntentClassification

def main():
    print("1. Loading Test Data...")
    test_df = pd.read_csv("sample_data/test.csv")
    
    print("2. Initializing Inference Model...")
    classifier = IntentClassification(model_path="configs/inference.yaml")
    
    # We will grab all texts and true labels
    texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist()
    predicted_labels = []

    print(f"3. Running evaluation over {len(texts)} test samples...")
    start_time = time.time()
    
    # Loop continuously with a progress bar
    for text in tqdm(texts, desc="Predicting Intents"):
        pred = classifier(text)
        predicted_labels.append(pred)
        
    end_time = time.time()
    
    print("\n--- Evaluation Results ---")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    
    # Calculate Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    # Print the scikit-learn classification report
    # Note: Depending on your sample size, some classes might not appear in the test set.
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

if __name__ == "__main__":
    main()