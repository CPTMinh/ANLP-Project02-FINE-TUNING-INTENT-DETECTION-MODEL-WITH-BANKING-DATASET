import yaml
import torch
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, model_path):
        """
        Initialize the inference class.
        As per requirements, `model_path` points to a configuration file 
        that contains the path to the saved model checkpoint.
        """
        # Load configuration
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Extract the saved model directory from the loaded configuration
        checkpoint_dir = self.config.get("save_model_path", "saved_model")
        max_seq_length = self.config.get("max_seq_length", 128)

        print(f"Loading checkpoint from: {checkpoint_dir}")

        # Load the fine-tuned model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_dir,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )

        # Enable Unsloth's native 2x faster
        FastLanguageModel.for_inference(self.model)
    
    def __call__(self, message):
        """
        Predict the intent label for a single text input.
        """
        # Format the prompt exactly as it was during training, stopping at "Intent ID: "
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Classify the intent of this text into a category (0-76):\n"
            f"Text: {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"Intent ID: "
        )
        
        # Check if CUDA is available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tokenize and send to the correct device
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        
        # Generate the response (we only need a few tokens for a number between 0 and 76)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=5, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the generated output text
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the predicted label ID from the generated output
        try:
            # Look for the text after "Intent ID: "
            response = output_text.split("Intent ID:")[-1].strip()
            # Extract just the numeric part using regex or a filter just in case there are spaces/punctuation
            predicted_label = int(''.join(filter(str.isdigit, response)))
            return predicted_label
        except Exception as e:
            print(f"Error parsing predicted label: {e}\nRaw output: {output_text}")
            return -1


if __name__ == "__main__":
    print("1. Initializing Inference Model")
    # Pass the CONFIG file path
    classifier = IntentClassification(model_path="configs/inference.yaml")

    # Test single inference
    test_message = "I lost my credit card yesterday, how do I freeze my account?"
    print(f"\nInput Message: '{test_message}'")
    
    # Call the model
    predicted_intent = classifier(test_message)
    
    # Give output
    print(f"> Predicted Intent ID: {predicted_intent}")