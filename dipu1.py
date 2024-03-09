import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config

# Define the paths to the model and configuration files
model_path = "E:/content/model.safetensors"
config_path = "E:/content/config.json"

# Load the model configuration
config = GPT2Config.from_json_file(config_path)

# Load the model
model = GPT2ForSequenceClassification.from_pretrained(model_path, config=config)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to generate response
def generate_response(question):
    # Tokenize the input question
    input_ids = tokenizer.encode(question, return_tensors='pt', max_length=512)

    # Generate output from the model
    output = model(input_ids)

    # Get the predicted label
    predicted_label = torch.argmax(output.logits, dim=1).item()

    # Convert predicted label to text (You might need a label encoder)
    predicted_answer = predicted_label  # You need to replace this with your label decoding logic

    return predicted_answer

# Example question
question = "How can I stop smoking?"
response = generate_response(question)
print(f"Question: {question}")
print(f"Response: {response}")
