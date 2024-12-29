import torch
import random
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk_utils import bag_of_words, tokenize, stem

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Fine-tune GPT-2
def fine_tune_gpt2(data, tokenizer, model, epochs=3, batch_size=4):
    """
    Fine-tune GPT-2 model on your intents data
    """
    # Prepare the dataset and DataLoader
    inputs = []
    labels = []
    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            # Combine the intent tag and pattern to create a sentence
            sentence = f"{tag}: {pattern}"
            inputs.append(sentence)
            labels.append(tag)

    # Tokenize inputs and prepare DataLoader
    encoding = tokenizer(inputs, truncation=True, padding=True, return_tensors='pt', max_length=50)
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    # DataLoader to batch data
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask = batch
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

# Fine-tune GPT-2 with your data
fine_tuned_model = fine_tune_gpt2(intents, tokenizer, model, epochs=3)

# Save the fine-tuned model
fine_tuned_model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

# Function to generate a response using the fine-tuned GPT-2 model
def get_response(msg):
    """
    Generate a response using the fine-tuned GPT-2 model.
    """
    # Prepare the input sentence for GPT-2
    input_sentence = f"User: {msg} Bot:"
    inputs = tokenizer(input_sentence, return_tensors='pt').to(model.device)
    
    # Generate response
    output = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7)
    
    # Decode the response and return it
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace("User:", "").replace("Bot:", "").strip()

def chat():
    """
    Main function for chatting with the bot.
    """
    print("Chatbot: Let's chat! (type 'quit' to exit)")

    while True:
        sentence = input("You: ")  # Get user input
        if sentence.lower() == "quit":
            print("Chatbot: Goodbye!")
            break

        # Get and display the response
        resp = get_response(sentence)
        print(f"Chatbot: {resp}")

if __name__ == "__main__":
    chat()
