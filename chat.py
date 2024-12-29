import random
import json
import torch
from model import ChatbotModel  # Import the new GPT-2 based model
from nltk_utils import bag_of_words, tokenize

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Initialize the model (GPT-2 model from model.py)
chatbot = ChatbotModel().to(device)

# Define bot name
bot_name = "Sam"

def get_response(msg):
    """
    Function to get a response from the GPT-2 model based on the input message.
    """
    # Generate response using the GPT-2 model
    response = chatbot.generate_response(msg)
    return response

def chat():
    """
    Main function for chatting with the bot.
    """
    print(f"{bot_name}: Let's chat! (type 'quit' to exit)")

    while True:
        sentence = input("You: ")  # Get user input
        if sentence.lower() == "quit":
            print(f"{bot_name}: Goodbye!")
            break

        # Get and display the response
        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")

if __name__ == "__main__":
    chat()
