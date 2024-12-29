import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatbotModel(torch.nn.Module):
    def __init__(self):
        super(ChatbotModel, self).__init__()
        # Load pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, input_text):
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']

        # Generate response from GPT-2 model
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=self.pad_token_id)
        
        # Decode the generated text
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def generate_response(self, input_text):
        return self.forward(input_text)
