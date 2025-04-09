import torch
import torch.nn as nn

# Define a simple transformer-based language model
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        # self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src)
        output = self.transformer_encoder(embedded, mask=src_mask)
        output = self.fc_out(output)
        return output, None
        #tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        #output = self.transformer(src, tgt)
        # output = self.fc_out(output)
        # return output, None
    
    def predict_next_token(self, input_ids, temperature=1.0):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids.
        :param input_ids: Input sequence token IDS
        :param temperature: temperature setting for sampling:
        :return: next token ID, hidden state
        """
     # Ensure the model is in evaluation mode
        self.eval()

        # Convert input_ids to tensor if not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device="mps").unsqueeze(0)  # Add batch dimension

        hidden = None
        with torch.no_grad():
            # Pass input_ids through the transformer model
            logits, hidden = self.forward(input_ids, hidden) # 32 is the batch size

            # Get the logits for the last token in the sequence
            last_token_logits = logits[:, -1, :]

            # Apply temperature scaling
            scaled_logits = last_token_logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(scaled_logits, dim=-1)

            # Sample the next token ID from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1)

        return next_token_id.item(), hidden
        
    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='mps'):
        """
        Generate a full ouput sequence given a prompt.

        :param tokenizer: the trained SentencePiece tokenizer
        :param prompt: the input prompt (plain text string)
        :param max_length: Maximum number of tokens to generate auto-regressively before stopping
        :param eos_token_id: the token ID of the EOS token
        :param temperature: Temperature setting for sampling
        :param device: Device we are using to run the model
        """

        self.eval() # set the model to evaluation mode
        input_ids = tokenizer.encode(prompt, out_type=int) # Encode the input string into token IDs
        # convert token ID list to tensor,move to device memory, and adding a batch dimension (1D to 2D)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = [] # this will store the generated token IDs
        hidden = None # initial hidden state is None

        # loop over max output tokens
        for i in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, temperature)

            # exit early if the model generated <eos> token ID
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            # keep track of generated tokens
            generated_ids.append(next_token_id)

            #the input to the next step is just the new token and the hidden state
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        # decode generated token IDs into tokens
        return tokenizer.decode(generated_ids, out_type=str)
