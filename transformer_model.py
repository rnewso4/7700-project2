import torch
import torch.nn as nn

# Define a simple transformer-based language model
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=360, nhead=6, num_encoder_layers=3, dim_feedforward=512):
        super(TransformerLanguageModel, self).__init__()

        # Embedding layer for tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Output projection to vocabulary size
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Initialization
        #self._init_weights()


    def forward(self, src, src_mask=None):
        """
        Forward pass for the transformer language model.

        :param input_ids: Tensor of token IDs (batch_size, seq_len).
        :return: Logits for the vocabulary (batch_size, seq_len, vocab_size).
        """
        # Embed tokens and scale by sqrt(d_model)
        x = self.token_embedding(src) * (self.token_embedding.embedding_dim ** 0.5)

        # Add positional encodings
        seq_length = x.size(1)
        #x += self.positional_encoding[:seq_length, :]
        x = self.positional_encoding(x)
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_mask)

        # Project to vocabulary size
        logits = self.output_layer(x)

        return logits, None
    
    def predict_next_token(self, input_ids, temperature):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids.
        :param input_ids: Input sequence token IDS
        :param temperature: temperature setting for sampling:
        :return: next token ID, hidden state
        """

        self.eval()
        hidden = None
        with torch.no_grad():
            logits, hidden = self.forward(input_ids, hidden) # 32 is the batch size
            next_token_id = top_p_sampling(logits[0, -1, :], p=temperature)
            return next_token_id.item(), hidden
        
    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=0.9, device='mps'):
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

 # Define top-p sampling function
def top_p_sampling(output_logits, p=0.9):
    """
    Perform top-p (nucleus) sampling on the output logits.
    Args:
        output_logits (Tensor): The logits output from the model (shape: [vocab_size]).
        p (float): The cumulative probability threshold for top-p sampling.
    Returns:
        int: The sampled token index.
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(output_logits, dim=-1)

    # Sort probabilities and indices in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff index where cumulative probability exceeds 'p'
    cutoff_index = torch.where(cumulative_probs > p)[0][0]

    # Keep only the top-p tokens
    top_p_probs = sorted_probs[:cutoff_index + 1]
    top_p_indices = sorted_indices[:cutoff_index + 1]

    # Normalize the top-p probabilities
    top_p_probs /= top_p_probs.sum()

    # Sample from the top-p tokens
    sampled_index = torch.multinomial(top_p_probs, 1).item()
    return top_p_indices[sampled_index]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)