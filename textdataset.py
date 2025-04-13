from torch.utils.data import Dataset, DataLoader
import torch
from gru_module import GRUModule
import sentencepiece as spm
import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from timeit import default_timer as timer

class TextDataSet(Dataset):
  def __init__(self, filepath, tokenizer, max_seq_len=128):
    """
    Create a text dataset for PyTorch Dataset that handles our jsonl prompts+completions for Casual LM
    :param filepath: path to the jsonl file
    :param tokenizer: instance of trained SentencePiece tokenizer
    :param max_seq_len: maximum sequence length we want to allow
    """
    self.samples = []
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len

    # open the jsonl file and tokenize each samples
    with open(filepath, "r", encoding="utf-8") as f:
      for line in f:
        item = json.loads(line)
        
        # we are using causal language modeling, prompts and completions treated the same way!
        text = item["prompt"] + " " + item["completion"]

        # tokenize the full prompt + completion (truncate at max sequence length)
        token_ids = tokenizer.encode(text, out_type=int)[:max_seq_len]

        # make sure we don't have any overly short samples
        if len(token_ids) < 2:
          continue

        # append tokenized sample to list
        self.samples.append(token_ids)

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx):
    """
    Get and format a samples at the given index.
    For Causal Language Modeling, we will train the model to predict every next token in the sequence given the prior ones
    :param idx: index of the sample to get
    :return: input_ids, target_ids
    """
    tokens = self.samples[idx]
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long)
    return input_ids, target_ids

  def collate_fn(self, batch):
    """
    Ensure batch is appropriately sized and padded for efficient training
    :param batch: batch from DataLoader, which will be a list of Tuples of token ID tensors (which could be different sizes)
    :return: collated input and target batch
    """
    input_ids, target_ids = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=3)
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=3)
    return input_ids, target_ids
  
  def train_model(self, model, epochs=3):
    device = torch.device("mps")

    # load in out tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("bpe_tokenizer.model")
    tokenizer = sp
    vocab_size = tokenizer.get_piece_size() # this gets the vocabulary size

    # load in the training and validation datasets
    train_dataset = TextDataSet("data/train.jsonl", tokenizer, 50) # 128 = max sequence length
    val_dataset = TextDataSet("data/train.jsonl", tokenizer, 50) # 128 = max sequence length

    train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [round(len(train_dataset) * .8), round(len(train_dataset) * .2)])

    # This will handle batching and shuffling during training. collate_fn handles padding of uneven sequences
    train_loader = DataLoader(train_dataset_split, batch_size=128, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset_split, batch_size=128, shuffle=False, collate_fn=val_dataset.collate_fn)

    # Instantiate our model and move it to the correct device memory

    # Using AdamW optimizer on the trainable params. This is a standard for LMs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # going to use a learning rate scheduler that reduces LR by half after stagnation for 1 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=3) # ignore the padding token

    best_val_loss = float('inf') # keep track of the best validation loss
    no_improvement = 0 # keep track of the number of epochs with no improvement

    # this will store the train and validation loss curves
    train_losses, val_losses = [], []

    train_time_start = timer()

    for ep in range(epochs): # loop over epochs
      model.train() # set the model to training mode
      total_train_loss = 0 # keep track of training loss total
      total_val_loss = 0

      # loop through each sample batch in training
      for input_ids, target_ids in train_loader:
        # move input and target tensors to device memory
        input_ids = input_ids.to("mps")
        target_ids = target_ids.to("mps")

        # reset gradients between batches
        optimizer.zero_grad()

        # compute output logits
        logits, _ = model(input_ids) # TODO: WE ONLY NEED TARGET IDS FOR TRANSFORMERE

        # apply cross entropy
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        loss.backward()
        optimizer.step()

        # Step the scheduler based on validation loss
        #scheduler.step(loss)

        total_train_loss += loss.item()
      avg_train_loss = total_train_loss / len(train_loader)
      scheduler.step(avg_train_loss)
      train_losses.append(avg_train_loss)

      # validation
      model.eval()
      with torch.inference_mode():
        for input_ids, target_ids in val_loader:
          input_ids = input_ids.to("mps")
          target_ids = target_ids.to("mps")

          logits, _ = model(input_ids)

          val_loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

          total_val_loss += val_loss.item()

      avg_val_loss = total_val_loss / len(val_loader)
      val_losses.append(avg_val_loss)

      print(f"Epoch {ep+1}, Training Loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")
    
    # Calculate training time
    train_time_end = timer()
    self.print_train_time(train_time_start, train_time_end)
    return train_losses, val_losses


  def evalutate_model(self, model: torch.nn.Module):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=3)
    loss = 0
    vocab_size = 10000
    # load in out tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("bpe_tokenizer.model")
    tokenizer = sp
    vocab_size = tokenizer.get_piece_size() # this gets the vocabulary size
    test_dataset = TextDataSet("data/test.jsonl", tokenizer, 50) # 128 = max sequence length
    data_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=test_dataset.collate_fn)

    model.eval()
    all_references = []  # List of lists of reference token lists
    all_hypotheses = []  # List of hypothesis token lists
    with torch.inference_mode():
      for input_ids, target_ids in data_loader:
        input_ids = input_ids.to("mps")
        target_ids = target_ids.to("mps")

        logits, _ = model(input_ids)

        # Convert logits to predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode target and predicted IDs back to text
        for target, predicted in zip(target_ids, predicted_ids):
          reference = tokenizer.decode(target.cpu().tolist(), out_type=str).split()
          hypothesis = tokenizer.decode(predicted.cpu().tolist(), out_type=str).split()
            
          # Append references and hypotheses for BLEU computation
          all_references.append([reference])  # BLEU expects a list of references

          all_hypotheses.append(hypothesis)

        loss += criterion(logits.view(-1, vocab_size), target_ids.view(-1))

      bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=SmoothingFunction().method4)
      loss /= len(data_loader)
      perplexity = torch.exp(loss)

      print(f"Perplexity: {perplexity} | Bleu score: {bleu_score}")


  def print_train_time(self, start: float, end: float):
    """Prints difference between start and end time."""
    total_time = end-start
    print(f"\nTrain time: {total_time:.3f} seconds")
    # return total_time