from typing import Tuple
import sentencepiece as spm
import os

def add_special_tokens(pairs: Tuple[list]):
  """
  Insert <bos> and <eos> special tokens into a dataset
  :param pairs: original prompts and completions
  :return: prompts / completion pairs with special tokens inserted
  """

  new_prompts = []
  new_completions = []

  # NOTE: Removed zip(pairs) here because it wouldn't work. but may need it in the future
  for prompt, completion in pairs:
    # If the beginning of the prompt is upper case, then we assume it is the start of a sequence
    if prompt[0].isupper():
      prompt = "<bos>" + prompt

    # If the end of the completion is a terminating punctuation, then we assume it is the end of a sequence
    if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
      completion += "<eos>"
    new_prompts.append(prompt)
    new_completions.append(completion)

  return new_prompts, new_completions

# Merge all text files into a single corpus
def merge_text_files(data_dir=os.getcwd()+"/data/raw/", output_file="corpus.txt"):
  """
  This will merge all textual data in a directory into a single corpus
  :param data_dir: path to the directory containing the raw text files
  :param output_file: path to file where corpus will be saved 
  """

  # open new file
  # TODO: not sure if this is right or not. Just using a single file for now
  with open(output_file, "w", encoding="utf-8") as outfile:
    # iterate through all files in the directory
    for filename in os.listdir(data_dir):
      if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as infile:
          # write the contents of the file to the new file
          outfile.write(infile.read())
          outfile.write("\n")

  if True:#__name__ == "__main__":
   # DATA_DIR = "./data/raw" # path to the directory containing the raw text files
    TOKENIZER_PREFIX = 'bpe_tokenizer' # this will be used for naming the tokenizer
    VOCAB_SIZE = 10000 # stopping condition for tokenizing
    CORPUS_FILE = "corpus.txt" # path to new combined corpus file
    #merge_text_files(DATA_DIR, CORPUS_FILE)

    # Train the tokenizer with special tokens
    spm.SentencePieceTrainer.Train(
      input=CORPUS_FILE,
      model_prefix=TOKENIZER_PREFIX,
      vocab_size=VOCAB_SIZE,
      bos_id=1, # this is set to 1 because 0 is <unk>
      eos_id=2,
      pad_id=3,
      user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"])
    )
    
  print("Tokenizer trained successfully!")
