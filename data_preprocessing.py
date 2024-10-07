import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

batch_size = 32 
spacy_english = spacy.load("en_core_web_sm")
spacy_german = spacy.load("de_core_news_sm")

def tokenize_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]

def tokenize_german(text):
    return [token.text for token in spacy_german.tokenizer(text)]


german = Field(init_token = "<sos>", eos_token = "<eos>", lower = True, tokenize = tokenize_german, batch_first = True, fix_length = 30)
english = Field(init_token = "<sos>", eos_token = "<eos>", lower = True, tokenize = tokenize_english, batch_first = True, fix_length = 30)

train_data, dev_data, test_data = Multi30k.splits(exts = (".de", ".en"), fields = (german, english), root = "/content//Transformer/data")

train_iterator, dev_iterator, test_iterator = BucketIterator.splits((train_data, dev_data, test_data), batch_size = batch_size,
                                                                    sort_key = lambda x: len(x.src))

german.build_vocab(train_data, min_freq = 3, max_size = 10000)
english.build_vocab(train_data, min_freq = 3, max_size = 10000)
