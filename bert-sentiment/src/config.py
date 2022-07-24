import transformers

def tokenizer():
    return transformers.BertTokenizer.from_pretrained(pretrained_model_name_or_path="../data/bert-based-uncased", do_lower_case=True)
