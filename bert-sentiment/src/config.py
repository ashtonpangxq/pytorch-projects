import transformers

TOKENIZER = transformers.BertTokenizer.from_pretrained(pretrained_model_name_or_path="../data/bert-based-uncased", do_lower_case=True)
