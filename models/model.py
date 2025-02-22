from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Tải mô hình và tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, use_auth_token=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

print(model, tokenizer, sep='\n')