import re

from transformers import GPT2Tokenizer

# Lam sach du lieu
def clean_text(text):
    # Loai bo ky tu dac biet va dau cau khong can thiet
    text = re.sub(r'[^\w\s,]', '', text)

    # Chuan hoa khoang trang
    text = re.sub(r'\s+', ' ', text).strip()

    # Chuyen thanh chu thuong
    text = text.lower()

    return text

# Phan tach cau thanh cac nhip
def split_into_phrases(text):
    # Ngat nhip bang dau phay
    phrases = text.split(', ')
    return phrases

# Tao cap du lieu
def create_pairs(phrases):
    pairs = []
    for phrase in phrases:
        for i in range(len(phrase) - 1):
            input_text = phrase[i]
            target_text = phrase[i + 1]
            pairs.append((input_text, target_text))
    return pairs


# Tai tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize_text(pairs):
    tokenized_data = []
    for input_text, target_text in pairs:
        input_ids = tokenizer.encode(input_text)
        output_ids = tokenizer.encode(target_text)
        tokenized_data.append((input_ids, output_ids))
    return tokenized_data


if __name__ == '__main__':
    with open('data.txt', 'r', encoding='utf-8') as f:
        lyrics = f.readlines()

    cleaned_text = [clean_text(line) for line in lyrics]
    print(cleaned_text)

    phrases = [split_into_phrases(line) for line in cleaned_text]
    print(phrases)

    pairs = create_pairs(phrases)
    print(pairs)

    tokenized_data = tokenize_text(pairs)
    print(tokenized_data)