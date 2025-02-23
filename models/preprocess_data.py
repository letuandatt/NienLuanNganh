import re
import pickle

from transformers import T5Tokenizer
from sklearn.model_selection import train_test_split

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
    # Ngat nhip bang cac dau ngat cau nhu dau cham, phay
    phrases = re.split(r'[.,]', text)  # Tách theo dấu chấm và dấu phẩy
    phrases = [phrase.strip() for phrase in phrases if phrase]  # Xóa khoảng trắng và bỏ phần rỗng
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
tokenizer = T5Tokenizer.from_pretrained('VietAI/vit5-base')

def tokenize_text(pairs):
    tokenized_data = []
    for input_text, target_text in pairs:
        # Tokenize đầu vào và mục tiêu bằng tokenizer của ViT5
        input_ids = tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=20, return_tensors='pt')
        output_ids = tokenizer.encode(target_text, truncation=True, padding='max_length', max_length=20, return_tensors='pt')
        tokenized_data.append((input_ids, output_ids))
    return tokenized_data


# Lưu = pickle
def save_data_to_pickle(train_data, test_data, train_file='train_data.pkl', test_file='test_data.pkl'):
    with open(train_file, 'wb') as f_train:
        pickle.dump(train_data, f_train)
    with open(test_file, 'wb') as f_test:
        pickle.dump(test_data, f_test)
    print('Train data saved to pickle file.')
    return


if __name__ == '__main__':
    with open('data.txt', 'r', encoding='utf-8') as f:
        lyrics = f.readlines()

    cleaned_text = [clean_text(line) for line in lyrics]
    print(f"Cleaned text: {cleaned_text}")

    phrases = [split_into_phrases(line) for line in cleaned_text]
    print(f"Phrases: {phrases}")

    pairs = create_pairs(phrases)
    print(f"Pairs: {pairs}")

    tokenized_data_ = tokenize_text(pairs)
    print(f"Tokenized text: {tokenized_data_}")

    train_data, test_data = train_test_split(tokenized_data_, test_size=0.2, random_state=42)

    save_data_to_pickle(train_data, test_data)

