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

# Tạo cặp dữ liệu
themes = {
    "Bài hát về tình yêu đôi lứa": ["yêu", "thương", "vấn vương", "hẹn", "đợi", "nhớ", "thầm"],
    "Bài hát về thiên nhiên và phong cảnh đồng quê": ["nắng", "gió", "mưa", "trắng", "cây", "bông", "nước"],
    "Bài hát về sự nhớ nhung và xa cách": ["nhớ", "xa cách", "xa", "thương nhớ", "vấn vương", "đợi chờ"],
    "Bài hát về sự lãng mạn và hẹn hò": ["lãng mạn", "hẹn", "tình yêu", "hẹn hò"],
    "Bài hát về sự buồn bã và cô đơn": ["buồn", "cô đơn", "đau", "vắng lặng"],
    "Bài hát về sự chờ đợi và hy vọng": ["chờ", "hy vọng", "niềm tin", "tương lai", "mong", "đón"]
}

def add_themes(text, themes):
    pairs = []

    for theme, keywords in themes.items():
        if any(keyword in text for keyword in keywords):
            pairs.append((theme, text))

    return pairs

# Phan tach cau thanh cac nhip
def split_into_phrases(text):
    # Ngat nhip bang cac dau ngat cau nhu dau cham, phay
    phrases = re.split(r'[.,]', text)  # Tách theo dấu chấm và dấu phẩy
    phrases = [phrase.strip() for phrase in phrases if phrase]  # Xóa khoảng trắng và bỏ phần rỗng
    return phrases

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
    with open('../data/data.txt', 'r', encoding='utf-8') as f:
        lyrics = f.readlines()

    cleaned_text = [clean_text(line) for line in lyrics]
    print(f"Cleaned text: {cleaned_text}")

    add_themes_lyrics = [add_themes(line, themes) for line in cleaned_text]

    for lyrics in add_themes_lyrics:
        print(lyrics)

    # pairs = add_themes(phrases, themes)
    # print(f"Pairs: {pairs}")

    # tokenized_data_ = tokenize_text(pairs)
    # print(f"Tokenized text: {tokenized_data_}")

    # with open('data_after_tokenize.pkl', 'wb') as f_save:
    #     pickle.dump(tokenized_data_, f_save)
    #
    # train_data, test_data = train_test_split(tokenized_data_, test_size=0.2, random_state=42)
    #
    # save_data_to_pickle(train_data, test_data)

