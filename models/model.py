import pickle

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Tải lại dữ liệu từ file pickle
def load_data_from_pickle(train_file='train_data.pkl', test_file='test_data.pkl'):
    with open(train_file, 'rb') as f_train:
        train_data = pickle.load(f_train)
    with open(test_file, 'rb') as f_test:
        test_data = pickle.load(f_test)
    return train_data, test_data


if __name__ == '__main__':
    # Tải mô hình và tokenizer
    model_name = 'VietAI/vit5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name, use_auth_token=True)
    print(model)

    train_data, test_data = load_data_from_pickle()

