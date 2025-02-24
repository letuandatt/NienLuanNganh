import pickle
import torch
import evaluate

from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments


# Tải lại dữ liệu từ file pickle
def load_data_from_pickle(train_file='train_data.pkl', test_file='test_data.pkl'):
    """Hàm này dùng để tải dữ liệu từ các file pickle đã lưu trước đó."""
    with open(train_file, 'rb') as f_train:
        train_data = pickle.load(f_train)
    with open(test_file, 'rb') as f_test:
        test_data = pickle.load(f_test)
    return train_data, test_data


# Định nghĩa class Dataset để chuẩn bị dữ liệu cho Trainer
class LyricsDataset(Dataset):
    def __init__(self, tokenized_data):
        """
        Khởi tạo Dataset với dữ liệu đã được mã hóa (tokenized).
        tokenized_data là một danh sách của các cặp input_ids và output_ids.
        """
        self.tokenized_data = tokenized_data

    def __len__(self):
        """Hàm trả về độ dài của Dataset, cần thiết cho quá trình batching."""
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        """
        Hàm lấy một mẫu dữ liệu tại vị trí idx trong Dataset.
        Trả về input_ids, labels (output_ids) và decoder_input_ids.
        """
        input_ids = self.tokenized_data[idx][0].squeeze(0)  # Loại bỏ chiều thừa
        output_ids = self.tokenized_data[idx][1].squeeze(0)

        # Tạo decoder_input_ids bằng cách dịch output_ids sang trái một bước (shift left)
        decoder_input_ids = output_ids.clone()
        decoder_input_ids = torch.cat([torch.tensor([tokenizer.pad_token_id]), decoder_input_ids[:-1]])

        return {
            'input_ids': input_ids,
            'labels': output_ids,  # Mô hình T5 sử dụng 'labels' để tính toán loss
            'decoder_input_ids': decoder_input_ids
        }


# Hàm để sinh văn bản từ mô hình đã huấn luyện
def generate_text(input_text, max_length=20, temperature=0.7, num_beams=4):
    """
    Hàm sinh văn bản từ input_text.
    max_length: giới hạn độ dài của văn bản sinh ra.
    temperature: điều chỉnh tính "sáng tạo" của mô hình.
    num_beams: số lượng beam trong beam search để tìm kết quả tốt nhất.
    """
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True)
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=num_beams, temperature=temperature,
                                early_stopping=True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# Tính các chỉ số đánh giá (metrics) như BLEU, ROUGE
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """Hàm tính toán các chỉ số đánh giá dựa trên dự đoán và nhãn thật."""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result


if __name__ == '__main__':
    # Tải mô hình ViT5 và tokenizer từ Huggingface
    model_name = 'VietAI/vit5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name, use_auth_token=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Tải dữ liệu train và test từ các file pickle
    train_data, test_data = load_data_from_pickle()

    # Chuẩn bị dataset cho Trainer
    train_dataset = LyricsDataset(train_data)
    test_dataset = LyricsDataset(test_data)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Thiết lập các tham số huấn luyện
    training_args = TrainingArguments(
        output_dir='./results',  # Thư mục lưu kết quả huấn luyện
        per_device_train_batch_size=32,  # Kích thước batch khi train
        per_device_eval_batch_size=32,  # Kích thước batch khi đánh giá
        num_train_epochs=50,  # Số lượng epoch
        logging_dir='./logs',  # Thư mục lưu logs
        eval_strategy='steps',  # Đánh giá sau mỗi một số bước nhất định
        logging_steps=100,  # Ghi logs sau mỗi 100 bước
        save_steps=500,  # Lưu mô hình sau mỗi 500 bước
        eval_steps=500  # Đánh giá sau mỗi 500 bước
    )

    # Tạo Trainer với mô hình, dữ liệu và các tham số đã thiết lập
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Tính các chỉ số đánh giá
    )

    # Bắt đầu huấn luyện mô hình
    trainer.train()

    # Lưu mô hình và tokenizer sau khi huấn luyện
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')
