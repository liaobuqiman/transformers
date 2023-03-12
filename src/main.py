from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert import BertForQuestionAnswering, BertTokenizerFast, BertConfig


from test_code.dataset import QA_Dataset
from test_code.parse_arg import parse_args
from test_code.training import train
from test_code.utilities import read_data, same_seeds


def main(seed, batch_size, learning_rate, device, num_epoch, n_workers, logging_step, warmup_steps):
    same_seeds(seed)

    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    fp16_training = False
    if fp16_training:
        from accelerate import Accelerator
        accelerator = Accelerator(fp16=True)
        device = accelerator.device

    # import model
    # model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    config = BertConfig()
    model = BertForQuestionAnswering(config).to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # read data
    train_questions, train_paragraphs = read_data("hw7_train.json")
    dev_questions, dev_paragraphs = read_data("hw7_dev.json")
    test_questions, test_paragraphs = read_data("hw7_test.json")

    # Tokenize questions and paragraphs separately
    # 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions],
                                          add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions],
                                        add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions],
                                         add_special_tokens=False)

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
    # Note: Do NOT change batch size of dev_loader / test_loader !
    # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=True)


    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # Total number of training steps
    total_steps = len(train_loader) * num_epoch
    print('total_steps', total_steps)
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    train(
        train_loader=train_loader,
        dev_loader=dev_loader,
        dev_questions=dev_questions,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        fp16_training=fp16_training,
        device=device,
        num_epoch=num_epoch,
        logging_step=logging_step,
        validation=True
    )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(**parse_args())