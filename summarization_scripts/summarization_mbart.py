import evaluate
import nltk
import numpy as np
from datasets import load_dataset

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MBartTokenizer, MBartForConditionalGeneration

nltk.download('punkt')


# delete it later
def save_labels(labels):
    with open("labels.txt", "w") as file:
        file.write("Labels < 0\n")
        file.write("Indices:\n")
        file.write("{}\n".format(np.argwhere(labels < 0)))
        file.write("{}\n".format(labels[labels < 0]))
        file.write("Labels >= 0\n")
        file.write("Indices:\n")
        file.write("{}\n".format(np.argwhere(labels >= 0)))
        file.write("{}\n".format(labels[labels >= 0]))


class MBartModel:
    def __init__(self,
                 model_name="facebook/mbart-large-cc25",
                 max_input_length=1024,
                 max_target_length=512,
                 src_lang="en_XX",
                 tgt_lang="en_XX"):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.metric = evaluate.load("rouge")
        self.tokenizer = MBartTokenizer.from_pretrained(model_name, model_max_length=max_input_length,
                                                        src_lang=src_lang, tgt_lang=tgt_lang)
        self.summarization_trainer = None

    def preprocess_function(self, data):
        model_inputs = self.tokenizer(data["text"],
                                      max_length=self.max_input_length,
                                      truncation=True)

        summaries = self.tokenizer(text_target=data["target"],
                                   max_length=self.max_target_length,
                                   truncation=True)

        model_inputs["labels"] = summaries["input_ids"]
        return model_inputs

    def decode_labels(self, predictions, labels):
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # delete it later. Just to check if -100 is possible in mbart
        save_labels(labels)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        return decoded_preds, decoded_labels

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds, decoded_labels = self.decode_labels(predictions, labels)

        result = self.metric.compute(predictions=decoded_preds,
                                     references=decoded_labels,
                                     use_stemmer=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def create_training_arguments(self):
        batch_size = 4
        model_name = self.model_name.split("/")[-1]
        return Seq2SeqTrainingArguments(
            "{}-finetuned-xlsum".format(model_name),
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
            fp16=False
        )

    def create_summarization_trainer(self, tokenized_datasets):
        summarization_model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        summarization_model.config.decoder_start_token_id = \
            self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=summarization_model)
        return Seq2SeqTrainer(
            summarization_model,
            self.create_training_arguments(),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

    def train(self):
        data_es = load_dataset("GEM/xlsum", "spanish")

        data_en = load_dataset("GEM/xlsum", "english")
        data_ru = load_dataset("GEM/xlsum", "russian")

        data = data_en
        tokenized_datasets = data.map(self.preprocess_function, batched=True)
        self.summarization_trainer = self.create_summarization_trainer(tokenized_datasets)
        self.summarization_trainer.train()


if __name__ == '__main__':
    model = MBartModel()
    model.train()
