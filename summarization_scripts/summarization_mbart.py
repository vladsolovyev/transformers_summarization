import evaluate
import nltk
import numpy as np
from datasets import load_dataset

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MBartTokenizer, MBartForConditionalGeneration

nltk.download('punkt')

data_es = load_dataset("GEM/xlsum", "spanish")
data_en = load_dataset("GEM/xlsum", "english")
data_ru = load_dataset("GEM/xlsum", "russian")


def calculate_rouge_score(predictions, references):
    return evaluate.load("rouge").compute(predictions=predictions,
                                          references=references,
                                          use_stemmer=True)


def calculate_bert_score(predictions, references):
    bert_result = evaluate.load("bertscore").compute(predictions=predictions,
                                                     references=references,
                                                     model_type="bert-base-multilingual-cased")
    if bert_result["hashcode"]:
        del bert_result["hashcode"]
    return {"bert_score_{}".format(k): np.mean(v) for k, v in bert_result.items()}


class MBartModel:
    def __init__(self,
                 dataset,
                 model_name="facebook/mbart-large-cc25",
                 max_input_length=1024,
                 max_target_length=512,
                 src_lang="en_XX",
                 tgt_lang="en_XX"):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = MBartTokenizer.from_pretrained(model_name, model_max_length=max_input_length,
                                                        src_lang=src_lang, tgt_lang=tgt_lang)
        self.summarization_trainer = None
        self.tokenized_datasets = dataset.map(self.preprocess_function, batched=True)

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

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        return decoded_preds, decoded_labels

    def calculate_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds, decoded_labels = self.decode_labels(predictions, labels)

        results = calculate_rouge_score(decoded_preds, decoded_labels) | calculate_bert_score(decoded_preds,
                                                                                              decoded_labels)
        result = {key: value * 100 for key, value in results.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def create_training_arguments(self):
        batch_size = 4
        model_name = self.model_name.split("/")[-1]
        return Seq2SeqTrainingArguments(
            "{}-finetuned-xlsum".format(model_name),
            evaluation_strategy="epoch",
            do_train=True,
            do_eval=True,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
            fp16=False,
            overwrite_output_dir=True
        )

    def train(self):
        summarization_model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        summarization_model.config.decoder_start_token_id = \
            self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
        summarization_model.config.forced_bos_token_id = \
            self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=summarization_model)
        self.summarization_trainer = Seq2SeqTrainer(
            summarization_model,
            self.create_training_arguments(),
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.calculate_metrics
        )
        self.summarization_trainer.train(resume_from_checkpoint=False)

    def test_predictions(self):
        test_results = self.summarization_trainer.predict(
            self.tokenized_datasets["test"],
            metric_key_prefix="test",
            max_length=self.max_target_length,
            num_beams=5
        )
        self.summarization_trainer.log_metrics("test", test_results.metrics)
        self.summarization_trainer.save_metrics("test", test_results.metrics)


if __name__ == '__main__':
    model = MBartModel(data_en)
    model.train()
    model.test_predictions()
