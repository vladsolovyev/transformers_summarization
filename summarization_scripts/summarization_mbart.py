import evaluate
import nltk
import numpy as np

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MBartTokenizer, MBartForConditionalGeneration

nltk.download('punkt')


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


def create_training_arguments(output_dir):
    batch_size = 4
    return Seq2SeqTrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False,
        overwrite_output_dir=True
    )


class MBartSummarizationModel:
    def __init__(self,
                 model_name="facebook/mbart-large-cc25",
                 max_input_length=1024,
                 max_target_length=512,
                 src_lang="en_XX",
                 tgt_lang="en_XX",
                 output_dir=None):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", model_max_length=max_input_length,
                                                        src_lang=src_lang, tgt_lang=tgt_lang)
        self.summarization_model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.summarization_model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
        self.summarization_model.config.forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
        self.summarization_model.tokenizer = self.tokenizer
        self.summarization_model.output_dir = output_dir
        self.summarization_trainer = None

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

    def train(self, train_data, validation_data, output_dir, save_model=False):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.summarization_model)
        self.summarization_trainer = Seq2SeqTrainer(
            self.summarization_model,
            create_training_arguments(output_dir),
            train_dataset=train_data,
            eval_dataset=validation_data,
            data_collator=data_collator,
            compute_metrics=self.calculate_metrics
        )
        self.summarization_trainer.train(resume_from_checkpoint=False)
        if save_model:
            self.summarization_trainer.save_model(output_dir)

    def test_predictions(self, test_data):
        test_results = self.summarization_trainer.predict(
            test_data,
            metric_key_prefix="test",
            max_length=self.max_target_length,
            num_beams=5
        )
        self.summarization_trainer.log_metrics("test", test_results.metrics)
        self.summarization_trainer.save_metrics("test", test_results.metrics)
        return test_results.metrics
