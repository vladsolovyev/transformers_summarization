import evaluate
import nltk
import numpy as np
from torch import nn

from utils.multilingual_tokenizer import MultilingualTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    MBartTokenizer, MBartForConditionalGeneration

nltk.download('punkt')


def freeze_embeds(model):
    freeze_params(model.model.shared)
    for d in [model.model.encoder, model.model.decoder]:
        freeze_params(d.embed_positions)
        freeze_params(d.embed_tokens)


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def calculate_rouge_score(predictions, references, language):
    return evaluate.load("rouge", cache_dir="./cache").compute(predictions=predictions,
                                                               references=references,
                                                               tokenizer=MultilingualTokenizer(language=language,
                                                                                               use_stemmer=True).tokenize)


def calculate_bert_score(predictions, references):
    bert_result = evaluate.load("bertscore", cache_dir="./cache").compute(predictions=predictions,
                                                                          references=references,
                                                                          model_type="bert-base-multilingual-cased")
    if bert_result["hashcode"]:
        del bert_result["hashcode"]
    return {"bert_score_{}".format(k): np.mean(v) for k, v in bert_result.items()}


def create_training_arguments(output_dir):
    batch_size = 4
    return Seq2SeqTrainingArguments(
        output_dir,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=100,
        do_train=True,
        do_eval=False,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        predict_with_generate=True,
        weight_decay=0.01,
        num_train_epochs=1,
        fp16=True,
        overwrite_output_dir=True
    )


class MBartSummarizationModel:
    def __init__(self,
                 model_name="facebook/mbart-large-cc25",
                 max_input_length=1024,
                 max_target_length=512,
                 src_lang="en_XX",
                 tgt_lang="en_XX",
                 output_dir=None,
                 freeze_embeddings=False):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", model_max_length=max_input_length,
                                                        src_lang=src_lang, tgt_lang=tgt_lang, cache_dir="./cache")
        self.summarization_model = MBartForConditionalGeneration.from_pretrained(self.model_name, cache_dir="./cache")
        self.summarization_model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
        self.summarization_model.config.forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
        self.summarization_model.tokenizer = self.tokenizer
        self.summarization_model.output_dir = output_dir
        if freeze_embeddings:
            freeze_embeds(self.summarization_model)

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

        results = calculate_rouge_score(decoded_preds, decoded_labels, language=self.tokenizer.tgt_lang) \
                  | calculate_bert_score(decoded_preds, decoded_labels)
        result = {key: value * 100 for key, value in results.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self, train_data, save_model=False):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.summarization_model)
        trainer = Seq2SeqTrainer(
            self.summarization_model,
            create_training_arguments(self.summarization_model.output_dir),
            train_dataset=train_data,
            data_collator=data_collator
        )
        trainer.train(resume_from_checkpoint=False)
        if save_model:
            trainer.save_model(self.summarization_model.output_dir)

    def test_predictions(self, test_data):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.summarization_model)
        trainer = Seq2SeqTrainer(
            self.summarization_model,
            create_training_arguments(self.summarization_model.output_dir),
            data_collator=data_collator,
            compute_metrics=self.calculate_metrics
        )
        test_results = trainer.predict(
            test_data,
            metric_key_prefix="test",
            max_length=self.max_target_length,
            num_beams=5
        )
        trainer.log_metrics("test", test_results.metrics)
        trainer.save_metrics("test", test_results.metrics)
        return test_results.metrics
