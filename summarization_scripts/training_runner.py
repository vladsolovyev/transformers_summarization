import gc
from datetime import datetime

import pandas as pd
import torch
from GPUtil import showUtilization
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import train_test_split

from summarization_mbart import MBartSummarizationModel
from transformers import MBartTokenizer

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512
dataset_languages = ["english", "spanish", "russian"]
model_languages = ["en_XX", "es_XX", "ru_RU"]
tokenized_datasets = []
metrics = dict()


def save_metrics(output_dir):
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv("{}/metrics.csv".format(output_dir))


def free_memory():
    showUtilization()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    showUtilization()


def preprocess_function(dataset_split, tokenizer):
    model_inputs = tokenizer(dataset_split["text"],
                             max_length=MAX_INPUT_LENGTH,
                             truncation=True)
    summaries = tokenizer(text_target=dataset_split["target"],
                          max_length=MAX_TARGET_LENGTH,
                          truncation=True)
    model_inputs["labels"] = summaries["input_ids"]
    return model_inputs


def initialize_xlsum_dataset(tokenizer):
    for dataset_language, model_language in zip(dataset_languages, model_languages):
        dataset = load_dataset("GEM/xlsum", dataset_language, cache_dir="./cache")
        dataset.pop("validation")
        tokenize_dataset(dataset, model_language, tokenizer)


def initialize_wikilingua_datasets(tokenizer):
    for dataset_language, model_language in zip(dataset_languages, model_languages):
        wikilingua_dataset = dict({"text": list(), "target": list()})
        dataset = load_dataset("wiki_lingua", name=dataset_language, cache_dir="./cache")
        for article in dataset["train"]["article"]:
            for document, summary in zip(article["document"], article["summary"]):
                wikilingua_dataset["text"].append(document)
                wikilingua_dataset["target"].append(summary)
        wikilingua_dataset_df = pd.DataFrame(wikilingua_dataset)
        train, test = train_test_split(wikilingua_dataset_df, test_size=0.1, shuffle=True, random_state=42)
        split_dataset = DatasetDict({"train": Dataset.from_pandas(train), "test": Dataset.from_pandas(test)})
        tokenize_dataset(split_dataset, model_language, tokenizer)


def tokenize_dataset(dataset, model_language, tokenizer):
    tokenizer.src_lang = model_language
    tokenizer.tgt_lang = model_language
    for split in dataset:
        dataset[split] = dataset[split].filter(
            lambda sample: len(sample["text"].split()) > 20 and len(sample["target"].split()) > 10)
    tokenized_datasets.append(dataset.map(lambda sample: preprocess_function(sample, tokenizer), batched=True))
    del dataset


def train_and_evaluate_models(freeze_embeddings=False, initialize_dataset=initialize_xlsum_dataset):
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", model_max_length=MAX_INPUT_LENGTH,
                                               cache_dir="./cache")
    initialize_dataset(tokenizer)
    del tokenizer
    free_memory()
    output_dir = datetime.today().strftime('%Y-%m-%d')

    # every language separately
    for tokenized_dataset, model_language in zip(tokenized_datasets, model_languages):
        model = MBartSummarizationModel(src_lang=model_language,
                                        tgt_lang=model_language,
                                        output_dir="{}/{}".format(output_dir, model_language),
                                        freeze_embeddings=freeze_embeddings)
        model.train(tokenized_dataset["train"],
                    save_model=(model_language == "en_XX"))
        metrics[model_language] = model.test_predictions(tokenized_dataset["test"])
        save_metrics(output_dir)
        del model
        free_memory()

    # zero shot. Evaluate spanish and russian datasets using english model
    for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
        model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                        src_lang=model_language,
                                        tgt_lang=model_language,
                                        output_dir="{}/en_XX_zero_{}".format(output_dir, model_language))
        metrics["en_XX_zero_{}".format(model_language)] = model.test_predictions(tokenized_dataset["test"])
        save_metrics(output_dir)
        del model
        free_memory()

    # few shot experiments. Tune english model using few data from spanish and russian datasets
    for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
        for data_size in [10, 100, 1000, 10000]:
            model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                            src_lang=model_language,
                                            tgt_lang=model_language,
                                            output_dir="{}/en_XX_tuned_{}_{}".format(output_dir, model_language,
                                                                                     data_size),
                                            freeze_embeddings=freeze_embeddings)
            model.train(Dataset.from_dict(tokenized_dataset["train"][:data_size]))
            metrics["en_XX_tuned_{}_{}".format(model_language, data_size)] = \
                model.test_predictions(tokenized_dataset["test"])
            save_metrics(output_dir)
            del model
            free_memory()

    # tune english model using complete data from spanish and russian datasets
    for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
        model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                        src_lang=model_language,
                                        tgt_lang=model_language,
                                        output_dir="{}/en_XX_tuned_{}".format(output_dir, model_language),
                                        freeze_embeddings=freeze_embeddings)
        model.train(tokenized_dataset["train"])
        metrics["en_XX_tuned_{}".format(model_language)] = model.test_predictions(tokenized_dataset["test"])
        save_metrics(output_dir)
        del model
        free_memory()

    # all three languages together
    multilingual_dataset = \
        concatenate_datasets([tokenized_dataset["train"] for tokenized_dataset in tokenized_datasets]).shuffle(seed=42)
    model = MBartSummarizationModel(output_dir="{}/multilingual".format(output_dir), freeze_embeddings=freeze_embeddings)
    model.train(multilingual_dataset, save_model=True)
    del model
    free_memory()

    for tokenized_dataset, model_language in zip(tokenized_datasets, model_languages):
        model = MBartSummarizationModel(model_name="{}/multilingual".format(output_dir),
                                        src_lang=model_language,
                                        tgt_lang=model_language,
                                        output_dir="{}/multilingual_{}".format(output_dir, model_language))
        metrics["multilingual_{}".format(model_language)] = model.test_predictions(tokenized_dataset["test"])
        save_metrics(output_dir)
        del model
        free_memory()

    save_metrics(output_dir)


if __name__ == "__main__":
    train_and_evaluate_models(freeze_embeddings=False, initialize_dataset=initialize_xlsum_dataset)
