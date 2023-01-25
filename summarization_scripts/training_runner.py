import gc
from datetime import datetime

import pandas as pd
import torch
from GPUtil import showUtilization
from datasets import load_dataset, Dataset

from summarization_mbart import MBartSummarizationModel
from transformers import MBartTokenizer

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", model_max_length=MAX_INPUT_LENGTH)
dataset_languages = ["english", "spanish", "russian"]
model_languages = ["en_XX", "es_XX", "ru_RU"]
tokenized_datasets = []
metrics = dict()


def free_memory():
    showUtilization()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    showUtilization()


def preprocess_function(dataset_split):
    tokenizer.src_lang = model_language
    tokenizer.tgt_lang = model_language
    model_inputs = tokenizer(dataset_split["text"],
                             max_length=MAX_INPUT_LENGTH,
                             truncation=True)

    summaries = tokenizer(text_target=dataset_split["target"],
                          max_length=MAX_TARGET_LENGTH,
                          truncation=True)

    model_inputs["labels"] = summaries["input_ids"]
    return model_inputs


for dataset_language, model_language in zip(dataset_languages, model_languages):
    dataset = load_dataset("GEM/xlsum", dataset_language)
    tokenized_datasets.append(dataset.map(preprocess_function, batched=True))
    del dataset

del tokenizer
free_memory()
output_dir = datetime.today().strftime('%Y-%m-%d')

# every language separately
for tokenized_dataset, model_language in zip(tokenized_datasets, model_languages):
    model = MBartSummarizationModel(src_lang=model_language,
                                    tgt_lang=model_language,
                                    output_dir="{}/{}".format(output_dir, model_language))
    model.train(tokenized_dataset["train"],
                tokenized_dataset["validation"],
                save_model=(model_language == "en_XX"))
    metrics[model_language] = model.test_predictions(tokenized_dataset["test"])
    del model
    free_memory()

# zero shot. Evaluate spanish and russian datasets using english model
for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
    model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                    src_lang=model_language,
                                    tgt_lang=model_language,
                                    output_dir="{}/en_XX_zero_{}".format(output_dir, model_language))
    metrics["en_XX_zero_{}".format(model_language)] = model.test_predictions(tokenized_dataset["test"])
    del model
    free_memory()

# few shot experiments. Tune english model using few data from spanish and russian datasets
for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
    for data_size in [10, 100, 1000, 10000]:
        model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                        src_lang=model_language,
                                        tgt_lang=model_language,
                                        output_dir="{}/en_XX_tuned_{}_{}".format(output_dir, model_language, data_size))
        model.train(Dataset.from_dict(tokenized_dataset["train"][:data_size]),
                    Dataset.from_dict(tokenized_dataset["validation"][:data_size]))
        metrics["en_XX_tuned_{}_{}".format(model_language, data_size)] = \
            model.test_predictions(Dataset.from_dict(tokenized_dataset["test"][:data_size]))
        del model
        free_memory()

# tune english model using complete data from spanish and russian datasets
for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
    model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                    src_lang=model_language,
                                    tgt_lang=model_language,
                                    output_dir="{}/en_XX_tuned_{}".format(output_dir, model_language))
    model.train(tokenized_dataset["train"],
                tokenized_dataset["validation"],
                save_model=(model_language == "es_XX"))
    metrics["en_XX_tuned_{}".format(model_language)] = model.test_predictions(tokenized_dataset["test"])
    del model
    free_memory()

# all three languages together
model = MBartSummarizationModel(model_name="{}/en_XX_tuned_es_XX".format(output_dir),
                                src_lang="ru_RU",
                                tgt_lang="ru_RU",
                                output_dir="{}/en_XX_tuned_es_XX_and_ru_RU".format(output_dir))
model.train(tokenized_datasets[2]["train"],
            tokenized_datasets[2]["validation"],
            save_model=True)
del model
free_memory()

for tokenized_dataset, model_language in zip(tokenized_datasets, model_languages):
    model = MBartSummarizationModel(model_name="{}/en_XX_tuned_es_XX_and_ru_RU".format(output_dir),
                                    src_lang=model_language,
                                    tgt_lang=model_language,
                                    output_dir="{}/en_XX_tuned_es_XX_and_ru_RU_{}".format(output_dir, model_language))
    metrics["en_XX_tuned_es_XX_and_ru_RU_{}".format(model_language)] = model.test_predictions(tokenized_dataset["test"])
    del model
    free_memory()

metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
metrics_df.to_csv("{}/metrics.csv".format(output_dir))
