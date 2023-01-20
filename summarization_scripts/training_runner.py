from datetime import datetime

from datasets import load_dataset, Dataset
from transformers import MBartTokenizer

from summarization_mbart import MBartSummarizationModel

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", model_max_length=MAX_INPUT_LENGTH)
dataset_languages = ["english", "spanish", "russian"]
model_languages = ["en_XX", "es_XX", "ru_RU"]
tokenized_datasets = []


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

output_dir = datetime.today().strftime('%Y-%m-%d')

# every language separately
for tokenized_dataset, model_language in zip(tokenized_datasets, model_languages):
    model = MBartSummarizationModel(src_lang=model_language, tgt_lang=model_language)
    model.train(tokenized_dataset["train"],
                tokenized_dataset["validation"],
                "{}/{}".format(output_dir, model_language),
                save_model=(model_language == "en_XX"))
    model.test_predictions(tokenized_dataset["test"])

# zero shot. Evaluate spanish and russian datasets using english model
for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
    model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                    src_lang=model_language,
                                    tgt_lang=model_language,
                                    output_dir="{}/en_XX_zero_{}".format(output_dir, model_language))
    model.test_predictions(tokenized_dataset["test"])

# few shot experiments. Tune english model using few data from spanish and russian datasets
for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
    for data_size in [10, 100, 1000, 10000]:
        model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                        src_lang=model_language,
                                        tgt_lang=model_language)
        model.train(Dataset.from_dict(tokenized_dataset["train"][:data_size]),
                    Dataset.from_dict(tokenized_dataset["validation"][:data_size]),
                    "{}/en_XX_tuned_{}_{}".format(output_dir, model_language, data_size))
        model.test_predictions(Dataset.from_dict(tokenized_dataset["test"][:data_size]))

# tune english model using complete data from spanish and russian datasets
for tokenized_dataset, model_language in zip(tokenized_datasets[1:3], model_languages[1:3]):
    model = MBartSummarizationModel(model_name="{}/en_XX".format(output_dir),
                                    src_lang=model_language,
                                    tgt_lang=model_language)
    model.train(tokenized_dataset["train"],
                tokenized_dataset["validation"],
                "{}/en_XX_tuned_{}".format(output_dir, model_language),
                save_model=(model_language == "es_XX"))
    model.test_predictions(tokenized_dataset["test"])

# all three languages together
model = MBartSummarizationModel(model_name="{}/en_XX_tuned_es_XX".format(output_dir),
                                src_lang="ru_RU",
                                tgt_lang="ru_RU")
model.train(tokenized_datasets[2]["train"],
            tokenized_datasets[2]["validation"],
            "{}/en_XX_tuned_es_XX_and_ru_RU".format(output_dir),
            save_model=True)
for tokenized_dataset, model_language in zip(tokenized_datasets, model_languages):
    model = MBartSummarizationModel(model_name="{}/en_XX_tuned_es_XX_and_ru_RU".format(output_dir),
                                    src_lang=model_language,
                                    tgt_lang=model_language,
                                    output_dir="{}/en_XX_tuned_es_XX_and_ru_RU_{}".format(output_dir, model_language))
    model.test_predictions(tokenized_dataset["test"])
