from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
# TrainingArguments will be necessary later when I implement a custom compute_metrics method
import numpy

# read data_preprocessor.py first
# multi-label classification using setfit
# loosely followed https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

# load two datasets from csv files in dataset dictionary
dataset = load_dataset('csv', data_files= {
    "train": "setfit-dataset-train-new.csv",
    "test": "setfit-dataset-test-new.csv"
})

# extract labels
labels = dataset["train"].column_names
labels.remove("text")

# further preprocess data
# used guide https://medium.com/@farnazgh73/few-shot-text-classification-on-a-multilabel-dataset-with-setfit-e89504f5fb75 for help here
# .map takes a method and applies it to each entry in the dataset
# the lambda method converts the label:value pairs in the original dataset to one label
# ex. [Time Management: 0, Python and Coding: 1] becomes [0, 1]
dataset["train"] = dataset["train"].map(lambda entry: {"label": [entry[label] for label in labels]})
dataset["test"] = dataset["test"].map(lambda entry: {"label": [entry[label] for label in labels]})

# collect exactly eight examples of every labeled class in training dataset
# elegant line of code taken from above guide (line 20)
eight_examples_of_each = numpy.concatenate([numpy.random.choice(numpy.where(dataset["train"][label])[0], 8) for label in labels])
# replace training dataset with the eight examples of each
dataset["train"] = dataset["train"].select(eight_examples_of_each)

# remove unnecessary labels
dataset["train"] = dataset["train"].select_columns(["text", "label"])
dataset["test"] = dataset["test"].select_columns(["text", "label"])

# tokenization as specified in the "Fine tuning BERT (and friends)" notebook is not necessary or worthwhile (as far as I know) working with SetFit models
# I say this because my sentiment analysis SetFit model did not require me to tokenize the text fields

# base pretrained model from SetFit library
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2",
                                    # creates a multi-label classification head and uses it in evaluation
                                    multi_target_strategy = "one-vs-rest"
                                    )

# fine tune pretrained model using datasets using default hyperparameters
# doing something like a grid search to optimize hyperparameters looks like it will be impractical because
# training the model (at least on my machine) will take many hours - how much speedup can I get on our lab machines?
model = Trainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"]
)

# Train model
model.train()

# get accuracy of model against test data. I'm not sure how the evaluate method will handle a multi-label model.
metrics = model.evaluate()

# TODO - implement multi-label metrics. I'm not entirely sure how to do this.

model.push_to_hub("setfit-multilabel-test")

print(metrics)
