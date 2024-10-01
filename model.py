from datasets import load_dataset
from setfit import SetFitModel, Trainer
import numpy
from sklearn.metrics import multilabel_confusion_matrix
import data_preprocessor


# Generate a confusion matrix for each label in the dataset. For each column/vector
# in the label_num by reflection_num matrix of predictions output by the model,
# one confusion matrix will be created. That will represent the confusion for
# that label. Repeat process for each label. Hopefully, with enough predictions
# for each class, a minimally noisy confusion matrix can be created for each label
def compute_metrics(y_pred, y_true) -> dict[str, float]:
    # confusion_matrices is a list of n-dimensional numpy arrays
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    # initialize labels
    labels = ["None",
              "Python and Coding",
              "Github",
              "MySQL",
              "Assignments",
              "Quizzes",
              "Understanding requirements and instructions",
              "Learning New Material",
              "Course Structure and Materials",
              "Time Management and Motivation",
              "Group Work",
              "API",
              "Project"]
    result = {}
    x = 0
    for matrix in confusion_matrices:
        # flatten confusion matrix to list
        matrix.ravel()
        # populate results with information from the label's confusion matrix
        result.update({f"{labels[x]}-tp": matrix[0]})
        result.update({f"{labels[x]}-fn": matrix[1]})
        result.update({f"{labels[x]}-tn": matrix[2]})
        result.update({f"{labels[x]}-fp": matrix[3]})
        x += 1
        if x >= len(labels):
            break
    return result


def main():
    # merge array of zeroes and raw data from data folder into correctly formatted files which can be encoded
    # and which the model can be trained/tested on
    # DataPreprocessor.process_data("train", "setfit-dataset-train-new2.xlsx")
    # DataPreprocessor.process_data("test", "setfit-dataset-test-new2.xlsx")
    # multi-label classification using setfit
    # loosely followed https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

    data_preprocessor.process_data("train", "setfit-dataset-train.csv")
    data_preprocessor.process_data("test", "setfit-dataset-test.csv")

    # load two datasets from csv files in dataset dictionary
    dataset = load_dataset('csv', data_files={
        "train": "setfit-dataset-train.csv",
        "test": "setfit-dataset-test.csv"
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
    trainer = Trainer(
        model = model,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        metric = compute_metrics
    )

    trainer.train()

    metrics = trainer.evaluate()

    model.push_to_hub("setfit-multilabel-test")

    print(metrics)


if __name__ == "__main__":
    main()
