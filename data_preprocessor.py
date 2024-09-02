# data preprocessor used to populate the excel sheet of zeroes with data
# Not ideal way of data preprocessing but it does the job for now
# run data preprocessor first and then convert the excel file to a csv and run main to
# begin training the model
from openpyxl import load_workbook

# edit workbook to "text_train" and "zeroes_train" for training dataset preprocessing
text_workbook = load_workbook("data/text-train.xlsx")
dataset_workbook = load_workbook("data/zeroes-train.xlsx")
# similar, edit file name from test to train if necessary
file_name = "setfit-dataset-train-new.xlsx"

text = text_workbook.active
dataset = dataset_workbook.active

# excluded less common labels for now
labels = {
    "None": 1,
    "Python and Coding": 2,
    "Github": 3,
    "MySQL": 4,
    "Assignments ": 5, # space after "Assignments" is not a typo;
    # funnily enough, in the original dataset's "Issue" dropdown,
    # there is a stray space after the issues "Assignments", "Quizzes",
    # and "Understanding requirements and instructions". It is just
    # easiest to fix it this way.
    "Quizzes ": 6,
    "Understanding requirements and instructions ": 7,
    "Learning New Material": 8,
    "Course Structure and Materials": 9,
    "Time Management and Motivation": 10,
    "Group Work": 11,
    "API": 12,
    "Project": 13
}

response_num = 2

current_response = text.cell(row=2, column=2).value
print(current_response)

# iterate over zeroes and text spreadsheets concurrently and populate as needed
# zeroes excel file will become the dataset we use
for row in text.iter_rows(min_row=2, min_col=1, max_row=text.max_row, max_col=2, values_only=True):
    if current_response != row[1]:
        current_response = row[1]
        response_num += 1
    try:
        dataset.cell(row=response_num, column=labels[row[0]]).value = "1"
    except KeyError:
        print("Key not found, skipping label")
    dataset.cell(row=response_num, column=14).value = current_response

# save populated dataset
dataset_workbook.save(file_name)




