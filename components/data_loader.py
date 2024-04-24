import csv

def load_poems_from_csv(file_path):
    preprocessed_poems_data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            title = row[0]
            author = row[1]
            lines = row[2].strip("[]").split(", ")
            preprocessed_lines = [preprocess_poem(line) for line in lines]
            preprocessed_poems_data.append((title, author, preprocessed_lines))
    return preprocessed_poems_data
