This folder contains the extracted subset data files from the original Yelp reviews dataset (https://huggingface.co/datasets/yelp_review_full).

yelp-subset.train.csv - 50k instances
yelp-subset.dev.csv - 2.5k instances
yelp-subset.test.csv - 5k instances

Each instance is separated one a line.
Columns in the .csv file:
label, text
label: Classification label for the corresponding text, which get from 0 to 4
text: Unprocessed text from the Yelp dataset

