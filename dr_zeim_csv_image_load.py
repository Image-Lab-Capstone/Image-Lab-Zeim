"""
Loading input data from excel from Dr. Zeim
and converting to pandas dataframe with 
binary labels for each of the possible labels
"""

import pandas as pd
pd.options.display.max_columns = 50

#Set filepaths
input_data_path = 'sample/zeim_input.csv'
label_dictionary_path = 'sample/labels.csv'
output_path = 'sample/output.csv'

#Read data set and label list
df = pd.read_csv(input_data_path)
labels = pd.read_csv(label_dictionary_path)

#Set arrays
file_names = df['Image Index']
file_labels = df['Finding Labels']
label_dict = labels['Finding Label']
isTraining = df['Train Image']

#Create binary column for each classification
for label in label_dict:
    new_label_column = []
    for classifications in file_labels:
        if label in classifications:
            new_label_column.append(1)
        else:
            new_label_column.append(0)
    df[label] = new_label_column
    
#Export dataframe to csv
df.to_csv (output_path, index = False, header=True)

            



