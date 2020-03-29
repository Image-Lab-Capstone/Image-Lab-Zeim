# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:37:55 2020
Loading data from excel into python data
structures for Dr. Zeim
"""

import numpy as np
import pandas as pd

def read_data(filepath, images):
    csv_dataframe = pd.read_csv(filepath)
    return csv_dataframe
    
    

if __name__ == '__main__':
    df = read_data('input_data/dr_zeim_input.csv', 'input_data/images')
    names = df['Filename']
    labels = df['Label']
    isTraining = df['Train_Image']