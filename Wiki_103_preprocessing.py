import pandas as pd

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
stops = set(stopwords.words('english'))
from collections import Counter
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import os

def get_dataset_values_in_list():
    data_test = pd.read_csv(os.path.join('..', '_utils', 'wikitext-2', 'test.csv'))
    return data_test

if __name__ == '__main__':
    data = get_dataset_values_in_list()