import pandas as pd

import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
from optparse import OptionParser

import logging
from datetime import datetime

from MordinezNLP.preprocessing.Basic import BasicProcessor

nltk.download('stopwords')
nltk.download('punkt')
stops = set(stopwords.words('english'))


def get_dataset_values_in_list(input_path):
    """
    Based on wikitext dataset available on https://course.fast.ai/datasets
    Function takes test and train dataset files and process it to list
    :return: list containing train and test data
    """
    logging.info("Starting reading files...")

    data_test = pd.read_csv(os.path.join(input_path, 'test.csv'), header=None)[0].tolist()
    data_train = pd.read_csv(os.path.join(input_path, 'train.csv'), header=None)[0].tolist()

    return data_test + data_train


def preprocess_text(input_data_list, output_text_file_path):
    """
    Function that preprossesses data from wikitext-2/wikitext-3 to ready to feed tokens.
    Some values like numbers or dates are replaced by special token like <number> or <date>.
    There is no sense to make a vector of number or date. It is better to make a unique token
    of number or date to make for it a vector than for a specified one.
    :param input_data_list: preprocessed input list from get_dataset_values_in_list function
    :param output_text_file_path: A path for output file
    :return:
    """
    title_regex = re.compile("((= )+.*(= )+)\s+")
    none_regex = re.compile("( Source : (\n .+)+)")
    bp = BasicProcessor()

    logging.info("Processing text lists")

    with open(output_text_file_path, "w", encoding="utf8") as f:
        tokenized_items = bp.process(input_data_list, pre_rules=[
            lambda x: re.sub(title_regex, "", x),
            lambda x: re.sub(none_regex, "", x)
        ])

        logging.info("Saving processed texts to file")

        for item in tqdm(tokenized_items, desc='Saving file'):
            f.write(item + "\n")


if __name__ == '__main__':
    log_file_name = 'logs/1. Wiki_103_preprocessing-{}.log'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    logging.basicConfig(filename=log_file_name, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p', level=logging.DEBUG)

    parser = OptionParser()

    parser.add_option(
        "-i",
        "--input_dir_path",
        dest="input_dir_path",
        help="Path to wikitext-2/wikitext-103 dir downloaded from fast.ai",
        metavar="PATH"
    )

    parser.add_option(
        "-o",
        "--output_file_path",
        dest="output_file_path",
        help="Path to output text file",
        metavar="FILE"
    )

    options, args = parser.parse_args()

    logging.info("Input dir path set to: {}".format(options.input_dir_path))
    logging.info("Output file path set to: {}".format(options.output_file_path))

    data = get_dataset_values_in_list(options.input_dir_path)
    preprocess_text(data, options.output_file_path)

    logging.info("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
    print("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
