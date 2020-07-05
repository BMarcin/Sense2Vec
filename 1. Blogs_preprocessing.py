import os

from bs4 import BeautifulSoup
from glob import glob

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from optparse import OptionParser

import logging
from datetime import datetime

from MordinezNLP.preprocessing.Basic import BasicProcessor

nltk.download('stopwords')
nltk.download('punkt')
stops = set(stopwords.words('english'))


def get_dataset_values_in_list(input_path):
    """
    Based on 'blogs' dataset. Function reads contents and saves to one list
    :return: list containing train and test data
    """
    blogs_glob = glob(input_path + "*.xml")
    texts = []
    errors = 0

    logging.info("Starting reading files...")

    for file in tqdm(blogs_glob, desc="Reading XML files"):
        try:
            parsed = BeautifulSoup(open(file).read(), "html.parser")
            elements = parsed.findAll('post')
            logging.info("Reading {}".format(file))
            for single_entry in elements:
                text = single_entry.text

                preprocessed_text = text.replace("\n", "").replace("\t", "")
                texts.append(preprocessed_text)
        except Exception:
            logging.error("Error opening file: {}".format(file))
            errors += 1

    print("{} files had errors while reading".format(errors))
    return texts


def preprocess_text(input_data_list, output_text_file_path):
    """
    Function that preprossesses data from blogs dataset to ready to feed tokens.
    Some values like numbers or dates are replaced by special token like <number> or <date>.
    There is no sense to make a vector of number or date. It is better to make a unique token
    of number or date to make for it a vector than for a specified one.
    :param input_data_list: preprocessed input list from get_dataset_values_in_list function
    :param output_text_file_path: A path for output file
    :return:
    """
    bp = BasicProcessor()

    logging.info("Processing text lists")

    ''' we have to divide blogs dataset to separate files, because Spacy will consume tons of RAM '''
    tokenized_items = bp.process(input_data_list, pre_rules=[
        lambda x: x.replace("urlLink", "")
    ])

    file_counter = 0
    for line_number in range(0, len(input_data_list), 10000):
        with open(output_text_file_path + "-" + str(file_counter) + ".txt", "w", encoding="utf8") as f:
            logging.info(
                "Saving processed texts to file: {}".format(output_text_file_path + "-" + str(file_counter) + ".txt"))

            for item in tqdm(tokenized_items[line_number:line_number + 10000],
                             desc='Saving file {}'.format(output_text_file_path + "-" + str(file_counter) + ".txt")):
                f.write(item + "\n")
        file_counter += 1


if __name__ == '__main__':
    log_file_name = 'logs/1. Blogs_preprocessing-{}.log'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    logging.basicConfig(filename=log_file_name, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p', level=logging.DEBUG)

    parser = OptionParser()

    parser.add_option(
        "-i",
        "--input_dir_path",
        dest="input_dir_path",
        help="Path to blogs dir",
        metavar="PATH"
    )

    parser.add_option(
        "-o",
        "--output_file_prefix",
        dest="output_file_prefix",
        help="Path to output text file",
        metavar="FILE"
    )

    options, args = parser.parse_args()

    logging.info("Input dir path set to: {}".format(options.input_dir_path))
    logging.info("Output file path set to: {}".format(options.output_file_prefix))

    data = get_dataset_values_in_list(options.input_dir_path)
    preprocess_text(data, options.output_file_prefix)

    logging.info("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
    print("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
