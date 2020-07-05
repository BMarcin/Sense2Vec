import os
from glob import glob
from optparse import OptionParser

import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

from MordinezNLP.exploration.Exploration import Exploration

if __name__ == '__main__':
    log_file_name = 'logs/Analyze_corpus-{}.log'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    logging.basicConfig(filename=log_file_name, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p', level=logging.DEBUG)

    parser = OptionParser()

    parser.add_option(
        "-i",
        "--input_file_path",
        dest="input_file_path",
        help="Path to input corpus",
        metavar="FILE"
    )

    parser.add_option(
        "-o",
        "--output_file_path",
        dest="output_file_path",
        help="Path to output xlsx file",
        metavar="FILE"
    )

    parser.add_option(
        "-p",
        "--use_parts",
        dest="use_parts",
        help="Define if script has to use parts",
        action='store_true'
    )

    options, args = parser.parse_args()

    logging.info("Input file path set to: {}".format(options.input_file_path))
    logging.info("Output file path set to: {}".format(options.output_file_path))

    logging.info("Reading file...")
    if options.use_parts:
        texts_to_process = []
        for file in tqdm(glob(options.input_file_path+"*.txt"), desc="Reading inputs"):
            texts_to_process.append(open(file).read())

        text_to_process = "\n".join(texts_to_process)
    else:
        text_to_process = open(options.input_file_path).read()

    logging.info("Starting exploration")
    nlp_exp = Exploration(text_to_process.split("\n"))
    nlp_exp.write_to_xlsx(options.output_file_path, 10000, plot_results=100)

    if len(nlp_exp.anomalies) > 0:
        print("Found some anomalies. Check logs for more informations.")
        for anomaly in nlp_exp.anomalies:
            logging.error("Found anomaly in sentence: {}".format(anomaly))

    logging.info("Unique characters found in file: {}".format(" ::: ".join(nlp_exp.unique_characters)))
    logging.info("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
    print("Average word length: {}".format(np.average(nlp_exp.word_lengths)))
    print("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
