import logging
import os
from datetime import datetime
from optparse import OptionParser
import spacy
from tqdm import tqdm
from glob import glob
from Sense2Vec.tokenizer import create_custom_tokenizer as my_custom_tokenizer

nlp = spacy.load("en_core_web_sm")


def preprocess_pipeline(input_dir_path, output_file_path, custom_tokenizer_func, threads=4, batch=200):
    logging.info("Reading files")
    file_inputs = [line.replace("\n", "") for file in
                   tqdm(glob(input_dir_path + "*.txt"), desc="Reading files") for
                   line in open(file, encoding="utf8").readlines()]

    logging.info("Starting Spacy pipeline")
    pipe = nlp.pipe(file_inputs, disable=["ner"],
                    n_process=threads, batch_size=batch)

    combined_list = []
    sentences = set()

    tokenizer = custom_tokenizer_func(nlp)
    nlp.tokenizer = tokenizer

    logging.info("Processing...")
    for doc in tqdm(pipe, desc='Processing files', total=len(file_inputs)):
        for sent in doc.sents:
            if sent not in sentences:
                combs = []
                for token in sent:
                    combs.append(token.text + "|" + token.pos_)

                combined_list.append(combs)
                sentences.add(sent)

    logging.info("Saving to file")
    with open(output_file_path, 'w', encoding='utf8') as fp:
        for sentence in combined_list:
            fp.write(" ".join(sentence) + " \n")


if __name__ == '__main__':
    log_file_name = 'logs/2. Text Corpus to Tagged Corpus-{}.log'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    logging.basicConfig(filename=log_file_name, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p', level=logging.DEBUG)

    parser = OptionParser()

    parser.add_option(
        "-i",
        "--input_dir_path",
        dest="input_dir_path",
        help="Path to preprocessed txt files",
        metavar="PATH"
    )

    parser.add_option(
        "-o",
        "--output_file_path",
        dest="output_file_path",
        help="Path to output text file",
        metavar="FILE"
    )

    parser.add_option(
        "-t",
        "--threads",
        dest="threads",
        help="number of threads for spacy pipeline",
    )

    parser.add_option(
        "-b",
        "--bs",
        dest="bs",
        help="Batchsize for spacy pipeline"
    )

    options, args = parser.parse_args()

    logging.info("Starting preprocessing with arguments: input_dir={}, output_file_path={}, threads={}, bs={}".format(
        options.input_dir_path, options.output_file_path, options.threads, options.bs))

    preprocess_pipeline(
        options.input_dir_path,
        options.output_file_path,
        my_custom_tokenizer,
        threads=int(options.threads),
        batch=int(options.bs)
    )

    logging.info("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
    print("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
