from optparse import OptionParser
import spacy
from tqdm import tqdm
from glob import glob
from Sense2Vec.tokenizer import create_custom_tokenizer as my_custom_tokenizer

nlp = spacy.load("en_core_web_sm")


def preprocess_pipeline(input_dir_path, output_file_path, custom_tokenizer_func, threads=4, batch=100):
    file_inputs = [line.replace("\n", "") for file in tqdm(glob(input_dir_path + "*.txt"), desc="Processing files") for line in open(file).readlines()]
    pipe = nlp.pipe(file_inputs, disable=["ner"],
                    n_threads=threads, batch_size=batch)

    combined_list = []

    tokenizer = custom_tokenizer_func(nlp)
    nlp.tokenizer = tokenizer

    for doc in tqdm(pipe, desc='Processing files', total=len(file_inputs)):
        for sent in doc.sents:
            combs = []
            for token in sent:
                combs.append(token.text+"|"+token.pos_)

            combined_list.append(combs)

    with open(output_file_path, 'w', encoding='utf8') as fp:
        for sentence in combined_list:
            fp.write("\t".join(sentence) + " \n")


if __name__ == '__main__':
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

    options, args = parser.parse_args()

    preprocess_pipeline(
        options.input_dir_path,
        options.output_file_path,
        my_custom_tokenizer
    )
