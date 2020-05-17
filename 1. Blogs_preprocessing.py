from bs4 import BeautifulSoup
from glob import glob

import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from optparse import OptionParser

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

    for file in tqdm(blogs_glob, desc="Reading XML files"):
        try:
            parsed = BeautifulSoup(open(file).read(), "html.parser")
            elements = parsed.findAll('post')
            for single_entry in elements:
                text = single_entry.text

                preprocessed_text = text.replace("\n", "").replace("\t", "")
                texts.append(preprocessed_text)
        except Exception as e:
            # print(e)
            errors += 1

    print("{} files had errors while reading".format(errors))
    return texts


def plot_most_common_words(data_list, n=30):
    """
    Plot a chart of most common tokens in dataset
    :param data_list: datalist from get_dataset_values_in_list function
    :param n: how many tokens to plot
    :return:
    """
    tokens_counter = Counter()

    for i, item in enumerate(tqdm(data_list, desc='Processing data to plot')):
        for sentence in nltk.sent_tokenize(item):
            for token in nltk.word_tokenize(sentence):
                tokens_counter[token.lower()] += 1

    plt.bar(dict(tokens_counter.most_common(n)).keys(), dict(tokens_counter.most_common(n)).values())
    plt.xticks(rotation=90)
    plt.title("Top words")
    plt.show()


def plot_most_common_without_stop_words(data_list, n=30):
    """
    Plot a chart of most common tokens (tokens that are not a stopwords) in dataset
    :param data_list: data_list: datalist from get_dataset_values_in_list function
    :param n: how many tokens to plot
    :return:
    """
    tokens_counter = Counter()

    for i, item in enumerate(tqdm(data_list, desc='Processing data to plot')):
        for sentence in nltk.sent_tokenize(item):
            for token in nltk.word_tokenize(sentence):
                if token.lower() not in stops:
                    tokens_counter[token.lower()] += 1

    plt.bar(dict(tokens_counter.most_common(n)).keys(), dict(tokens_counter.most_common(n)).values())
    plt.xticks(rotation=90)
    plt.title("Top words")
    plt.show()


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
    space_regex = re.compile("(\s{2,})")
    date_regex = re.compile(
        "((\d{4}s)|(\d{2,4}\s–\s\d{2,4})|(\d{1,2} [A-Z][a-z]+ \d{2,4})|(((January)|(February)|(March)|(April)|(May)|(June)|(July)|(October)|(September)|(August)|(November)|(December))\s+\d{1,4})|(\d{1,4}\s+((January)|(February)|(March)|(April)|(May)|(June)|(July)|(October)|(September)|(August)|(November)|(December))))")
    number_regex = re.compile("((\d{2,4}(st|nd|rd|th)))")
    none_regex = re.compile("((\(.*\) )|(\[.*\] )|(@.@)|(\d+ – \d+)|( Source : (\n .+)+))")
    digits_regex = re.compile("(\d+ )")
    quote_regex = re.compile('(("[a-zA-Z0-9\' ]+"))')
    limit_regex = re.compile("([^a-zA-Z0-9,\.<>' ])")
    space_before_regex = re.compile("^(\s)")
    dot_regex = re.compile("\s+\.")
    com_regex = re.compile("\s+,")
    comm_regex = re.compile(",+")
    multi_dot_regex = re.compile("\.+")
    multi_unk_regex = re.compile("(<unk>[ ]*){2,}")

    with open(output_text_file_path, "w", encoding="utf8") as f:
        for i, item in enumerate(tqdm(input_data_list, desc='Processing input')):
            regexed = item.replace("urlLink", "")
            regexed = re.sub(space_regex, " ", regexed)
            #     regexed = regexed.replace("  ", " ")
            regexed = re.sub(date_regex, "<date>", regexed)
            regexed = re.sub(number_regex, "<number>", regexed)
            regexed = re.sub(quote_regex, "<quote>", regexed)
            regexed = re.sub(none_regex, "", regexed)
            regexed = re.sub(digits_regex, "<number> ", regexed)
            regexed = re.sub(limit_regex, "", regexed)
            regexed = re.sub(dot_regex, ".", regexed)
            regexed = re.sub(com_regex, ",", regexed)
            regexed = re.sub(comm_regex, ",", regexed)
            regexed = regexed.replace(",.", ".")
            regexed = regexed.replace(" 's", "'s")
            regexed = regexed.replace(" 't", "'t")
            regexed = regexed.replace("' ", "'")
            regexed = re.sub(multi_dot_regex, ".", regexed)
            regexed = re.sub(multi_unk_regex, "<unk> ", regexed)
            regexed = re.sub(space_before_regex, "", regexed)
            regexed = re.sub(space_regex, " ", regexed)

            regexed = regexed.replace("<unk>", "<unknown>")

            f.write(regexed + "\n")


if __name__ == '__main__':
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
        "--output_file_path",
        dest="output_file_path",
        help="Path to output text file",
        metavar="FILE"
    )

    options, args = parser.parse_args()

    data = get_dataset_values_in_list(options.input_dir_path)
    preprocess_text(data, options.output_file_path)
