import os
from datetime import datetime
import re
from optparse import OptionParser

from comcrawl import IndexClient
import pandas as pd

from selectolax.parser import HTMLParser
from tqdm import tqdm

from MordinezNLP.preprocessing.Basic import BasicProcessor

import logging


def fix_reddit_stuff(text):
    line_split = text.split()
    if line_split[0:3] == ['Rendered', 'by', 'PID']:
        return ""
    elif line_split[0:3] == ['REDDIT', 'and', 'the']:
        return ""
    else:
        return text


def preprocess_text(input_data_list, output_text_file_path):
    bp = BasicProcessor()

    logging.info("Processing text lists")

    reddit_and_the_regex = re.compile(r"(REDDIT and the .* .)")
    rendered_by_regex = re.compile(r"(Rendered by PID .* .)")

    tokenized_items = bp.process(input_data_list, post_rules=[
        lambda x: x.replace("sorry , this has been archived and can no longer be voted on", ""),
        lambda x: x.replace(
            "Press J to jump to the feed . Press question mark to learn the rest of the keyboard shortcuts ", ""),
        lambda x: re.sub(reddit_and_the_regex, "", x),
        lambda x: re.sub(rendered_by_regex, "", x),
        lambda x: x.replace(
            "use the following search parameters to narrow your results Please keep the discussion clean and neutral . ",
            ""),
        lambda x: fix_reddit_stuff(x),
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


def get_text_selectolax(html):
    ## based on https://rushter.com/blog/python-fast-html-parser/
    tree = HTMLParser(html)

    if tree.body is None:
        return None

    for tag in tree.css('script'):
        tag.decompose()
    for tag in tree.css('style'):
        tag.decompose()

    text = tree.body.text(separator='\n')
    return text


if __name__ == '__main__':
    log_file_name = 'logs/1. Common crawl_preprocessing-{}.log'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    logging.basicConfig(filename=log_file_name, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p', level=logging.DEBUG)

    parser = OptionParser()

    parser.add_option(
        "--search_string",
        dest="search_string",
        help='search strings as an array',
        type=str
    )

    parser.add_option(
        "--threads",
        dest="threads",
        help="How many threads You want to download with",
        type=int
    )

    parser.add_option(
        "--output_file_prefix",
        dest="output_file_prefix",
        help="path to save results",
        metavar="FILE"
    )

    options, args = parser.parse_args()

    logging.info("Output file path set to: {}".format(options.output_file_prefix))

    to_write = []

    for search in options.search_string.split(","):
        try:
            client = IndexClient()
            print("Searching for {}".format(search))
            logging.info("Searching for {}".format(search))
            client.search(search)

            client.results = (
                pd.DataFrame(client.results).sort_values(by="timestamp").drop_duplicates("urlkey", keep="last").to_dict(
                    "records")
            )

            client.download(threads=options.threads)

            logging.info("Processing data")
            for entity in tqdm(client.results, "Processing data"):
                if 'warc' in entity['filename']:
                    parsed_text = get_text_selectolax(entity['html'])
                    to_write.append(" ".join([line for line in parsed_text.split("\n") if len(line) > 50]))
        except Exception as e:
            print("Error with {}, {}".format(search, str(e)))

    preprocess_text(to_write, options.output_file_prefix)

    logging.info("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
    print("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
