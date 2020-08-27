from datetime import datetime
import logging
import os
import string
import sys
import random
from glob import glob
from optparse import OptionParser
import re

from elasticsearch import Elasticsearch
from tqdm import tqdm

from MordinezNLP.preprocessing.Basic import BasicProcessor
from MordinezNLP.preprocessing.HTMLRemover import remove_html


class ElasticSearchClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self.connection = Elasticsearch([{'host': host, 'port': port}], timeout=100)
        print("Connected")

        self.available_indexes = [index for index in self.connection.indices.get("*")]
        # self.available_indexes = ['eco_agenda']

        self.base_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "language_version_info.en.keyword": "original"
                            }
                        }
                    ]
                }
            }
        }

    def _process_policy_insider_scrolled_doc(self, document: dict, language: str) -> str:
        try:
            if 'content' in document['_source'][language].keys():
                content_parts = document['_source'][language]['content']
                content = "".join([content_part['content'] for content_part in content_parts])

                return remove_html(content)
            else:
                return ""
        except Exception as e:
            print(e)
            # print(document['_source']['en']['content'])
            sys.exit(0)

    def _save_list_to_txt(self, file_contents: list, output_path: str, index_name=None):
        for file_content in tqdm(file_contents, desc="Saving to txt files"):
            file_name = self._get_random_string()
            while os.path.exists(os.path.join(output_path, file_name + ".txt")):
                file_name = self._get_random_string()

            logging.info("Saving file: {}".format(file_name))
            with open(os.path.join(output_path, file_name + ".txt"), "w", encoding="utf8") as f:
                # if 'library' in file_content:
                #     with open("lib.txt", "a", encoding="utf8") as f2:
                #         f2.write('Found "library" in {} in index {}\n'.format(file_name, index_name))
                #     print('Found "library" in {} in index {}'.format(file_name, index_name))
                # if 'libraryt' in file_content:
                #     with open("libt.txt", "a", encoding="utf8") as f2:
                #         f2.write('Found "libraryt" in {} in index {}\n'.format(file_name, index_name))
                #     print('Found "libraryt" in {} in index {}'.format(file_name, index_name))
                f.write(file_content)

    def _get_random_string(self, length=64):
        return "".join([random.choice(string.ascii_uppercase + string.digits) for _ in range(length)])

    def scroll_data(self, index_name: str, save_path: str) -> None:
        if self.connection.indices.exists(index=index_name):
            current_docs = []

            data = self.connection.search(
                index=index_name,
                scroll="2m",
                size=100,
                body=self.base_query
            )

            scroll_id = data['_scroll_id']
            scroll_size = len(data['hits']['hits'])

            for doc in tqdm(data['hits']['hits'], desc="Processing"):
                processed_doc = self._process_policy_insider_scrolled_doc(doc, 'en')
                if len(processed_doc) > 10:
                    current_docs.append(processed_doc)

            print("Scrolling index: {}".format(index_name))
            logging.info("Scrolling index: {}".format(index_name))
            while scroll_size > 0:
                data = self.connection.scroll(scroll_id=scroll_id, scroll="2m")

                for doc in tqdm(data['hits']['hits'], desc="Processing"):
                    processed_doc = self._process_policy_insider_scrolled_doc(doc, 'en')
                    if len(processed_doc) > 10:
                        current_docs.append(processed_doc)

                scroll_id = data['_scroll_id']
                scroll_size = len(data['hits']['hits'])

            logging.info("Scrolled {} docs".format(len(current_docs)))
            print("Scrolled {} docs".format(len(current_docs)))
            self._save_list_to_txt(current_docs, save_path, index_name=index_name)


def remove_tables(text):
    return re.sub('<table.+?</table>', '', text, flags=re.DOTALL)


def post_check(text):
    if 'library' in text:
        print(text)
    return text


def preprocess_text(input_data_list, output_text_file_path):
    bp = BasicProcessor()

    logging.info("Processing text lists")
    tokenized_items = bp.process(input_data_list, pre_rules=[
        lambda x: remove_tables(x)
    ])

    file_counter = 0
    for line_number in range(0, len(input_data_list), 1000):
        with open(output_text_file_path + "-" + str(file_counter) + ".txt", "w", encoding="utf8") as f:
            logging.info(
                "Saving processed texts to file: {}".format(output_text_file_path + "-" + str(file_counter) + ".txt"))

            for item in tqdm(tokenized_items[line_number:line_number + 1000],
                             desc='Saving file {}'.format(output_text_file_path + "-" + str(file_counter) + ".txt")):
                f.write(item + "\n")
            file_counter += 1


if __name__ == '__main__':
    log_file_name = 'logs/1. Policy insider preprocessing-{}.log'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    logging.basicConfig(filename=log_file_name, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p', level=logging.INFO)

    parser = OptionParser()

    parser.add_option(
        "--elastic_search_host",
        dest="elastic_search_host",
        help="Elastic search host ip",
        type=str
    )

    parser.add_option(
        "--elastic_search_port",
        dest="elastic_search_port",
        help="Elastic search host port",
        type=int
    )

    parser.add_option(
        "--temp_files_path",
        dest="temp_files_path",
        help="",
        metavar="PATH"
    )

    parser.add_option(
        "--output_file_prefix",
        dest="output_file_prefix",
        help="path to save results",
        metavar="FILE"
    )

    options, args = parser.parse_args()

    logging.info("Output file path set to: {}".format(options.output_file_prefix))

    # esc = ElasticSearchClient(
    #     options.elastic_search_host,
    #     options.elastic_search_port
    # )
    #
    # print(esc.available_indexes)
    #
    # for available_index in esc.available_indexes:
    #     try:
    #         esc.scroll_data(
    #             available_index,
    #             options.temp_files_path
    #         )
    #     except Exception as e:
    #         print("Got exception: {}".format(str(e)))

    to_write = []

    for file in glob(options.temp_files_path + "/*.txt"):
        current_lines = []
        with open(file, encoding="utf8") as f:
            for line in f.readlines():
                if len(line) > 50:
                    current_lines.append(line)
        to_write.append("\n".join(current_lines))

    preprocess_text(to_write, options.output_file_prefix)

    logging.info("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
    print("Done. Log is available under: {}".format(os.path.abspath(log_file_name)))
