import json
import os
from glob import glob
from io import BytesIO
from itertools import repeat
from multiprocessing import Pool
from threading import Thread

import requests
from tqdm import tqdm

from MordinezNLP.parsing.process_pdf import process_pdf
from MordinezNLP.preprocessing.Basic import BasicProcessor
from Sense2Vec.ElasticSearchClient import ElasticSearchClient


def download_and_save(base_dir: str, url: str):
    try:
        pdf = BytesIO(requests.get(url).content)
        pdf_output = process_pdf(pdf)
        ElasticSearchClient.save_text_to_file(pdf_output, base_dir)
    except Exception as e:
        print('Got exception {exception} on URL {url}'.format(exception=e, url=url))


def calc_downloading_progress(base_dir: str, items: list):
    progress = tqdm(total=len(items), desc="Downloading and saving text files")
    files = glob(base_dir + "/*.txt")

    last_len = len(files)
    progress.update(last_len)

    while len(files) <= len(items):
        files_len = len(glob(base_dir + "/*.txt"))
        if files_len != last_len:
            progress.update(files_len - last_len)
            last_len = files_len


if __name__ == '__main__':
    if not os.path.exists("doc_urls.json"):
        esc = ElasticSearchClient(
            "192.168.113.100",
            9200
        )

        doc_urls = []

        for available_index in esc.available_indexes:
            try:
                print("Processing index {}".format(available_index))
                urls = esc.scroll_data_get_pdf_urls(available_index, 'en')
                doc_urls = doc_urls + urls

                # ' for tests '
                # if len(doc_urls) > 4:
                #     break
            except Exception as e:
                print("Got exception {}".format(e))

        print('Got {} URLs'.format(len(doc_urls)))

        with open("doc_urls.json", "w", encoding="utf8") as f:
            json.dump(doc_urls, f, ensure_ascii=True, indent=4)
    else:
        doc_urls = []
        with open("doc_urls.json", "r", encoding="utf8") as f:
            doc_urls = json.loads(f.read())

    # thread = Thread(target=calc_downloading_progress, args=('../_utils/policyinsider_en', doc_urls))
    # thread.daemon = True
    # thread.start()
    #
    # with Pool(8) as p:
    #     p.starmap(download_and_save, zip(repeat('../_utils/policyinsider_en'), doc_urls))

    input_data_list = set()
    for file_name in tqdm(glob("../_utils/policyinsider_en/*.txt"), desc="Reading files"):
        with open(file_name, "r", encoding="utf8") as f:
            input_data_list.update(set(f.readlines()))

    input_data_list = list(input_data_list)

    bp = BasicProcessor()
    input_data_list = bp.process(input_data_list)

    file_counter = 0
    for line_number in range(0, len(input_data_list), 1000):
        with open("data/preprocessed/policyinsider_en" + "-" + str(file_counter) + ".txt", "w", encoding="utf8") as f:
            for item in tqdm(input_data_list[line_number:line_number + 1000],
                             desc='Saving file {}'.format("data/preprocessed/policyinsider_en" + "-" + str(file_counter) + ".txt")):
                f.write(item + "\n")
            file_counter += 1
