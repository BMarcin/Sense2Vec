import gzip
import io
import json
import logging
import os
import re
import sys
from glob import glob
from threading import Thread
from itertools import repeat

import requests
import urllib.parse

from tqdm import tqdm

from multiprocessing import Pool

from MordinezNLP.preprocessing.HTMLRemover import remove_html


class CommonCrawlDownloader:
    def __init__(self, links_to_search: list, txt_files_path: str, index_name: str = "CC-MAIN-2020-24"):
        self.links_to_search = links_to_search
        self.txt_files_path = txt_files_path
        self.index_name = index_name

        self.file_name_regex = re.compile("[^\w]+")

        self.post_sources = []

        print(links_to_search)

        # use multithreading
        with Pool(5) as p:
            records = tqdm(p.map(self.get_sources_for_url, links_to_search), desc="Downloading indexes")

        # flat the output from threads
        for record in records:
            self.post_sources += record

        # for entry in self.post_sources:
        #     print(entry['filename'], entry['length'], entry['offset'])

        # func = partial(self.download_files, self.txt_files_path)
        thread = Thread(target=self.calc_downloading_progress, args=(self.txt_files_path,))
        thread.daemon = True
        thread.start()
        with Pool(7) as p:
            p.starmap(self.download_files, zip(repeat(self.txt_files_path), self.post_sources))

        # progress = tqdm(total=len(self.links_to_search), desc="Downloading indexes")
        # while len(self.links_to_search) > 0:
        #     link = self.links_to_search.pop()
        #     progress.update(1)
        #     try:
        #         self.post_sources += self.get_sources_for_url(link)
        #     except Exception as e:
        #         progress.update(-1)
        #         self.links_to_search.append(link)
        #         logging.error("Got exception: {}. During processing: {}".format(str(e), link))
        # progress.close()
        # time.sleep(1)
        #
        # progress = tqdm(total=len(self.post_sources), desc="Downloading and saving text files")
        # while len(self.post_sources) > 0:
        #     entry = self.post_sources.pop()
        #     progress.update(1)
        #     try:
        #         self.download_files(entry, self.txt_files_path)
        #     except Exception as e:
        #         self.post_sources.append(entry)
        #         progress.update(-1)
        #         logging.error("Error during downloading and saving: {}. Error: {}.".format(entry['filename'], str(e)))
        # progress.close()

    def calc_downloading_progress(self, base_dir: str):
        progress = tqdm(total=len(self.post_sources), desc="Downloading and saving text files")
        files = glob(base_dir + "/*.txt")

        last_len = len(files)
        progress.update(last_len)

        while len(files) <= len(self.post_sources):
            files_len = len(glob(base_dir + "/*.txt"))
            if files_len != last_len:
                progress.update(files_len - last_len)
                last_len = files_len

    def get_sources_for_url(self, url, mime: str = "text/html", base_url: str = "http://index.commoncrawl.org"):
        url_list = [url]

        while len(url_list) > 0:
            url_list.pop()

            pre_url = urllib.parse.quote(url, safe='')
            post_url = "{}/{}-index?url={}&output=json".format(base_url, self.index_name, pre_url)

            try:
                response = requests.get(post_url)

                if response.status_code == 200:
                    processed_data = []

                    res = response.text.split("\n")
                    for line in res:
                        if len(line) > 10:
                            entry = json.loads(line)
                            if 'mime-detected' in entry.keys() and 'status' in entry.keys() and \
                                    'languages' in entry.keys():
                                if entry['mime-detected'] == mime and entry['status'] == "200" and \
                                        entry['languages'] == 'eng':
                                    processed_data.append(entry)
                    return processed_data
                elif response.status_code == 404:
                    return []
                else:
                    url_list.append(url)
                    # raise Exception("Status code: {}".format(response.status_code))
            except Exception as e:
                # raise e
                url_list.append(url)

    def download_files(self, base_dir: str, entry: dict, base_url: str = "https://commoncrawl.s3.amazonaws.com"):
        # print(entry, base_dir)
        entry_list = [entry]
        while len(entry_list) > 0:
            entry_list.pop()
            offset, length = int(entry["offset"]), int(entry["length"])
            offset_end = offset + length - 1

            headers = {"Range": "bytes={}-{}".format(str(offset), str(offset_end))}

            path_to_save = os.path.join(base_dir, re.sub(self.file_name_regex, "_", entry['filename']) + "--" +
                                        str(offset) + "-" + str(offset_end) + ".txt")

            response = requests.get("{}/{}".format(base_url, entry['filename']), stream=True,
                                    headers=headers)
            if not os.path.exists(path_to_save):
                if response.status_code == 206:
                    zipped_file = io.BytesIO(response.content)
                    unzipped_file = gzip.GzipFile(fileobj=zipped_file)
                    raw_data: bytes = unzipped_file.read()
                    try:
                        text_data = raw_data.decode("utf8")

                        data_parts = text_data.strip().split("\r\n\r\n", 2)
                        if len(data_parts) == 3:
                            with open(path_to_save, "w", encoding="utf8") as f:
                                f.write(remove_html(data_parts[2]))
                            return path_to_save
                    except Exception:
                        with open(path_to_save, "w", encoding="utf8") as f:
                            f.write("")
                else:
                    entry_list.append(entry)
                    # raise Exception("Http response code for parital tar archive: {}".format(response.status_code))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p', level=logging.DEBUG)
    ccd = CommonCrawlDownloader(
        [
            "reddit.com/r/space/*",
            "reddit.com/r/medicine*",
            "reddit.com/r/spacex/*"
            "reddit.com/r/AMA*"
        ],
        "../../data/commoncrawl")
