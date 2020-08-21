import json
import os
import random
import string

from elasticsearch import Elasticsearch
from tqdm import tqdm


class ElasticSearchClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self.connection = Elasticsearch([{'host': host, 'port': port}], timeout=100)

        self.available_indexes = [index for index in self.connection.indices.get("*")]

    @staticmethod
    def _build_query(language: str):
        return {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "language_version_info." + language + ".keyword": "original"
                            }
                        }
                    ]
                }
            }
        }

    @staticmethod
    def get_random_string(length=64):
        return "".join([random.choice(string.ascii_uppercase + string.digits) for _ in range(length)])

    @staticmethod
    def save_text_to_file(text: str or list or dict, save_dir: str):
        file_name = ElasticSearchClient.get_random_string(length=64)
        while os.path.exists(os.path.join(save_dir, file_name + ".json")):
            file_name = ElasticSearchClient.get_random_string(length=64)

        with open(os.path.join(save_dir, file_name + ".txt"), "w", encoding="utf8") as f:
            f.write(text)

    def scroll_data_get_pdf_urls(self, index_name: str, language: str):
        if self.connection.indices.exists(index=index_name):
            docs_links = []

            data = self.connection.search(
                index=index_name,
                scroll="2m",
                size=100,
                body=self._build_query(language)
            )

            scroll_id = data['_scroll_id']
            scroll_size = len(data['hits']['hits'])

            for doc in tqdm(data['hits']['hits'], desc="Processing"):
                if 'pdf' in doc['_source'][language]['url'].keys():
                    docs_links.append(doc['_source'][language]['url']['pdf'])

            while scroll_size > 0:
                data = self.connection.scroll(scroll_id=scroll_id, scroll="2m")

                for doc in tqdm(data['hits']['hits'], desc="Processing"):
                    if 'pdf' in doc['_source'][language]['url'].keys():
                        docs_links.append(doc['_source'][language]['url']['pdf'])

                scroll_id = data['_scroll_id']
                scroll_size = len(data['hits']['hits'])

            return docs_links
