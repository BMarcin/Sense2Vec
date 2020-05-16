from tqdm import tqdm
import json
import os


def build_dataset(tokens_file_path, pos_file_path, output_file_path):
    dataset = []

    corpus_tokens = open(tokens_file_path).readlines()
    corpus_token_poss = open(pos_file_path).readlines()

    corpus_tokens = [sentence.replace("\n", "").split("\t") for sentence in corpus_tokens]
    corpus_token_poss = [sentence.replace("\n", "").split("\t") for sentence in corpus_token_poss]

    for xx in tqdm(zip(corpus_tokens, corpus_token_poss), desc="Building dataset", total=len(corpus_tokens)):
        ##todo rename
        zdanie = xx[0]
        zdanie_pos = xx[1]
        combined = [x[0] + "===" + x[1] for x in zip(xx[0], xx[1])]

        dataset.append({
            "sentence": zdanie,
            "pos": zdanie_pos,
            "combined": combined
        })

    with open(output_file_path, 'w', encoding="utf8") as f:
        json.dump(dataset, f, indent=1)


if __name__ == '__main__':
    build_dataset(
        os.path.join("data", "tagged_data_big", "corpus_tokens.txt"),
        os.path.join("data", "tagged_data_big", "corpus_pos.txt"),
        os.path.join("data", "dataset_big.json")
    )
