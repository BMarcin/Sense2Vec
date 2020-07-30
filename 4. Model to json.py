import os
import json
from optparse import OptionParser

import torch
from tqdm import tqdm

from Sense2Vec.Sense2VecCBOW import Sense2VecCBOW

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option(
        "--model_pickle_path",
        dest="model_pickle_path",
        help="Path to saved model",
        metavar="FILE"
    )

    parser.add_option(
        "--embeddings_size",
        dest="embeddings_size",
        help="Embedding vector size",
        type=int
    )

    parser.add_option(
        "--target_vectors",
        dest="target_vectors",
        help="Target vector size for each token",
        type=int
    )

    parser.add_option(
        "--dataset_pickle_path",
        dest="dataset_pickle_path",
        help="Path to save pickled dataset",
        metavar="FILE"
    )

    parser.add_option(
        "--output_file_path",
        dest="output_file_path",
        help="Path to target JSON file",
        metavar="FILE"
    )

    options, args = parser.parse_args()

    token2idx = torch.load(options.dataset_pickle_path)
    device = torch.device("cpu")
    model = Sense2VecCBOW(
        len(token2idx),
        options.embeddings_size,
        options.target_vectors,
        5
    ).to(device)

    model.load_state_dict(torch.load(options.model_pickle_path))
    weights = model.get_weights()

    sense2vec = {}

    for token, weight in tqdm(zip(list(token2idx.keys()), weights), total=len(weights)):
        sense2vec[token] = weight

    with open(options.output_file_path, 'w', encoding="utf8") as f:
        json.dump(sense2vec, f, indent=1)
