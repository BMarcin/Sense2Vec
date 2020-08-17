import json
from optparse import OptionParser

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option(
        "--input_json",
        dest="input_json",
        help="Sense2Vec vectors in JSON file format",
        metavar="FILE"
    )

    options, args = parser.parse_args()

    vectors = json.loads(open(options.input_json).read())

    while True:
        user_input = input("Provide token please: ")
        if user_input == 'q':
            break
        top_n = input("Top ?: ")

        if user_input not in vectors.keys():
            print("There is no such word in vocabulary")
        else:
            target = vectors[user_input]

            similarities = {}
            for token, vector in tqdm(vectors.items(), desc="Calculating"):
                if vector != 0.0:
                    similarity = cosine_similarity(
                        [target],
                        [vector]
                    )
                    similarities[token] = similarity[0][0]

            similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))

            for token in list(similarities.keys())[:int(top_n)]:
                print(token, similarities[token])
