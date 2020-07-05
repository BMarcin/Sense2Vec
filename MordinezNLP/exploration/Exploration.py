from collections import Counter
import spacy
from tqdm import tqdm
from spellchecker import SpellChecker

import xlsxwriter
import logging

from MordinezNLP.tokenizers.spacy_tokenizer_tag import spacy_tokenizer_tag


class Exploration:
    def __init__(self, text, spacy_module="en_core_web_sm"):
        self.text_data = text
        self.counter = Counter()
        self.counter_non_stop = Counter()
        self.counter_tokens_without_vectors = Counter()
        self.counter_bigrams = Counter()
        self.counter_bigrams_non_stop = Counter()
        self.counter_trigrams = Counter()
        self.counter_trigrams_non_stop = Counter()
        self.counter_sentence_length = Counter()
        self.counter_word_length = Counter()
        self.word_lengths = []

        self.nlp = spacy.load(spacy_module)
        self.nlp.tokenizer = spacy_tokenizer_tag(self.nlp)
        self.spell_checker = SpellChecker()

        self.anomalies = []
        self.unique_characters = set()

        logging.info("Starting processing...")
        for doc in tqdm(self.nlp.pipe(self.text_data, disable=["ner", "tagger", "textcat"], n_process=8,
                                      batch_size=200), desc="Processing", total=len(self.text_data)):
            for sent in doc.sents:
                tokens = []
                tokens_non_stop = []

                self.counter_sentence_length[str(len(sent))] += 1

                ' set minimum length of sentence '
                if len(sent) > 3:
                    for token in sent:
                        if token.text == '<' or token.text == '>':
                            self.anomalies.append(sent)
                        if token.text == "number>" or \
                                token.text == "<date" or \
                                token.text == "<number" or \
                                token.text == "date>" or \
                                token.text == "<unknown" or \
                                token.text == "unknown>":
                            self.anomalies.append(sent)

                        if not token.is_punct:
                            self.counter[token.text.lower()] += 1
                            tokens.append(token.text.lower())

                            ' lengths '
                            self.word_lengths.append(len(token.text))
                            self.counter_word_length[str(len(token.text))] += 1

                            for letter in token.text:
                                self.unique_characters.add(letter)

                            if not token.is_stop:
                                tokens_non_stop.append(token.text.lower())
                                self.counter_non_stop[token.text.lower()] += 1

                            if token.text.lower() not in self.spell_checker:
                                self.counter_tokens_without_vectors[token.text.lower()] += 1
                else:
                    logging.error("Sentence too short: {}".format(sent))

                ' bigrams and trigrams processing '
                for i in range(0, len(tokens) - 1):
                    bigram = tokens[i:i + 2]
                    if len(bigram) == 2:
                        self.counter_bigrams[" ".join(bigram)] += 1

                for i in range(0, len(tokens_non_stop) - 1):
                    bigram = tokens_non_stop[i:i + 2]
                    if len(bigram) == 2:
                        self.counter_bigrams_non_stop[" ".join(bigram)] += 1

                for i in range(0, len(tokens) - 2):
                    trigram = tokens[i:i + 3]
                    if len(trigram) == 3:
                        self.counter_trigrams[" ".join(trigram)] += 1

                for i in range(0, len(tokens) - 2):
                    trigram = tokens_non_stop[i:i + 3]
                    if len(trigram) == 3:
                        self.counter_trigrams_non_stop[" ".join(trigram)] += 1

    def get_most_common(self, n):
        return self.counter.most_common(n)

    def get_most_common_non_stop(self, n):
        return self.counter_non_stop.most_common(n)

    def get_most_least(self, n):
        return self.counter.most_common()[:-n - 1:-1]

    def get_most_least_non_stop(self, n):
        return self.counter_non_stop.most_common()[:-n - 1:-1]

    def get_most_common_abnormal(self, n):
        return self.counter_tokens_without_vectors.most_common(n)

    def get_most_least_abnormal(self, n):
        return self.counter_tokens_without_vectors.most_common()[:-n - 1:-1]

    def get_most_common_bigrams(self, n):
        return self.counter_bigrams.most_common(n)

    def get_most_common_bigrams_non_stop(self, n):
        return self.counter_bigrams_non_stop.most_common(n)

    def get_most_least_bigrams(self, n):
        return self.counter_bigrams.most_common()[:-n - 1:-1]

    def get_most_least_bigrams_non_stop(self, n):
        return self.counter_bigrams_non_stop.most_common()[:-n - 1:-1]

    def get_most_common_trigrams(self, n):
        return self.counter_trigrams.most_common(n)

    def get_most_common_trigrams_non_stop(self, n):
        return self.counter_trigrams_non_stop.most_common(n)

    def get_most_least_trigrams(self, n):
        return self.counter_trigrams.most_common()[:-n - 1:-1]

    def get_most_least_trigrams_non_stop(self, n):
        return self.counter_trigrams_non_stop.most_common()[:-n - 1:-1]

    def get_word_lengths(self):
        return self.counter_word_length.most_common()

    def get_sentence_lengths(self):
        return self.counter_sentence_length.most_common()

    def _write_sheet_with_chart_for_tokens_counts(self, chart_name, xlsx_file, counter_values, plot_counter_values=100, c1_name='Token', c2_name='Count'):
        # todo asserts for plot counter values
        sheet = xlsx_file.add_worksheet(chart_name)

        sheet.write(0, 0, c1_name)
        sheet.write(0, 1, c2_name)

        for i, (token, count) in enumerate(counter_values):
            sheet.write(i + 1, 0, token)
            sheet.write(i + 1, 1, count)

        chart = xlsx_file.add_chart({'type': 'column'})
        chart.add_series({
            'values': "='{}'!$B$2:$B${}".format(chart_name, plot_counter_values),
            'categories': "='{}'!$A$2:$A${}".format(chart_name, plot_counter_values),
            'name': "='{}'!$A$1".format(chart_name),
            'data_labels': {'value': True}
        })
        chart.set_title({'name': chart_name})
        chart.set_y_axis({'name': 'Count'})
        chart.set_legend({'position': 'none'})

        sheet.insert_chart('D1', chart, {'x_scale': 4, 'y_scale': 3})

    def write_to_xlsx(self, xlsx_path, n_results=2000, plot_results=200):
        logging.info("Writing to xlsx")
        xlsx_file = xlsxwriter.Workbook(xlsx_path)

        self._write_sheet_with_chart_for_tokens_counts(
            'Most common tokens',
            xlsx_file,
            self.get_most_common(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most common tokens non stops',
            xlsx_file,
            self.get_most_common_non_stop(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most least tokens',
            xlsx_file,
            self.get_most_least(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most least tokens non stops',
            xlsx_file,
            self.get_most_least_non_stop(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most common abnormal',
            xlsx_file,
            self.get_most_common_abnormal(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most least abnormal',
            xlsx_file,
            self.get_most_least_abnormal(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most common bigrams',
            xlsx_file,
            self.get_most_common_bigrams(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most common bigrams non stops',
            xlsx_file,
            self.get_most_common_bigrams_non_stop(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most least bigrams',
            xlsx_file,
            self.get_most_least_bigrams(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most least bigrams non stops',
            xlsx_file,
            self.get_most_least_bigrams_non_stop(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most common trigrams',
            xlsx_file,
            self.get_most_common_trigrams(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most common trigrams non stops',
            xlsx_file,
            self.get_most_common_trigrams_non_stop(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most least trigrams',
            xlsx_file,
            self.get_most_least_trigrams(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Most least trigrams non stops',
            xlsx_file,
            self.get_most_least_trigrams_non_stop(n_results),
            plot_counter_values=plot_results
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Word length distribution',
            xlsx_file,
            self.get_word_lengths(),
            c1_name='Word length',
            c2_name='Count'
        )

        self._write_sheet_with_chart_for_tokens_counts(
            'Sentence length distribution',
            xlsx_file,
            self.get_sentence_lengths(),
            c1_name='Sentence length',
            c2_name='Count'
        )

        xlsx_file.close()
        logging.info("Writing to xlsx. Done.")

    def print(self):
        print("Most common: ", self.get_most_common(20), "\n")
        print("Most common non stop: ", self.get_most_common_non_stop(20), "\n")
        print("Most least: ", self.get_most_least(20), "\n")
        print("Most least non stop: ", self.get_most_least_non_stop(20), "\n")
        print("Most common abnormal: ", self.get_most_common_abnormal(20), "\n")
        print("Most least abnormal: ", self.get_most_least_abnormal(20), "\n")
        print("Most common bigrams: ", self.get_most_common_bigrams(20), "\n")
        print("Most common bigrams non stop: ", self.get_most_common_bigrams_non_stop(20), "\n")
        print("Most least bigrams: ", self.get_most_least_bigrams(20), "\n")
        print("Most least bigrams non stop: ", self.get_most_least_bigrams(20), "\n")
        print("Most common trigrams: ", self.get_most_common_trigrams(20), "\n")
        print("Most common trigrams non stop: ", self.get_most_common_trigrams_non_stop(20), "\n")
        print("Most least trigrams: ", self.get_most_least_trigrams(20), "\n")
        print("Most least trigrams non stop: ", self.get_most_least_trigrams_non_stop(20), "\n")


if __name__ == '__main__':
    text_to_process = [
        "In 2011 , the <unk> utilized a committee of running backs , with <unk> , Daniel Porter , and Jerome <unk> all receiving significant playing time . <unk> was used mostly in short @-@ yardage situations on the ground , while also being active as a receiver and on special teams . He played in 18 games , made eight starts , and finished with <unk> yards on 52 carries with no touchdowns . He also caught 22 passes for 150 yards and a touchdown . <unk> played in both of the <unk> ' playoff games . In the West <unk> @-@ Finals against the <unk> , he rushed for a goal @-@ line touchdown , in addition to making three receptions and two special @-@ teams tackles . <unk> played a more limited role in the West Finals against the BC Lions , where he was given only one carry for six yards , made one tackle on special teams , and caught two passes for a total of four yards ."
    ]

    nlp_exp = Exploration(text_to_process)
    # nlp_exp.print()
    nlp_exp.write_to_xlsx("C:\\_AI\\Sense2Vec\\results.xlsx")
