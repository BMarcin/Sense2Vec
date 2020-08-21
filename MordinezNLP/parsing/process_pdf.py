from io import BytesIO

import pdfplumber


def process_pdf(pdf_input: BytesIO):
    doc_output = []
    with pdfplumber.open(pdf_input) as pdf:
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            tables = page.extract_tables()

            ' extract words from PDF '
            ' dont use extract_text method because of overlapping '
            ' horizontal and vertical characters '
            text = " ".join([item['text'] for item in page.extract_words()])

            table_tokens = []
            unique_table_tokens = set()

            ' remove text which is in tables '
            ' first, store texts from table as string '
            for table in tables:
                for row in table:
                    for col in row:
                        if col != ' ' and col != '' and col is not None:
                            for item in col.split("\n"):
                                if len(set(item)) > 1:
                                    table_tokens.append(item)
                                    unique_table_tokens.add(item)

            unique_table_tokens = list(unique_table_tokens)

            # instead of .copy() use [:]
            original_text = text[:]

            for cnt, token in enumerate(unique_table_tokens):
                replace_token = token[:]
                ' make the same replacement on tokens to replace, that were made on source text '
                for i in range(cnt):
                    replace_token = replace_token.replace(unique_table_tokens[i], " ")

                ' finally check if string to replace matches count in whole page '
                ' with it we can define if text occurs only in table '
                if original_text.count(token) == unique_table_tokens.count(token):
                    text = text.replace(replace_token, " ")

            ' return not null and non space lines '
            to_process = [line for line in text.split("\n") if len(set(line)) > 1]
            doc_output.append("\n".join(to_process))
    return "\n".join(doc_output)


if __name__ == '__main__':
    import requests
    pdf = BytesIO(requests.get("http://dipbt.bundestag.de/dip21/btd/19/207/1920757.pdf").content)
    output = process_pdf(pdf)
    print(output)
