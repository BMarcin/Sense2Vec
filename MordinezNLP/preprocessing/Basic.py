import re

from tqdm import tqdm


class BasicProcessor:
    def __init__(self):
        self.months = [
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'October', 'September', 'August',
            'November',
            'December',  # en
            'Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November',
            'Dezember'  # de
        ]

        ' make unique '
        self.months = list(set(self.months))
        months_builded = "|".join(["({month})".format(month=month) for month in self.months])

        self.more_less_left = re.compile(r"([^\s])([<>])")
        self.more_less_right = re.compile(r"([<>])([^\s])")
        self.space_regex = re.compile(r"(\s{2,})")
        self.html_regex = re.compile(r"<[^>]*>")
        self.date_regex = re.compile(
            r"((early\s\d{2,4})|(In\s\d{4})|(in\s\d{4})|(\d{4}s)|(\s\d{2,4}\s–\s\d{2,4})|(\s\d{1,2} [A-Z][a-z]+ \d{2,4})|( \d{2}\.\d{2}\.\d{2,4} )|((" + months_builded + ")\s+\d{1,4}(\s+,\s+\d{2,4})*)|(\d{1,4}\s+(" + months_builded + "))|" + months_builded + ")")
        self.number_regex = re.compile(r"((\d{1,4}(st|nd|rd|th))|((Nr|Num)\s*\.\s*(\d*\.\d*)*))")
        self.double_upper_case_letters = re.compile(r"([A-Z][^\s^A-Z^\-^.^,^?^!]+)([A-Z][^\s^A-Z^\-^.^,^?^!]+)")
        self.none_regex = re.compile(r"( \(([^)]+)\)|(\[.*\] )|(\d+ – \d+))")
        self.digits_regex = re.compile(r"( \d+ )")
        # self.left_digits_regex = re.compile(r"( \d+)")
        # self.right_digits_regex = re.compile(r"(\d+ )")
        self.no_space_digits_regex = re.compile(r"(\d+)")
        self.limit_regex = re.compile(r"(([^a-zA-Z0-9,\.<> !?äöüÄÖÜßùàûâüæÿçéèêëïîôœ])|([^\s]{30,}))")
        self.space_before_regex_fix = re.compile(r"^(\s)")
        self.dot_regex = re.compile(r"\s+\.")
        self.com_regex = re.compile(r"\s+,")
        self.multi_com_regex = re.compile(r",+")
        self.multi_dot_regex = re.compile(r"\.+")
        self.multi_tag_regex = re.compile(r"((<unk>|<number>|<date>|<web>|<email>|<more>|<less>)[ ]*){2,}")
        self.starting_space_regex = re.compile(r"^( +)")
        self.ending_space_regex = re.compile(r"( +)$")
        self.special_token_with_text_pre = re.compile(
            r"[\s,\.!?]*([a-zA-Z0-9,\.<>!?]+)(<number>|<date>|<unknown>|<web>|<email>|<more>|<less>)[\s,\.!?]*")
        self.special_token_with_text_post = re.compile(
            r"[\s,\.!?]*(<number>|<date>|<unknown>|<web>|<email>|<more>|<less>)([a-zA-Z0-9,\.<>!?]+)[\s,\.!?]*")
        self.hex_regex = re.compile(r"[^\x20-\x7e]")
        self.email_regex = re.compile(
            r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*")
        self.url_email_regex = re.compile(
            r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&\/=]*)")
        self.separation_regex = re.compile(r"(.)([\.,\?\!])(.|$)")
        self.more_than_regex = re.compile(r"<number>\s[>]\s<number>")
        self.less_than_regex = re.compile(r"<number>\s[<]\s<number>")

        self.multiple_characters_regex = re.compile(r"(.)\1{3,}")
        self.multiple_characters_non_sense = re.compile(
            r"(<number>|<date>|<unknown>|<web>|<email>|<more>|<less>)|[,\.<>!?]")

    def process_multiple_characters(self, text_to_process):
        """
        Function replaces two types of entities -> to form:
        'EEEEEEEEEEEE!' -> ''
        'supeeeeeer' -> 'super'

        Other examples:
        'EEEE<number>!' -> ''
        'suppppprrrrrpper' -> 'suprpper'
        :param text: text to process
        :return: post processed text
        """

        for entity in re.findall(self.multiple_characters_regex, text_to_process):
            match = ""
            if match not in ['.']:
                for match in re.findall("(([^" + entity + "^\s.]*)([" + entity + "]{3,})([^" + entity + "^\s.]*))",
                                        text_to_process):
                    text_replaced = re.sub(self.multiple_characters_non_sense, "", match[0])
                    if text_replaced == match[2]:
                        text_to_process = text_to_process.replace(match[0], "")
                    else:
                        text_to_process = text_to_process.replace(match[1] + match[2] + match[3],
                                                                  match[1] + entity + match[3])

        return text_to_process

    def process(self, input_data_list, pre_rules=[], post_rules=[]):
        """
        Function that preprocesses list of string. Additionaly according to special tokens in dataset user can add custom_rule,
        which is a lambda function. Example is definied in function body.
        Some values like numbers or dates are replaced by special token like <number> or <date>
        :param input_data_list: text list to preprocess
        :param custom_rules: list of lambda functions to add custom preprocessing
        :return: lists of strings
        """
        output = []
        rules = [
            lambda x: re.sub(self.html_regex, "", x),
            lambda x: re.sub(self.more_less_right, r"\1 \2 ", x),
            lambda x: re.sub(self.more_less_left, r" \1 \2", x),
            lambda x: re.sub(self.space_regex, " ", x),
            lambda x: re.sub(self.email_regex, " <email> ", x),
            lambda x: re.sub(self.url_email_regex, " <web> ", x),
            lambda x: re.sub(self.double_upper_case_letters, r" \1 \2 ", x),
            lambda x: re.sub(self.none_regex, " ", x),
            lambda x: re.sub(self.date_regex, " <date> ", x, re.IGNORECASE),
            lambda x: re.sub(self.number_regex, " <number> ", x, re.IGNORECASE),
            lambda x: re.sub(self.digits_regex, " <number> ", x),
            # lambda x: re.sub(self.left_digits_regex, " <number>", x),
            # lambda x: re.sub(self.right_digits_regex, "<number> ", x),
            lambda x: re.sub(self.no_space_digits_regex, " <number> ", x),
            lambda x: re.sub(self.limit_regex, " ", x),
            lambda x: re.sub(self.dot_regex, ".", x),
            lambda x: re.sub(self.com_regex, ",", x),
            lambda x: re.sub(self.multi_com_regex, ",", x),
            lambda x: re.sub(self.multi_dot_regex, ".", x),
            lambda x: re.sub(self.multi_tag_regex, r"\1 ", x),
            lambda x: re.sub(self.space_before_regex_fix, "", x),
            lambda x: re.sub(self.space_regex, " ", x),
            lambda x: x.replace("<unk>", "<unknown>"),
            lambda x: re.sub(self.starting_space_regex, "", x),
            lambda x: re.sub(self.ending_space_regex, "", x),
            lambda x: re.sub(self.special_token_with_text_pre, r' \1 \2 ', x),
            lambda x: re.sub(self.special_token_with_text_post, r' \1 \2 ', x),
            lambda x: x.replace(",.", "."),
            lambda x: x.replace(" 's", "'s"),
            lambda x: x.replace(" 't", "'t"),
            lambda x: x.replace("' ", "'"),
            lambda x: x.replace(".,", "."),
            lambda x: re.sub(self.multi_tag_regex, "\1 ", x),
            # lambda x: re.sub(self.hex_regex, "", x),
            lambda x: self.process_multiple_characters(x),
            lambda x: re.sub(self.separation_regex, r"\1 \2 \3", x),
            lambda x: re.sub(self.separation_regex, r"\1 \2 \3", x),
            lambda x: re.sub(self.more_than_regex, "<number> <more> <number>", x),
            lambda x: re.sub(self.less_than_regex, "<number> <less> <number>", x),
            lambda x: re.sub(self.space_regex, " ", x),
            lambda x: x.replace(" > ", " "),
            lambda x: x.replace(" < ", " "),
        ]
        rules = pre_rules + rules + post_rules + [lambda x: re.sub(self.space_regex, " ", x)]

        # todo to fix
        # day.i
        # o.o
        # .a
        # dosnt
        # .<number>.<number>
        # sh!t

        # --most least--
        # antiradicalloudyellingcatchyslogansthatmeannothingyoudontknowwhatthehellyoureevenprotesting

        for i, item in enumerate(tqdm(input_data_list, desc="Processing input")):
            post_regex = rules[0](item)
            for rule in rules[1:]:
                post_regex = rule(post_regex)

            if len(post_regex) > 0:
                output.append(post_regex)

        return output


if __name__ == '__main__':
    text = """ an Arbeitsproduktivität (Ausfall an Bruttowertschöpfung)  >ausgefallene Bruttowertschöpfung ............................................................................................ 145Mrd. €  >Ausfall an 
 """
    title_regex = re.compile("((= )+.*(= )+)")
    bp = BasicProcessor()

    print(bp.process(
        [line for line in text.split("\n")],
    ))
