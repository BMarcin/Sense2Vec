import re

from tqdm import tqdm


class BasicProcessor:
    def __init__(self):
        self.space_regex = re.compile(r"(\s{2,})")
        self.html_regex = re.compile(r"<[^>]*>")
        self.date_regex = re.compile(
            r"((early\s\d{2,4})|(In\s\d{4})|(in\s\d{4})|(\d{4}s)|(\d{2,4}\s–\s\d{2,4})|(\d{1,2} [A-Z][a-z]+ \d{2,4})|(((January)|(February)|(March)|(April)|(May)|(June)|(July)|(October)|(September)|(August)|(November)|(December))\s+\d{1,4}(\s+,\s+\d{2,4})*)|(\d{1,4}\s+((January)|(February)|(March)|(April)|(May)|(June)|(July)|(October)|(September)|(August)|(November)|(December)))|(January)|(February)|(March)|(April)|(May)|(June)|(July)|(October)|(September)|(August)|(November)|(December))")
        self.number_regex = re.compile(r"((\d{1,4}(st|nd|rd|th)))")
        self.none_regex = re.compile(r"((\(.*\) )|(\[.*\] )|(\d+ – \d+))")
        self.digits_regex = re.compile(r"( \d+ )")
        self.left_digits_regex = re.compile(r"( \d+)")
        self.right_digits_regex = re.compile(r"(\d+ )")
        self.no_space_digits_regex = re.compile(r"(\d+)")
        self.limit_regex = re.compile(r"([^a-zA-Z0-9,\.<> !?äöüÄÖÜßùàûâüæÿçéèêëïîôœ])")
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
            r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#%]+\.([a-zA-Z]){2,6}([a-zA-Z0-9%\.\&\/\?\:@\-_=#])*")
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
            lambda x: re.sub(self.space_regex, " ", x),
            lambda x: re.sub(self.email_regex, "<email>", x),
            lambda x: re.sub(self.url_email_regex, "<web>", x),
            lambda x: re.sub(self.none_regex, "", x),
            lambda x: re.sub(self.date_regex, "<date>", x),
            lambda x: re.sub(self.number_regex, "<number>", x),
            lambda x: re.sub(self.digits_regex, " <number> ", x),
            lambda x: re.sub(self.left_digits_regex, " <number>", x),
            lambda x: re.sub(self.right_digits_regex, "<number> ", x),
            lambda x: re.sub(self.no_space_digits_regex, "<number>", x),
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
    text = """
Deutscher Bundestag  19/20754
Drucksache 
19. Wahlperiode 
01.07.2020
Entschließungsantrag 
der Abgeordneten Lorenz Gösta Beutin, Hubertus Zdebel, Ralph Lenkert,   a
Dr. Gesine Lötzsch, Heidrun Bluhm-Förster, Jörg Cezanne, Susanne Ferschl,   s
Sylvia Gabelmann, Kerstin Kassner, Dr. Achim Kessler, Katja Kipping, Jutta   s
Krellmann, Caren Lay, Sabine Leidig, Michael Leutert, Pascal Meiser, Cornelia  u
Möhring, Victor Perli, Ingrid Remmers, Dr. Kirsten Tackmann, Jessica Tatti,   n
Andreas Wagner, Harald Weinberg, Pia Zimmermann, Sabine Zimmermann 
(Zwickau) und der Fraktion DIE LINKE. 
zu der dritten Beratung des Gesetzentwurfs der Bundesregierung  
– Drucksachen 19/17342, 19/18472, 19/18779 Nr. 1.13, 19/20714 – 
Entwurf eines Gesetzes zur Reduzierung und zur Beendigung  
der Kohleverstromung und zur Änderung weiterer Gesetze   d
(Kohleausstiegsgesetz)  e
Der Bundestag wolle beschließen:  o
I. Der Deutsche Bundestag stellt fest:  e
Die Bundesregierung kündigt mit dem vorliegenden Kohleausstiegsgesetz den so ge- t
nannten Kohlekompromiss der Regierungskommission „Wachstum, Strukturwandel 
und Beschäftigung“ (KWSB, nachfolgend „Kohlekommission“) zu Gunsten der Koh-
leindustrie. Bereits die Empfehlungen der Kohlekommission, die ohne Beteiligung der 
parlamentarischen Opposition erarbeitet wurden, sind zur Einhaltung des völkerrecht-
lich verbindlichen Pariser Abkommens durch die Bundesrepublik Deutschland unzu- s
reichend. Die Abschaltungen der Kohleverstromung erfolgen mit Hinblick auf das ver- s
bleibende Treibhausgasbudget Deutschlands deutlich zu spät. Entschädigungen an Be- u
treiber werden teilweise unbegründet sowie auf fraglichen Grundlagen gezahlt und  n
verlängern ein fossiles Geschäftsmodell, das wegen des Fortschritts der Erneuerbaren 
Energien zunehmend nicht mehr wirtschaftlich möglich ist.  
Mit dem vorliegenden Gesetzentwurf ignoriert die Bundesregierung die Forderungen  e
der Mehrheit der Bevölkerung und der Klimaschutzbewegung nach einem Mehr an  r
Klimaschutz, das zur Erhaltung der natürlichen Lebensgrundlagen wissenschaftlich  s
belegt zwingend notwendig ist. Des Weiteren werden ohne Not neben dem Geset- e
zesentwurf öffentlich-rechtliche Verträge mit den Kohlekonzernen geschlossen, die  t
Drucksache 19/20754 – 2 –  Deutscher Bundestag – 19. Wahlperiode
spätere Korrekturen am Ausstiegspfad erschweren und auf Kosten von Steuerzahlen-
den und öffentlichen Haushalten verteuern werden. Dies stellt nichts anderes dar als  V
eine Entmachtung des Parlaments. 
II. Der Deutsche Bundestag fordert die Bundesregierung auf,  a
erneut einen Entwurf für ein Kohleausstiegsgesetz vorzulegen, welcher folgenden Kri-
terien genügt:  a
1. Der Ausstieg aus der Kohleverstromung beginnt unverzüglich mit der sofortigen 
Abschaltung der zwanzig emissionsintensivsten Kraftwerke und wird weiter ge-
führt mit stetigen und planmäßigen Stilllegungen von Kraftwerksblöcken auf der 
Basis von blockscharfen Restlaufzeiten bzw. Reststrommengen. Spätestens im  n
Jahr 2030 wird der letzte Kohlekraftwerksblock in Deutschland stillgelegt.  g
2. Der Neubau von Kohlekraftwerken und der Neuaufschluss von Tagebauen wer-  
den untersagt, das kürzlich neu in Betrieb genommene Steinkohlekraftwerk Dat-
teln 4 wird wieder vom Netz genommen.  w
3. Es werden für den Braunkohleabbau keine weiteren Dörfer mehr abgebaggert, 
der Hambacher Wald bleibt vom Braunkohleabbau unberührt, seine ökologische  r
Funktionsfähigkeit wird vom Tagebau nicht beeinträchtigt. 
4. Bei der arbeitsmarkt-, wirtschafts- und sozialpolitischen Begleitung des schritt- d
weisen Ausstiegs aus der Kohleverstromung ist den Empfehlungen der Kohle- u
kommission zu folgen, wobei insbesondere Interessenvertreter*innen der Be-
schäftigten und Anwohner*innen vor Ort und der Region wirksam einzubinden  c
sind.  h
5. Als Grundprinzip sind Entschädigungen an Betreiber nur für nachzuweisende tat-  
sächliche Mehrkosten infolge eines vorgezogenen Kohleausstiegs zu akzeptieren.  d
Entgangene Gewinne gehören nicht dazu. Es werden daher keine pauschalen  i
Stilllegungsprämien für Kraftwerksblöcke gezahlt, sondern gegebenenfalls regel-
basierte, welche zeitnah zum Abschaltzeitpunkt das Alter der Anlagen sowie das  l
energiewirtschaftliche Umfeld (Ertragslage ohne vorzeitige Stilllegung) berück-
sichtigen.  k
6. Die insolvenzfeste Absicherung der bergbaulichen Wiedernutzbarmachungs- und 
Nachsorgeverpflichtungen der Tagebaubetreiber ist durch Sicherheitsleistungen 
der Betreiber an die Bundesländer sowie durch eine Reform der Konzernhaftung 
zu gewährleisten. 
7. Das Verbot zur Errichtung und Inbetriebnahmen neuer Stein- und Braunkohlean-
lagen wird ergänzt durch ein analoges Verbot zur Errichtung und Inbetriebnahme  e
neuer Stein- und Braunkohleanlagen im Ausland durch Unternehmen mit Sitz in   
Deutschland. Der Export und Verkauf von Steinkohle- und Braunkohleförderan- F
lagen und entsprechender Technologie ins Ausland wird gesetzlich untersagt,  a
diesbezügliche Förderungen und Garantien des Bundes sind unzulässig. 
Berlin, den 30. Juni 2020  u
Amira Mohamed Ali, Dr. Dietmar Bartsch und Fraktion  g
Deutscher Bundestag – 19. Wahlperiode  – 3 –  Drucksache 19/…
Begründung 
I.  Klimaschutzpfad  o
Der Gesetzentwurf bewegt sich bei den Enddaten der Kohleverstromung in Deutschland mit dem Jahr 2038 (ge- r
gebenenfalls nach Überprüfung 2035) im Rahmen der Empfehlung der Kohlekommission. Diese Empfehlung ist 
jedoch nicht kompatibel mit einem klimagerechten Beitrag der deutschen Energiewirtschaft zum Erreichen der 
Pariser Klimaschutzziele. Danach soll die Erderwärmung gegenüber vorindustriellen Zeiten auf 2 Grad, mög- f
lichst 1,5 Grad, begrenzt werden, was deutlich frühere Abschaltungen erfordert. 
Innerhalb des Zeitrahmens bis zum Enddatum weicht das Gesetz überdies von der Empfehlung der Kohlekom-
mission zu Lasten des Klimaschutzes ab. Zwar wird formal die Minderung der Kraftwerksleistung auf den je-
weiligen Kraftwerksleistungs-Umfang der so genannten Stützjahre 2022, 2030 und 2038 eingehalten. Da aber 
bei der besonders emissionsintensiven Braunkohle jeweils erst kurz vor den vereinbarten Stützjahren abgeschaltet n
werden soll, wird die Atmosphäre bei dieser „Kaskaden-Abschaltung“ ungleich mehr mit Treibhausgasen bela- g
den, als bei einer stetigen Abschaltung, welche laut Empfehlung Kohlekommission angestrebt werden soll. Die  
Mehremissionen gegenüber stetigen Abschaltungen werden vom Deutschen Institut für Wirtschaftsforschung -
(DIW) auf kumulativ auf bis zu 134 Mio. t CO  geschätzt. Diese Kaskaden-Abschaltungen vergrößern zudem w
die Herausforderungen mit Blick auf Systemdienstleistungen und Versorgungssicherheit dahingehend, dass Um-
stellung in einem großen Umfang jeweils in einem kurzen Zeitraum stattfinden müssen.  r
Die Bundesregierung versucht im Gesetzentwurf, ein Teil der Mehremissionen im Braunkohlebereich über eine 
frühzeitigere bzw. stärkere Stilllegung von Steinkohlekraftwerken zu kompensieren. Dies kollidiert allerdings  
mit der ebenfalls im Abschlussbericht der Kohlekommission verankerten Forderung, Steinkohlekapazitäten stetig 
stillzulegen. Zudem wird es somit aufwändiger, regional die Versorgungssicherheit (insbesondere der Wär-
meauskopplung der Stadtwerke) zu kompensieren. Insgesamt ergibt sich hier aus der Klimaschutzperspektive r
eine verkehrte Welt: Die Steinkohleverstromung wird früher beendet als die Braunkohleverstromung, obwohl 
die Verstromung der Braunkohle rund 25 Prozent emissionsintensiver ist als die der Steinkohle. 
Der Kohleausstieg in der Lausitz wird zeitlich am stärksten nach hinten verschoben gegenüber einem linearen d
Abschaltpfad. Ferner werden Jahr im Jahr 2025 in der Lausitz statt der in Kohlekommission vereinbarten 10 Mio. 
t CO  als ersten Zwischenschritt vorrausichtlich nur 2,5 Mio. t CO  eingespart.   e
2 2
Im Rheinland kommt es vergleichsweise früh zu relevanten Kraftwerksstilllegungen. Der Hambacher Wald wäre 
mit bekanntgewordenen Formulierungshilfen für die Endabstimmung des Gesetzes und des öffentlich-rechtlichen e
Vertrags (örV) zwischen der Bundesregierung und den Kohlekonzernen formell gerettet. Das Dorf Manheim k
würde jedoch weiterhin unnötig abgebaggert.   t
Vor allem aber soll der Tagebau Garzweiler nach wie vor im Umfang der Leitentscheidung aus dem Jahr 2016 
bis 2038 ausgekohlt werden, wobei - ein einmaliger Vorgang – die energiewirtschaftliche Notwendigkeit des 
Tagebaus ohne weiter Prüfung mit diesem Gesetz festgesetzt werden soll. Nach einer Analyse des DIW für ein e
Paris-kompatibles nationales CO -Budget lässt sich für NRW eine Höchstmenge von 280 Mio. Tonnen an Braun- r
kohle errechnen, die dort noch gefördert werden dürfte, um dem Ziel gerecht zu werden, die globale Erhitzung 
auf maximal 1,75°C zu begrenzen. Dies steht im Widerspruch dazu, Garzweiler in den Grenzen der Leitentschei-
dung 2016 bis zum Jahr 2038 auszukohlen. Ohnehin ist es ein völlig neues Verfahren, die energiewirtschaftliche F
Notwendigkeit eines Tagebaus einfach und ungeprüft in einem örV zwischen Bundesregierung und Betreibern 
feststellen zu lassen. 
Gleichzeitig wird mit dem Kohleausstiegsgesetz die Unsicherheit von Beschäftigten und Regionen über tatsäch-
liche Abschaltzeitpunkte auf Jahre verlängert. Schließlich rechnen Analysten damit, dass die Braunkohle aus 
wirtschaftlichen Gründen weit vor 2035 aus dem Markt gedrängt werden könnte. Schon heute erwirtschaften 
nach Untersuchungen des Öko-Instituts ältere Braunkohlekraftwerksblöcke ihre fixen Betriebskosten nur knapp n
und jegliche fixe Betriebskosten der Braunkohletagebaue sowie die Renaturierungskosten (jeweils anteilig) in g
keiner Weise. Neue Braunkohlekraftwerke erwirtschaften ihre fixen Betriebskosten und die kurzfristig abbauba-  
ren fixen Betriebskosten der Tagebaue (anteilig) noch voll, die mittelfristig abbaubaren fixen Betriebskosten 
(anteilig) werden aber nur noch teilweise abgedeckt und leisten keinerlei Beitrag mehr zur Refinanzierung der r
Investitionskosten. Die angekündigte Verschärfung der EU-Klimaschutzziele wird die wirtschaftliche Lage der 
Braunkohle zusätzlich verschlechtern, insbesondere durch die Kombination steigender CO -Preise im EU-Emis-
sionshandelssystem und Verdrängungseffekten des Ökostromausbaus.  t
Drucksache 19/20754 – 4 –  Deutscher Bundestag – 19. Wahlperiode
Das Kohleausstiegsgesetz verzögert nicht nur den Kohleausstieg, mit Datteln 4 kann unverständlicherweise gar 
ein neues Steinkohlekraftwerk ans Netz gehen. Der entsprechend zusätzliche CO -Ausstoß soll über zusätzliche V
Stilllegungen im Steinkohlebereich ausgeglichen werden. Das passiert aber nur zu 75 Prozent. 
II.  Entschädigungen an Betreiber / Sicherung der Widerherstellung  a
Die im Kohleausstiegsgesetz festgelegten und mit den Betreibern verhandelten Abschaltzeitpunkte entsprechen b
in der Lausitz ungefähr denen der Planung des dortigen Betreibers LEAG zum Zeitpunkt des Eigentumsüber- f
gangs von Vattenfall zur LEAG im Jahr 2016 („Businessplanungs-Szenario 1A“). Dennoch soll der Konzern 
1,75 Mrd. Euro Entschädigung erhalten obgleich viele Anlagen ein hohes Alter erreicht haben. Es drängt sich s
hier die Frage auf: Wofür sollen diese Entschädigungen eigentlich gezahlt werden? Auch Anlagen des rheini- s
schen Betreibers RWE sind weitgehend abgeschrieben. Gezahlt werden sollen hier 2,6 Mrd. Euro. Überdies sind u
die Entschädigungssummen für die LEAG je Gigawatt (GW) mit 580 Mio. €/GW höher als für RWE (470 Mio. n
€/GW), obgleich die LEAG (kapazitätsgewichtet) im Durchschnitt drei Jahre später abschaltet. Offensichtlich 
orientieren sich im Braunkohlebereich die verhandelten Entschädigungen nicht an nachvollziehbaren Regeln, 
sondern an politischen Deals. Hier stellt sich neben der Rechtfertigung auch die Frage, inwieweit solche Entschä- -
digung vor europäischen Beihilferecht Bestand haben werden.   
Nach den bekanntgewordenen Formulierungshilfen für die Endabstimmung des Gesetzes und des und des dazu-
gehörigen örV sollen nun Entschädigungen für das Lausitzer Revier zunächst direkt an jene Zweckgesellschaften ir
in Brandenburg bzw. Sachsen gehen, welche zur finanziellen Absicherung der bergbaulichen Wiedernutzbarma- d
chungs- und Nachsorgeverpflichtungen von Ländern und Betreiber gemeinsam gegründet wurden. Dabei werden  
jedoch lediglich 10 Prozent der Entschädigungszahlungen an Treuhänder gezahlt. Nach Ansicht von Experten d
wäre demgegenüber bundesweit die Zahlung von Sicherheitsleistungen der Tagebaubetreiber direkt an die be- u
treffenden Länder die sicherste Absicherung gegen Insolvenzverluste oder regelwidrige Transfers. Zudem muss r
der gesetzliche Rahmen für die Konzernhaftung so verändert werden, dass bei Insolvenzen von Tochterunterneh- c
men von Kohlekonzernen die zwingend Mutterunternehmen für deren Zukunftsverpflichtungen eintreten.  h
III.  Öffentlich-rechtlicher Vertrag 
Zu den umstrittensten Paragrafen des Kohleausstiegsgesetzes gehört § 42 des Gesetzentwurfs in der Fassung der 
Drucksache 19/17342, nachdem wesentliche Inhalte des Kohleausstiegs zwischen der Bundesregierung und den  
Betreibern in einem öffentlich-rechtlichen Vertrag statt im Gesetz selbst festgelegt werden. So unter anderem der e
eigentliche Stilllegungspfad, die Festlegung und Verwendung von Entschädigungszahlungen für die Braunkoh- k
leunternehmen, welche, wie geschildert, in der Höhe nicht regelbasiert sind, sondern willkürlich erscheinen. Dem t
Bundestag werden durch die pauschale Festlegung der Entschädigungssumme Beteiligungsrechte im Hinblick o
auf erhebliche Haushaltsmittel genommen. Aufgrund der zu erwartenden Entwertung der Braunkohleanlagen r
beim weiteren Ausbau erneuerbarer Energien sowie aufgrund der in Brüssel angekündigten Verschärfung der e
europäischen Klimaschutzziele am Ende Entschädigungen gezahlt, die weit über den Verlusten der Betreiber 
liegen, welche sie am Markt ohne diesen Gesetz erleiden würden. Zudem könnten die hohen Entschädigungen t
verhindern, dass Kraftwerksblöcke aus dem Markt gehen, die unrentabel werden.  
Äußerst kritisch ist ferner der Punkt im örV, in dem Kriterien und Rechtsfolgen (einschließlich Entschädigungs-
fragen) nachträglicher Eingriffe in die Braunkohleverstromung festgelegt werden. Angesichts dessen, dass der 
ohnehin nicht Paris-kompatible Ausstiegspfad des Kohleausstiegsgesetzes in Zukunft durch schärfere EU-Kli-
maschutzziele angepasst werden wird, besteht die Gefahr, dass hier Regeln im örV festgezurrt werden, die nach- s
träglich (ggf. durch eine andere Regierung) nur mit hohen weiteren Entschädigungszahlungen verändert werden s
könnten. In den letzten Stunden des Gesetzgebungsprozesses hat die Bundesregierung den Koalitionsfraktionen u
hier zwar Präzisierungen und Änderungen vorgelegt, nach denen etwaige Entschädigungszahlungen in vielen n
Fällen ausgeschlossen werden sollen. Ob diese Änderungen allerdings das angestrebte Ziel in ausreichender 
Weise sichern, ist in der Kürze der Zeit nicht mehr zu bewerten. Ohnehin müssen auch nach diesen Änderungen 
Laufzeitverkürzungen mindestens fünf bis acht Jahre vorher angekündigt werden, um entschädigungslos erfolgen e
zu können. 
 """
    title_regex = re.compile("((= )+.*(= )+)")
    bp = BasicProcessor()

    print(bp.process(
        [line for line in text.split("\n")],
    ))
