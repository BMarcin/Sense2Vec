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
        self.limit_regex = re.compile(r"([^a-zA-Z0-9,\.<> !?])")
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
            lambda x: re.sub(self.limit_regex, "", x),
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
            lambda x: re.sub(self.hex_regex, "", x),
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
    = Tropical Storm <unk> ( 2008 ) = 
 It was 12cm in 12 xd12.
 Tropical Storm <unk> was the tenth tropical storm of the 2008 Atlantic hurricane season . <unk> developed out of a strong tropical wave which moved off the African coast on August 31 . The wave quickly became organized and was declared Tropical Depression Ten while located 170 mi ( 270 km ) to the south @-@ southeast of the Cape Verde Islands on September 2 . The depression was quickly upgraded to Tropical Storm <unk> around noon the same day . Over the next several days , <unk> moved in a general west @-@ northwest direction and reached its peak intensity early on September 3 . Strong wind shear , some due to the outflow of Hurricane Ike , and dry air caused the storm to weaken . On September 6 , the combination of wind shear , dry air , and cooling waters caused <unk> to weaken into a tropical depression . <unk> deteriorated into a remnant low shortly after as convection continued to dissipate around the storm . The low ultimately dissipated while located 520 mi ( 835 km ) east of <unk> on September 10 . However , the remnant moisture led to minor flooding on the island of St. Croix . 
 
 = = Meteorological history = = 
 
 Tropical Storm <unk> formed as a tropical wave that emerged off the west coast of Africa near the end of August 2008 . It tracked south of Cape Verde and slowly developed , and on September 2 the disturbance became Tropical Depression Ten while located south @-@ southeast of the Cape Verde islands . As the depression became more organized , an eye @-@ like feature developed in the upper levels of the system . The depression was upgraded to Tropical Storm <unk> six hours after forming . <unk> was located in an area which was supportive for rapid intensification but was not forecast to intensify quickly . 
 <unk> continued to intensify throughout the afternoon as the storm became more symmetrical . However , due to the location of the storm , there was a lack of accurate wind speed readings , and the National Hurricane Center was uncertain of its actual intensity . Despite the lack of wind shear around the storm , the center became slightly exposed and ceased further intensification . The storm was also heading into an area where shear was <unk> to significantly increase due to an upper @-@ level trough diving southward . Despite convection being partially removed from the center of <unk> , the storm intensified slightly in the early morning hours on September 3 as thunderstorm activity to the south of the center became more organized . The intensification was forecast to be short in duration as the trough to the north was deepening , causing the wind shear to the west to become stronger . 
 <unk> reached its peak intensity of 65 mph ( 100 km / h ) around 8 a.m. ( <unk> ) as it continued to become more organized . However , there were indications that it had already begun to weaken . <unk> towards the north was becoming restricted and arc clouds began emanating from the storm , a sign that dry air was entering the system . During the afternoon hours , the structure of <unk> began to rapidly deteriorate as strong wind shear and dry air took their toll . By the late night , the center was almost completely exposed and only a band of convection persisted near the center . 
 Despite continuing effects from the strong wind shear , a large , deep burst of convection formed in the northern <unk> of <unk> . The center was found to have shifted towards the new convection leading to an increase in intensity . The forecast showed a slight decrease in wind shear as <unk> continued westward and no change in intensity over the 5 @-@ day forecast was predicted . However , the convection decreased once more and the low became completely exposed by the late morning hours and <unk> weakened again . By the afternoon , the center of <unk> was only a <unk> of clouds , devoid of convection . During the overnight hours on September 4 into the morning of September 5 , convection associated with <unk> began to <unk> somewhat , mostly to the north of the circulation , due to the strong <unk> wind shear . By mid @-@ morning , <unk> re @-@ intensified slightly due to the redevelopment of some convection . However , the redevelopment was short lived and wind shear again took its toll on <unk> by late morning . The convection around the system became <unk> from the center and <unk> weakened slightly . 
 The weakening trend continued through the afternoon as the storm was being affected by strong <unk> shear . <unk> became almost fully devoid of any convection by mid @-@ afternoon and the storm weakened to 40 mph ( 65 km / h ) , barely holding on to tropical storm status . <unk> regained a small amount of convection in the late night hours , but not enough to still be classified a tropical storm . Due to the lack of convection , <unk> was downgraded to a Tropical Depression at <unk> ( <unk> ) with winds of 35 mph ( 55 km / h ) . Since there was no convection around the system , it would have normally been classified a remnant low but , due to the possibility of the storm <unk> over the next several days , it was considered a tropical depression . The next morning , <unk> was downgraded to a remnant low as strong wind shear and dry air caused the demise of the storm . No redevelopment was expected with <unk> as it began to move over colder waters and remain under strong wind shear until it dissipated . 
 However , the remnant low associated with <unk> began to show signs of redevelopment during the afternoon on September 7 . <unk> around the system increased significantly and the low was no longer exposed . On September 8 , wind shear took over the system again . <unk> around the remnant low was torn away and the low was exposed once more . The National Hurricane Center did not state the chance of regeneration once the low became exposed . Finally , on September 9 , wind shear and dry air led to the remnants of <unk> deteriorating into an open wave . However , on September 10 , the remnants of <unk> redeveloped and global models picked up on the reformed system . Once more , the chance of regeneration was possible as the remnants of <unk> headed towards the Bahamas . However , on September 14 , dry air and wind shear caused the remnants to dissipate entirely . 
 
 = = Impact = = 
 
 As <unk> passed to the south of the Cape Verde islands on September 2 , outer rain bands produced minor rainfall , totaling around 0 @.@ 55 inches ( 14 mm ) . There were no reports of damage or flooding from the rain and overall effects were minor . 
 Several days after the low dissipated , the remnant moisture from <unk> brought showers and thunderstorms to St. Croix where up to 1 in ( 25 @.@ 4 mm ) of rain fell . The heavy rains led to minor street flooding and some urban flooding . No known damage was caused by the flood . 
  = Calvin <unk> = 
 
 Calvin <unk> ( born November 2 , 1984 ) is a Canadian football running back for the Edmonton <unk> of the Canadian Football League ( <unk> ) . He played as a <unk> until 2014 , when he became the starting fullback for the <unk> . <unk> is known for being able to fill many roles at his position , with <unk> <unk> Chris Schultz noting in 2010 that he is a "" multi @-@ purpose running back who catches the ball extremely well , blocks well and runs well "" . He is a champion of the <unk> Grey Cup . 
 Prior to being drafted by the Edmonton <unk> in the fourth round of the 2007 <unk> Draft , <unk> played high school football for the St. Thomas More Knights , where he broke multiple school records . He later played college football for the Western Washington Vikings . With the Vikings , <unk> developed into a dual threat , being used heavily as a rusher and receiver . <unk> has spent his entire professional career with the Edmonton <unk> , making him the most veteran player on the team as of the 2015 season . 
 
 = = High school career = = 
 
 <unk> played high school football for the St. Thomas More Knights in <unk> , British Columbia beginning in 1998 , playing as both a running back and middle linebacker on the Grade 8 team . He quickly became a key player on the team , including rushing for 185 yards and four touchdowns while adding 10 defensive tackles in the Grade 8 provincial semi @-@ finals on November 24 against the <unk> Royals . For that performance , he was named "" Star of the Week "" by The Vancouver Sun . The Knights went on to defeat the Vancouver College Fighting Irish 48 – 0 in the Grade 8 <unk> championship game , where <unk> scored another three touchdowns . 
 The following year , <unk> joined the senior team at St. Thomas More , <unk> the junior team entirely . He continued in his role as a running back while switching to the secondary <unk> . Despite being a rare Grade 9 player on the senior team , he was one of the Knights ' two leading rushers that year . <unk> scored a touchdown in the AAA <unk> championship game , helping the Knights win their first senior title as they beat the <unk> T @-@ Wolves 29 – 6 . While <unk> Steele , the Knights ' coach , has a policy of not recording player statistics , it is estimated that <unk> finished 1999 with over 1 @,@ 000 rushing yards and 15 touchdowns . 
 In 2000 , <unk> played a significant role in the Knights ' rushing and return game . He rushed for 150 yards and three touchdowns against the <unk> Central <unk> in the annual <unk> Bowl , adding two punt returns for touchdowns . By the end of October , the Knights had compiled a perfect 6 – 0 record , and defensive coordinator and former <unk> defensive back Lou <unk> called <unk> "" the best player for his age and talent that we 've ever had at our school "" . He recorded 238 yards and four consecutive touchdowns with only 23 carries in the Knights ' 46 – 12 victory over the W. J. <unk> Hawks in the AAA championship game . For his role in earning the Knights their second AAA title , <unk> was named the game 's MVP , becoming the youngest player to earn this award . 
 <unk> remained a presence on the field for the Knights in 2001 , expanding his role by playing some snaps as a fullback . He rushed for 160 yards in that year 's <unk> Bowl , including three touchdowns as a <unk> and one as a fullback . <unk> had another notable performance in a 67 – 7 <unk> against the South Delta Sun Devils , running for 200 yards and three touchdowns . He also showed himself to be a capable receiver , including making five catches for 66 yards in a match against the Holy Cross Crusaders . <unk> also finished that game with 110 yards and two touchdowns on only three carries . Despite giving up only 19 points during the entire regular season , the Knights lost 32 – 26 in the quarter @-@ finals of the playoffs against the Centennial <unk> after <unk> was stopped one yard away from the <unk> on the last play of the game . 
 As a senior , <unk> recorded 2 @,@ 400 yards and scored 33 touchdowns in just nine games , averaging 266 @.@ 7 yards and over three touchdowns per game . He ran for 380 yards and three touchdowns in a 21 – 20 loss against the W. J. <unk> Hawks in the quarter @-@ finals of the AAA playoffs . He was named the 2002 Provincial Player of the Year for his performance and finished his four @-@ year AAA career at St. Thomas More with 84 touchdowns , breaking a school record . 
 
 = = = Other sports = = = 
 
 <unk> played basketball and baseball during high school as well . He <unk> in basketball with the Knights . At the 2001 Big League World Series , <unk> played for Team Canada as a shortstop and center <unk> , helping the team to third place in the international competition . Despite being skilled as a baseball player , <unk> chose football over professional baseball due to the <unk> of the former sport . 
 
 = = College career = = 
 
 
 = = = Boise State = = = 
 
 <unk> originally committed to Boise State University and played for the Broncos . In 2003 , he was given redshirt status and did not play . <unk> played a limited role the following year , but capitalized on the opportunities he was given , rushing for 104 yards on only 10 carries . His only touchdown came on a 7 @-@ yard carry in the October 24 game against the Fresno State Bulldogs . The Bulldogs finished with an 11 – 1 season in 2004 . 
 
 = = = <unk> = = = 
 
 <unk> transferred to <unk> College and played for the Tigers in 2005 . He ended the season with 620 rushing yards , 500 receiving yards , and 14 touchdowns , as the Tigers earned a 10 – 1 record along their way to the Central Valley Conference championship title . <unk> also played baseball at <unk> . 
 
 = = = Western Washington = = = 
 
 After his single season at <unk> , <unk> transferred to Western Washington University and played football for the Vikings . <unk> was immediately a significant factor in the Vikings ' <unk> . In the season opener , he rushed for 139 yards and three touchdowns on 30 carries against the Humboldt State <unk> . He also played a large role in the passing game early in the season , making eight receptions for 126 yards through the first two games . After starting the first seven games for the Vikings , <unk> broke his foot in a game against the South Dakota <unk> . At the time of his injury , he led the Vikings in rushing and receiving yards . He finished the season with <unk> rushing yards and five touchdowns on 130 carries , as well as 30 receptions for <unk> yards . <unk> was also named a second @-@ team all @-@ star of the North Central Conference . 
 
 = = Professional career = = 
 
 
 = = = Edmonton <unk> = = = 
 
 Following his only season at Western Washington , <unk> declared himself eligible for the 2007 <unk> Draft . He was selected in the fourth round of the draft by the Edmonton <unk> with the 27th overall pick . He was re @-@ signed on December 19 , 2008 to a multiple @-@ year contract , and again following the 2011 season . 
 
 = = = = 2007 season = = = = 
 
 <unk> made the active roster and played in all 18 regular season games his rookie season . He was used in the passing game and on the special teams , finishing with seven receptions for 99 yards and a touchdown as well as five special @-@ teams tackles . <unk> made his <unk> debut on June 28 in the season opener against the Winnipeg Blue <unk> . He received his first carry and reception in a Week 10 game against the Calgary <unk> , where he was given two carries for one yard and caught one reception for 10 yards . 
 
 = = = = 2008 season = = = = 
 
 <unk> played a larger role in his second season with the <unk> , especially as a receiver . On September 1 in a match against the <unk> , starting running back A. J. Harris was injured , and <unk> rushed for 73 yards on 12 carries as a backup . <unk> in again for the injured Harris on September 13 against the Montreal <unk> , <unk> was named Canadian Player of the Week for the first time with 72 yards on only 9 carries and a touchdown . <unk> his first start of his career on October 4 , <unk> rushed for 88 yards on 19 carries with a touchdown along with eight catches for 80 yards , earning him another Canadian Player of the Week award . He played in all 18 games and started three times in his second year , finishing with 490 yards and four touchdowns on 88 carries . He had 70 catches , the second @-@ highest amount among running backs . He also continued his role on the special teams , ending the season with 11 special @-@ teams tackles . During the season , head coach Danny <unk> referred to <unk> as "" the best fourth @-@ round pick he 'd ever been associated with "" . 
 
 = = = = 2009 season = = = = 
 
 <unk> split time with <unk> <unk> in 2009 . He rushed for two touchdowns in Week 4 , being named the Canadian Player of the Week for the third time . He was also named the Canadian Player of the Month in July after continuing to play a large role in the rushing game . In August , <unk> injured his hamstring in a game against the <unk> and missed several games . He briefly returned in mid @-@ September before being sidelined again with recurring hamstring issues until late October . In Week 19 , <unk> ran for 81 yards and a touchdown off of 10 carries , helping the <unk> defeat the BC Lions in a 45 – 13 <unk> . Despite having his season <unk> by injuries , <unk> finished 2009 with 348 rushing yards and five touchdowns on 67 attempts , as well as seven special @-@ teams tackles . He saw a significantly smaller role as a receiver , catching 20 passes for only 124 yards . 
 
 = = = = 2010 season = = = = 
 
 <unk> remained in a multi @-@ purpose role in 2010 and was utilized more frequently in the passing game compared to the previous season . He made a reception for a first down following a fake punt in Week 6 . He missed two games later in the season due to a hand injury . In a September 26 game against the Toronto <unk> , <unk> rushed for 84 yards on 10 carries and two fourth @-@ quarter touchdowns , including a 46 @-@ yard <unk> . <unk> was utilized about equally on the ground and in the air , ending his season with <unk> rushing yards on 62 carries and <unk> receiving yards on 36 catches as well as five total touchdowns . He continued to play on the special teams where he made eight tackles . He started in six of the 15 games he played , and the <unk> nominated him for Most Outstanding Canadian . 
 
 = = = = 2011 season = = = = 
 
 In 2011 , the <unk> utilized a committee of running backs , with <unk> , Daniel Porter , and Jerome <unk> all receiving significant playing time . <unk> was used mostly in short @-@ yardage situations on the ground , while also being active as a receiver and on special teams . He played in 18 games , made eight starts , and finished with <unk> yards on 52 carries with no touchdowns . He also caught 22 passes for 150 yards and a touchdown . <unk> played in both of the <unk> ' playoff games . In the West <unk> @-@ Finals against the <unk> , he rushed for a goal @-@ line touchdown , in addition to making three receptions and two special @-@ teams tackles . <unk> played a more limited role in the West Finals against the BC Lions , where he was given only one carry for six yards , made one tackle on special teams , and caught two passes for a total of four yards . 
 
 = = = = 2012 season = = = = 
 
 <unk> played a limited role in 2012 , both due to injuries and competition from other backs , including Cory Boyd , Hugh Charles , and Jerome <unk> . <unk> missed six games due to a high ankle sprain suffered during the Labour Day <unk> . Playing in the other 12 games but starting in none , <unk> rushed only 12 times and made five receptions with a lone rushing touchdown . He added four special @-@ teams tackles . 
 
 = = = = 2013 season = = = = 
 
 <unk> played a role as a receiver and special teams player in 2013 , but was almost entirely absent from the rushing game . He rushed for 48 yards on 9 carries , but caught 20 passes for 186 yards and two touchdowns . He continued to play on the special teams , and recorded nine special @-@ teams tackles . 
 
 = = = = 2014 season = = = = 
 
 <unk> was shifted from playing mostly as a backup <unk> to the fullback position , where he started all 18 regular season games . In his new position , <unk> was primarily used for blocking and remained involved on special teams and as a receiver . He continued his extremely limited role as a rusher , finishing the season with just eight carries . <unk> had 16 catches for 123 yards and two touchdowns , as well as a career @-@ high 12 special @-@ teams tackles . 
 
 = = = = 2015 season = = = = 
 
 <unk> was again used as a fullback and special teams player in 2015 . After sustaining an unspecified injury in week 12 , he missed several games and was placed on the six @-@ game injured list . <unk> finished the season with only one carry and nine catches , his lowest total number of touches in any <unk> season . With 12 starts , <unk> finished with one carry for 20 yards , nine receptions for 68 yards , as well as five special teams tackles and one kick return for 17 yards . <unk> played in the West Final and had one special teams tackle . He became a Grey Cup champion for the first time after rushing for three yards on a fake punt in the championship game versus the <unk> . 
 
 = = = Season statistics = = = 
 
 
 = = Personal life = = 
 
 <unk> grew up in <unk> , Oklahoma . After his mother , Jackie Conway , was unable to financially support him , <unk> moved to Canada and lived with his father , Orlando <unk> . 
 His mother was a college <unk> player for the <unk> Lady Norse , while his father was a college basketball player for the <unk> Warriors and the Southeastern Oklahoma State Savage Storm . <unk> 's siblings were also college athletes ; <unk> played <unk> for the McPherson Bulldogs while Jordan was a quarterback with the <unk> Tigers . 
  Following the advance to the <unk> , there was a pause of about a week as the Australians had to wait for roads to be improved and supplies to be brought up , before attempting to cross the <unk> en <unk> . This allowed <unk> to re @-@ evaluate the situation and to issue new orders for the advance towards the <unk> and <unk> Rivers . As they waited for the advance to resume , the Australians carried out reconnaissance patrols deep into Japanese held territory and there were a couple of significant engagements during this time . As a part of these , the 24th Infantry Battalion sent a company across the <unk> and subsequently located a strong Japanese position on a feature that became known as <unk> 's Ridge , which , due to its location , commanded the main Australian axis of advance . 
 The main crossing was planned for 20 May , with the <unk> / 59th Infantry Battalion on the right tasked to cut the <unk> Road and the <unk> Track to the east of the river , while on the left the 57th / 60th Infantry Battalion would divert the attention of the Japanese off the 24th Infantry Battalion which would make the main frontal assault from the centre of the Australian line , crossing at the <unk> <unk> , advancing straight up the <unk> Road . Preliminary moves began before this , and on 15 May a platoon from the 24th Infantry Battalion along with two tanks attempted to carry out an attack on <unk> 's Ridge . After one of the tanks was held up and knocked out by a Japanese field gun , they were forced to withdraw . Meanwhile , the <unk> <unk> squadrons — now reinforced by No. 16 Squadron — began an eight @-@ day aerial campaign , attacking along the length of the <unk> and Commando Roads . During this period , the New Zealanders flew 381 sorties , while artillery and mortars fired "" thousands of rounds "" . 
 Two days later , on 17 May , the 57th / 60th Infantry Battalion began its diversionary move on the left flank , crossing the <unk> inland and advancing along the Commando Road with 32 <unk> and two batteries of artillery in support . Crossing 500 yd ( 460 m ) north of the <unk> , the centre company carried out an attack along the far bank of the river without its armoured support which had been unable to negotiate the crossing . Nevertheless , shortly before noon they had secured the crossing and began to fan out , carrying out further flanking moves before establishing a firm base to receive supplies and from where it began patrolling operations on 20 May . 
 In the centre , the main attack along the <unk> Road began at 08 : 30 on 20 May after 20 minutes of <unk> by New Zealand <unk> had prepared the ground . <unk> under a <unk> barrage , and with mortar and machine gun support , the 24th Infantry Battalion moved forward with three companies up front and one held back in reserve , along with two troops of Matilda tanks . <unk> the forward companies reached their objectives , but one of the companies was halted just short of their objective and was forced to dig @-@ in overnight after coming under heavy small arms and artillery fire and losing four killed and five wounded . The attack was resumed the following day , and the Australians were able to advance to the <unk> <unk> ; however , they were prevented from moving any further as the Japanese were still concentrated in large numbers further to the west where an Australian patrol encountered 70 Japanese and were forced to go to ground . Finally , a company from the 24th Infantry Battalion was able to move on to the high ground on <unk> 's Ridge , which they found to be heavily mined and booby trapped . Engineers and assault pioneers were called up to clear the feature . 
 On the right flank , the <unk> / 59th Infantry Battalion carried out a wide flanking move along a track that had been carved out of the west bank of the <unk> by <unk> . Beginning their move two days earlier , a number of patrols had had contacts with the Japanese . Meanwhile , using <unk> to drag the tanks through the mud , the Australian armour had crossed the river also and by 16 : 00 on 20 May the battalion had managed to establish itself in an assembly area to the east of the river , <unk> to the Japanese . The following day , the battalion left the line of departure and began to advance on its primary objective , which it reached in the early afternoon despite being held up while the tanks attempted to affect a creek crossing , and further delayed by stiff resistance . Later , after one of the battalion 's patrols came under heavy fire , the tanks moved up and attacked a Japanese gun position which the defenders quickly abandoned , leaving behind a 70 mm gun and a large amount of ammunition . 
 By 22 May , although there were still a number of Japanese in the area which continued to harass and ambush their line of communications , most of the Australian objectives had been secured and <unk> up operations began . The last remaining defensive location before the <unk> was <unk> 's Ridge , where the Japanese were sheltering in tunnels . A heavy aerial and artillery bombardment devastated the position and forced them to abandon the ridge . It was subsequently occupied by a company of Australian infantry . Within a short period of time the <unk> Road was subsequently opened , providing the Australians with the means with which to bring up supplies for the next stage of the campaign , being the advance to the <unk> , <unk> , and <unk> Rivers . The final phase of the battle cost the Japanese 106 killed , while the Australians lost 13 killed and 64 wounded . 
 
 = = Aftermath = = 
 
 During the course of the fighting around the <unk> , the Australians lost 38 men killed and 159 wounded , while the Japanese lost at least 275 men killed . Following the battle , the Australians continued their advance towards <unk> at the southern end of the island . Throughout the remainder of the month and into June , the 15th Brigade advanced along the <unk> Road , crossing the <unk> on 10 June . Beyond the river , the Japanese resolved to hold the food growing areas in order to protect their precarious food supply , and they consequently occupied a series of deep <unk> . These were steadily reduced with airstrikes and artillery , and the 15th Brigade subsequently crossed <unk> River before being relieved by Brigadier Noel Simpson 's 29th Brigade in early July . 
 As the 29th Brigade advanced toward the <unk> River , torrential rain and flooding ultimately brought the advance to a halt . The height of the river rose 2 metres ( 6 @.@ 6 ft ) . The <unk> Road was reduced , in the words of Gavin Long , "" to a sea of mud "" and many of the bridges upon which the Australian supply system was dependent were washed away . This rendered large @-@ scale offensive infantry operations impossible and as the situation worsened for a period of time the Australians even ceased patrolling operations across the <unk> ; meanwhile , the Japanese continued to harass the Australians , probing their positions and setting mines and traps , targeting the Australian line of communications . On 9 July , the 15th Infantry Battalion fought off a series of attacks around <unk> , at the junction of the <unk> River and the <unk> Road , which included a heavy Japanese artillery bombardment . Australian patrols were resumed in late July and continued into August . These attacks proved very costly , particularly amongst the Australian engineers that were tasked with rebuilding the bridges and roads that had been destroyed in the flooding . 
 Fighting in the northern sector continued during this time also , and although preparations in the south for the final advance towards <unk> continued into August , combat operations on the island ceased as the war came to an end before these were completed . As a result , the final Australian operations on <unk> took place on the <unk> front in the northern sector , where the Australians had been conducting a holding action since the failed landing at <unk> <unk> had forced them to abandon plans for an advance into the <unk> Peninsula . By mid @-@ August , however , following the dropping of two atomic bombs on <unk> and <unk> and Japan 's subsequent unconditional surrender , a cease fire was ordered on the island and although there were minor clashes following this , it spelt an end to major combat operations . 
 Following the end of the war , the Australian Army awarded three battle honours for the fighting around the <unk> River . The 2 / 4th Armoured Regiment , and the 9th , 24th , 57th / 60th and <unk> / 59th Infantry Battalions received the battle honour "" <unk> River "" . A second battle honour — "" <unk> 's Ridge – <unk> Ford "" — was also awarded to the 2 / 4th Armoured Regiment , and the 24th and <unk> / 59th Infantry Battalions for the second stage of the fighting , while the 57th / 60th Infantry Battalion received the separate battle honour of "" Commando Road "" for this period . 
 as that was 1997sa's heritage,
 """
    title_regex = re.compile("((= )+.*(= )+)")
    bp = BasicProcessor()

    print(bp.process(
        [line for line in text.split("\n")],
        custom_rules=[
            lambda x: re.sub(title_regex, "", x)
        ]
    ))
