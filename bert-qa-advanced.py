import torch
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

executed_answer_question_at_least_once = False

def answer_question(question, answer_text):                             # START of answer_question, which takes 2 strings: the question intended to be answered, and the data that the question's based on.
	global executed_answer_question_at_least_once
	input_ids = tokenizer.encode(question, answer_text)
	
	print('Query has {:,} tokens.\n'.format(len(input_ids)))            # Prints the number of tokens generated from the input data, when parsed by the tokenizer.
	
	sep_index = input_ids.index(tokenizer.sep_token_id)
	
	num_seg_a = sep_index + 1
	
	num_seg_b = len(input_ids) - num_seg_a
	
	segment_ids = [0]*num_seg_a + [1]*num_seg_b
	
	assert len(segment_ids) == len(input_ids)
	
	start_scores, end_scores = model(torch.tensor([input_ids]),                 # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
	
	answer_start = torch.argmax(start_scores)
	answer_end = torch.argmax(end_scores)
	
	tokens = tokenizer.convert_ids_to_tokens(input_ids)
	
	answer = tokens[answer_start]
	
	for i in range(answer_start + 1, answer_end + 1):
	
		if tokens[i][0:2] == '##':
			answer += tokens[i][2:]
		else:
			answer += ' ' + tokens[i]
	
	print("\""+question+"\", ")                                         # Prints the function's csv generation to the console, to function as a preview of which chunk of data is currently being parsed by answer_question
	print("\""+answer+"\"\n")
                                                                        # Note: CSV files separare columns with commas (outside of strings, except in Microsoft Excel -_-"), and rows with newline characters
	
	
	f = open("QA-Results.csv", "a+")                                    # Open CSV document
	f.write("\""+question+"\", ")                                       # Write: "Question", "Bert_Answer", "NULL"\n
	f.write("\""+answer+"\", \"NULL\"\n")                               # By leaving the third column as "NULL", we create an easy search-index to be overwritten by the transformer-xl results, 
                                                                        # generated later.
	f.close()                                                           # Close CSV document
	
	if executed_answer_question_at_least_once==True:                    # IF (the function has been executed at least once before):
		f = open("QC-Results.json", "a+")                               # Open JSON document
		f.write(",{\n")                                                 # Starts the new JSON Object array
		f.write("\t\t\t\"Question\": \""+question+"\",\n")              # Creates the first value inside of the first element in the new Array of JSON Objects (inside of the JSON file).
		f.write("\t\t\t\"BERT Answer\": \""+answer+"\",\n")             # Second value
		f.write("\t\t\t\"Transformer-XL Answer:\": \"NULL\"\n")         # Third value
		f.write("\t\t}")                                                # Finishes the JSON Object array
		f.close()                                                       # Close JSON document
	else:                                                               # ELSE (the function is running for the first time):
		f = open("QC-Results.json", "a+")                               # Open JSON document
		f.write("\t\t{\n")                                              # Starts the new JSON Object array
		f.write("\t\t\t\"Question\": \""+question+"\",\n")              # Creates the first value inside of the first element in the new Array of JSON Objects (inside of the JSON file).
		f.write("\t\t\t\"BERT Answer\": \""+answer+"\",\n")             # Second value
		f.write("\t\t\t\"Transformer-XL Answer:\": \"NULL\"\n")         # Third value
		f.write("\t\t}")                                                # Finishes the JSON Object array
		f.close()                                                       # Close JSON document
		
	if executed_answer_question_at_least_once != True:                  # A temp hack for more-easily generating the JSON File
		executed_answer_question_at_least_once = True                   # Last line of the short if statement
                                                                        # END of answer_question



print("\"Question\", \"Bert Answer\", \"Transformer-XL Answer\"\n")     # Previews the first line of the CSV to the one executing bert-qa-advanced.py
f = open("QA-Results.csv", "a+")                                        # Open CSV document
f.write("\"Question\", \"Bert Answer\", \"Transformer-XL Answer\"\n")   # Puts the titles of each category of data in the first line of the CSV (For column identification purposes)
f.close()                                                               # Close CSV document


f = open("QC-Results.json", "a+")                                       # Open JSON document
f.write("{\n\t\"QandA\": [\n")                                          # Write the header and data structure of the JSON file, prior to the data being parsed.
f.close()                                                               # Close JSON document

answer_question("Who was George Graham?", "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .")

answer_question("Where was George Graham born?", "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .")

answer_question("How large is the Cameron House estate?", "Cameron House, located on Loch Lomond near Balloch, Scotland, was first built in the mid-1700s, and later purchased by Sir James Smollett. The modern Baronial stone castle was built by William Spence in 1830 (rebuilt after a fire in 1865), with peaked gables and decorative turrets. The House is a Category B listed building. For three centuries, the land was part of the Smollett estate, now reduced to 44 hectares of wooded land that juts into the Loch. In 1985 Laird Patrick Telfer Smollett sold the House and land to De Vere Hotels.")

answer_question("When was the Cameron House Estate rebuilt?", "Cameron House, located on Loch Lomond near Balloch, Scotland, was first built in the mid-1700s, and later purchased by Sir James Smollett. The modern Baronial stone castle was built by William Spence in 1830 (rebuilt after a fire in 1865), with peaked gables and decorative turrets. The House is a Category B listed building. For three centuries, the land was part of the Smollett estate, now reduced to 44 hectares of wooded land that juts into the Loch. In 1985 Laird Patrick Telfer Smollett sold the House and land to De Vere Hotels.")

answer_question("Who owns Hawaiian Airlines?", "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.")

answer_question("What airline has never had a fatal accident?", "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.")

answer_question("Where is the main airport for Hawaiian Airlines?", "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.")

answer_question("Is Kahului airport the main hub for Hawaiian Airlines?", "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.")

answer_question("Where did Much Ado About Nothing premiere?", "Much Ado About Nothing is a 2012 black and white American romantic comedy film adapted for the screen, produced, and directed by Joss Whedon, from William Shakespeare's play of the same name. The film stars Amy Acker, Alexis Denisof, Nathan Fillion, Clark Gregg, Reed Diamond, Fran Kranz, Sean Maher, and Jillian Morgese. To create the film, director Whedon established the production studio Bellwether Pictures. The film premiered at the 2012 Toronto International Film Festival and had its North American theatrical release on June 21, 2013.")

answer_question("Did Quentin Tarantino direct Much Ado About Nothing?", "Much Ado About Nothing is a 2012 black and white American romantic comedy film adapted for the screen, produced, and directed by Joss Whedon, from William Shakespeare's play of the same name. The film stars Amy Acker, Alexis Denisof, Nathan Fillion, Clark Gregg, Reed Diamond, Fran Kranz, Sean Maher, and Jillian Morgese. To create the film, director Whedon established the production studio Bellwether Pictures. The film premiered at the 2012 Toronto International Film Festival and had its North American theatrical release on June 21, 2013.")

answer_question("Who is Ondria Harden?", "Ondria Hardin is an American fashion model. Throughout her career she accrued controversy for her young age. Hardin was discovered at a pageant in her home state of North Carolina. She began her career in Tokyo, Japan. After being noticed by casting director Ashley Brokaw she was cast in a Prada campaign photographed by Steven Meisel by age 13. Designer Marc Jacobs was criticized for using Hardin, then aged 15, in his F/W 2012 fashion show, which he defended. She received further controversy for appearing in a Chanel campaign at age 15, and a Numéro editorial entitled African Queen.")

answer_question("Was Ondria Harden in a Chanel Campaign?", "Ondria Hardin is an American fashion model. Throughout her career she accrued controversy for her young age. Hardin was discovered at a pageant in her home state of North Carolina. She began her career in Tokyo, Japan. After being noticed by casting director Ashley Brokaw she was cast in a Prada campaign photographed by Steven Meisel by age 13. Designer Marc Jacobs was criticized for using Hardin, then aged 15, in his F/W 2012 fashion show, which he defended. She received further controversy for appearing in a Chanel campaign at age 15, and a Numéro editorial entitled African Queen.")

answer_question("When was ELIZA written?","Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted \"blocks worlds\" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966. Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the \"patient\" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to \"My head hurts\" with \"Why do you say your head hurts?\".")

answer_question("What did ELIZA do?","Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted \"blocks worlds\" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966. Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the \"patient\" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to \"My head hurts\" with \"Why do you say your head hurts?\".")

answer_question("Who created the Game & Watch brand?","The Game & Watch brand is a series of handheld electronic games produced by Nintendo from 1980 to 1991. Created by game designer Gunpei Yokoi, each Game & Watch features a single game to be played on an LCD screen in addition to a clock, an alarm, or both. It was the earliest Nintendo video game product to gain major success.")

answer_question("What did each Game & Watch feature?","The Game & Watch brand is a series of handheld electronic games produced by Nintendo from 1980 to 1991. Created by game designer Gunpei Yokoi, each Game & Watch features a single game to be played on an LCD screen in addition to a clock, an alarm, or both. It was the earliest Nintendo video game product to gain major success.")

answer_question("What occurs in ferromagnetic materials?","Magnetism is a class of physical phenomena that are mediated by magnetic fields. Electric currents and the magnetic moments of elementary particles give rise to a magnetic field, which acts on other currents and magnetic moments. Magnetism is one aspect of the combined phenomenon of electromagnetism. The most familiar effects occur in ferromagnetic materials, which are strongly attracted by magnetic fields and can be magnetized to become permanent magnets, producing magnetic fields themselves. Demagnetizing a magnet is also possible. Only a few substances are ferromagnetic; the most common ones are iron, cobalt and nickel and their alloys.")

answer_question("Is aluminum ferromagnetic?", "Magnetism is a class of physical phenomena that are mediated by magnetic fields. Electric currents and the magnetic moments of elementary particles give rise to a magnetic field, which acts on other currents and magnetic moments. Magnetism is one aspect of the combined phenomenon of electromagnetism. The most familiar effects occur in ferromagnetic materials, which are strongly attracted by magnetic fields and can be magnetized to become permanent magnets, producing magnetic fields themselves. Demagnetizing a magnet is also possible. Only a few substances are ferromagnetic; the most common ones are iron, cobalt and nickel and their alloys.")

answer_question("What substances are ferromagnetic?","Magnetism is a class of physical phenomena that are mediated by magnetic fields. Electric currents and the magnetic moments of elementary particles give rise to a magnetic field, which acts on other currents and magnetic moments. Magnetism is one aspect of the combined phenomenon of electromagnetism. The most familiar effects occur in ferromagnetic materials, which are strongly attracted by magnetic fields and can be magnetized to become permanent magnets, producing magnetic fields themselves. Demagnetizing a magnet is also possible. Only a few substances are ferromagnetic; the most common ones are iron, cobalt and nickel and their alloys.")

answer_question("How did Wendy's enter the Asian market?","Wendy's founded the fried chicken chain Sisters Chicken in 1978 and sold it to its largest franchiser in 1987. In 1979, the first European Wendy's opened in Munich. The same year, Wendy's became the first fast-food chain to introduce the salad bar. Wendy's entered the Asian market by opening its first restaurants in Japan in 1980, in Hong Kong in 1982, and in the Philippines and Singapore in 1983. In 1984, Wendy's opened its first restaurant in South Korea.")

answer_question("When did Wendy's start a salad bar?", "Wendy's founded the fried chicken chain Sisters Chicken in 1978 and sold it to its largest franchiser in 1987. In 1979, the first European Wendy's opened in Munich. The same year, Wendy's became the first fast-food chain to introduce the salad bar. Wendy's entered the Asian market by opening its first restaurants in Japan in 1980, in Hong Kong in 1982, and in the Philippines and Singapore in 1983. In 1984, Wendy's opened its first restaurant in South Korea.")

answer_question("Was the first European Wendy's in Paris?", "Wendy's founded the fried chicken chain Sisters Chicken in 1978 and sold it to its largest franchiser in 1987. In 1979, the first European Wendy's opened in Munich. The same year, Wendy's became the first fast-food chain to introduce the salad bar. Wendy's entered the Asian market by opening its first restaurants in Japan in 1980, in Hong Kong in 1982, and in the Philippines and Singapore in 1983. In 1984, Wendy's opened its first restaurant in South Korea.")

answer_question("What happened in 1999?","The Market Theater Gum Wall is a brick wall covered in used chewing gum located in an alleyway in Post Alley under Pike Place Market in Downtown Seattle. Much like Bubblegum Alley in San Luis Obispo, California, the Market Theater Gum Wall is a local landmark. Parts of the wall are covered several inches thick, 15 feet (4.6 m) high along a 50-foot-long (15 m) section. The wall is by the box office for the Market Theater. The tradition began around 1993 when patrons of Unexpected Productions' Seattle Theatresports stuck gum to the wall and placed coins in the gum blobs. Theater workers scraped the gum away twice, but eventually gave up after market officials deemed the gum wall a tourist attraction around 1999. Some people created small works of art out of gum. It was named one of the top 5 germiest tourist attractions in 2009, second to the Blarney Stone. It is the location of the start of a ghost tour, and also a popular site with wedding photographers. The state governor, Jay Inslee, said it is his \"favorite thing about Seattle you can't find anywhere else\".")

answer_question("What did the state governor say?","The Market Theater Gum Wall is a brick wall covered in used chewing gum located in an alleyway in Post Alley under Pike Place Market in Downtown Seattle. Much like Bubblegum Alley in San Luis Obispo, California, the Market Theater Gum Wall is a local landmark. Parts of the wall are covered several inches thick, 15 feet (4.6 m) high along a 50-foot-long (15 m) section. The wall is by the box office for the Market Theater. The tradition began around 1993 when patrons of Unexpected Productions' Seattle Theatresports stuck gum to the wall and placed coins in the gum blobs. Theater workers scraped the gum away twice, but eventually gave up after market officials deemed the gum wall a tourist attraction around 1999. Some people created small works of art out of gum. It was named one of the top 5 germiest tourist attractions in 2009, second to the Blarney Stone. It is the location of the start of a ghost tour, and also a popular site with wedding photographers. The state governor, Jay Inslee, said it is his \"favorite thing about Seattle you can't find anywhere else\".")

answer_question("Does Maya Dmitrievna Koveshnikova have paintings in Japan?", "Maya Dmitrievna Koveshnikova was a Russian painter, most known for her landscapes. In 1986, she was recognized as an Honored Artist of the Russian Soviet Federative Socialist Republic. She has paintings in galleries and museums throughout Russia and in both the Russo-Japanese House of Friendship in Sapporo, Japan and the Rijksmuseum of Amsterdam, as well as other international locations.")

answer_question("What is Maya Dmitrievna Koveshnikova known for?", "Maya Dmitrievna Koveshnikova was a Russian painter, most known for her landscapes. In 1986, she was recognized as an Honored Artist of the Russian Soviet Federative Socialist Republic. She has paintings in galleries and museums throughout Russia and in both the Russo-Japanese House of Friendship in Sapporo, Japan and the Rijksmuseum of Amsterdam, as well as other international locations.")

answer_question("Where did Anthony Wayne Washington go to college?", "Anthony Wayne Washington (born February 4, 1958 in San Francisco, California) is a former professional American football cornerback for the Washington Redskins and Pittsburgh Steelers of the National Football League (NFL). He played college football at Fresno State University and was drafted in the second round of the 1981 NFL Draft. Washington started for the Redskins in Super Bowl XVIII.")

answer_question("What Super Bowl was Anthony Wayne Washington in?", "Anthony Wayne Washington (born February 4, 1958 in San Francisco, California) is a former professional American football cornerback for the Washington Redskins and Pittsburgh Steelers of the National Football League (NFL). He played college football at Fresno State University and was drafted in the second round of the 1981 NFL Draft. Washington started for the Redskins in Super Bowl XVIII.")

answer_question("What period are the Shulba Sutras from?", "The Shulba Sutras or Śulbasūtras (Sanskrit śulba: 'string, cord, rope') are sutra texts belonging to the Śrauta ritual and containing geometry related to fire-altar construction. The Shulba Sutras are part of the larger corpus of texts called the Shrauta Sutras, considered to be appendices to the Vedas. They are the only sources of knowledge of Indian mathematics from the Vedic period. Unique fire-altar shapes were associated with unique gifts from the Gods.")

answer_question("What are the Shulba Sutras?", "The Shulba Sutras or Śulbasūtras (Sanskrit śulba: 'string, cord, rope') are sutra texts belonging to the Śrauta ritual and containing geometry related to fire-altar construction. The Shulba Sutras are part of the larger corpus of texts called the Shrauta Sutras, considered to be appendices to the Vedas. They are the only sources of knowledge of Indian mathematics from the Vedic period. Unique fire-altar shapes were associated with unique gifts from the Gods.")

answer_question("What does Sidewalk Labs do?", "Sidewalk Labs is Alphabet Inc.'s urban innovation organization. Its goal is to improve urban infrastructure through technological solutions, and tackle issues such as cost of living, efficient transportation and energy usage. It is headed by Daniel L. Doctoroff, former deputy mayor of New York City for economic development and former chief executive of Bloomberg L.P. Other members include Craig Nevill-Manning, co-founder of Google's New York office and inventor of Froogle.")

answer_question("What does Daniel Doctoroff head?", "Sidewalk Labs is Alphabet Inc.'s urban innovation organization. Its goal is to improve urban infrastructure through technological solutions, and tackle issues such as cost of living, efficient transportation and energy usage. It is headed by Daniel L. Doctoroff, former deputy mayor of New York City for economic development and former chief executive of Bloomberg L.P. Other members include Craig Nevill-Manning, co-founder of Google's New York office and inventor of Froogle.")

answer_question("What replaces the aortic valve?", "Viking Olov Björk was a Swedish cardiac surgeon. In 1968, he collaborated with American engineer Donald Shiley to develop the Björk–Shiley valve, a mechanical prosthetic heart valve. It was the first 'tilting disc valve', used to replace the aortic or mitral valve. Many modifications followed, including the convexo-concave valve. The convexo-concave valve had defects in form of strut fractures. Therefore the monostrut valve was introduced to prevent outflow strut fractures.")

answer_question("Who do Donald Shiley collaborate with?", "Viking Olov Björk was a Swedish cardiac surgeon. In 1968, he collaborated with American engineer Donald Shiley to develop the Björk–Shiley valve, a mechanical prosthetic heart valve. It was the first 'tilting disc valve', used to replace the aortic or mitral valve. Many modifications followed, including the convexo-concave valve. The convexo-concave valve had defects in form of strut fractures. Therefore the monostrut valve was introduced to prevent outflow strut fractures.")

answer_question("What is Sha Na Na?", "Henry Gross (born April 1, 1951) is an American singer-songwriter best known for his association with the group Sha Na Na and for his hit song, 'Shannon'. At age 18, while a student at Brooklyn College, Gross became a founding member of 1950's Rock & Roll revival group, Sha Na Na, playing guitar and wearing the greaser clothes he wore while a student at Midwood High School. He produced a single, 'Shannon', a song written about the death of former Beach Boys member Carl Wilson's dog, who was named Shannon.")

answer_question("What was Gross' hit song?", "Henry Gross (born April 1, 1951) is an American singer-songwriter best known for his association with the group Sha Na Na and for his hit song, 'Shannon'. At age 18, while a student at Brooklyn College, Gross became a founding member of 1950's Rock & Roll revival group, Sha Na Na, playing guitar and wearing the greaser clothes he wore while a student at Midwood High School. He produced a single, 'Shannon', a song written about the death of former Beach Boys member Carl Wilson's dog, who was named Shannon.")

answer_question("Was there a song about a dog?", "Henry Gross (born April 1, 1951) is an American singer-songwriter best known for his association with the group Sha Na Na and for his hit song, 'Shannon'. At age 18, while a student at Brooklyn College, Gross became a founding member of 1950's Rock & Roll revival group, Sha Na Na, playing guitar and wearing the greaser clothes he wore while a student at Midwood High School. He produced a single, 'Shannon', a song written about the death of former Beach Boys member Carl Wilson's dog, who was named Shannon.")

answer_question("What's the age range for the Juniores?", "The Juniores are the Parma football team composed of footballers between 17 and 20 years old, which is the most senior youth category according to Italian football's hierarchy. Each season, the squad is trialled for promotion to the first team before the beginning of the season. Players deemed ready for first team football are registered. The team has competed in the Italian Campionato Nazionale Primavera, which has been known as the Trofeo Giacinto Facchetti since 2006, but has never won the title.")

answer_question("Where have the Juniores competed?", "The Juniores are the Parma football team composed of footballers between 17 and 20 years old, which is the most senior youth category according to Italian football's hierarchy. Each season, the squad is trialled for promotion to the first team before the beginning of the season. Players deemed ready for first team football are registered. The team has competed in the Italian Campionato Nazionale Primavera, which has been known as the Trofeo Giacinto Facchetti since 2006, but has never won the title.")

answer_question("When did Michael Kmit die?", "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.")

answer_question("Who won the Blake Prize in 1952?", "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.")

answer_question("When did Michael Kmit win the Sulman Prize?", "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.")

answer_question("What did Kmit bring to Australia?", "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.")

answer_question("Is Lindsey Czarniak English?", "Lindsay Ann Czarniak (born November 7, 1977), is an American sports anchor and reporter. She currently works for Fox Sports as a studio host for NASCAR coverage and a sideline reporter for NFL games. After spending six years with WRC-TV, the NBC owned-and-operated station in Washington, D.C., Czarniak joined ESPN as a SportsCenter anchor in August 2011 and left ESPN in 2017. Czarniak served as a host and sportsdesk reporter for NBC Sports coverage of the 2008 Summer Olympics in Beijing, China.")

answer_question("When did Lindsey Czarniak work at ESPN?", "Lindsay Ann Czarniak (born November 7, 1977), is an American sports anchor and reporter. She currently works for Fox Sports as a studio host for NASCAR coverage and a sideline reporter for NFL games. After spending six years with WRC-TV, the NBC owned-and-operated station in Washington, D.C., Czarniak joined ESPN as a SportsCenter anchor in August 2011 and left ESPN in 2017. Czarniak served as a host and sportsdesk reporter for NBC Sports coverage of the 2008 Summer Olympics in Beijing, China.")

answer_question("Who voiced Doraemon?", "Doris Lo (Chinese: 盧素娟, Pinyin: Lú Sùjuān; 20 July 1952 – 22 July 2006) was a Hong Kong voice actor who was best known for voicing the character Nobita Nobi for the Hong Kong version of the anime along with Lam Pou-chuen who voices the character Doraemon. Lo died at the age of 54 from colorectal cancer at Shatin Hospital in Hong Kong.")

answer_question("What character did Doris Lo voice?", "Doris Lo (Chinese: 盧素娟, Pinyin: Lú Sùjuān; 20 July 1952 – 22 July 2006) was a Hong Kong voice actor who was best known for voicing the character Nobita Nobi for the Hong Kong version of the anime along with Lam Pou-chuen who voices the character Doraemon. Lo died at the age of 54 from colorectal cancer at Shatin Hospital in Hong Kong.")

answer_question("What Czech Republic newspaper has a centre-left political position?", "Czech Republic has four main daily newspapers: Lidové noviny (former dissident publication); Mladá fronta DNES (with a centre-right orientation); Právo (with a centre-left political position) and Blesk, all based in Prague. Both Lidové noviny and Mladá fronta DNES are a part of the MAFRA publishing group, owned by Andrej Babiš, the current Prime Minister of the Czech Republic. As of 2018, the MAFRA group is a part of a trust fund along with other Babiš's companies.")

answer_question("Where is Lidové noviny based?", "Czech Republic has four main daily newspapers: Lidové noviny (former dissident publication); Mladá fronta DNES (with a centre-right orientation); Právo (with a centre-left political position) and Blesk, all based in Prague. Both Lidové noviny and Mladá fronta DNES are a part of the MAFRA publishing group, owned by Andrej Babiš, the current Prime Minister of the Czech Republic. As of 2018, the MAFRA group is a part of a trust fund along with other Babiš's companies.")

f = open("QC-Results.json", "a+")                                       # Open JSON document (Again)
f.write("\n\t]\n}")                                                     # Writes the footer of the data structure(After the data-processing).
f.close()                                                               # Close JSON document