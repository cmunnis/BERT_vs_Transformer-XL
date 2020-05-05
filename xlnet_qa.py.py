from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

to_predict = []

# Create dummy data to use for training.
train_data = [
    {
        'context': "This is the first context",
        'qas': [
            {
                'id': "00001",
                'is_impossible': False,
                'question': "Which context is this?",
                'answers': [
                    {
                        'text': "the first",
                        'answer_start': 8
                    }
                ]
            }
        ]
    },
    {
        'context': "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales,and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
        'qas': [
            {
                'id': "00002",
                'is_impossible': False,
                'question': "What was the cost to society?",
                'answers': [
                    {
                        'text': "low cost",
                        'answer_start': 225
                    }
                ]
            },
            {
                'id': "00003",
                'is_impossible': False,
                'question': "What was the name of the 1937 treaty?",
                'answers': [
                    {
                        'text': "Bald Eagle Protection Act",
                        'answer_start': 167
                    }
                ]
            }
        ]
    }
]

# Save as a JSON file
os.makedirs('data', exist_ok=True)
with open('data/train.json', 'w') as f:
    json.dump(train_data, f)


# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('xlnet', 'xlnet-large-cased', use_cuda=False, args={'reprocess_input_data': True, 'overwrite_output_dir': True})

# Train the model with JSON file
model.train_model('data/train.json')

# The list can also be used directly
# model.train_model(train_data)

# Evaluate the model. (Being lazy and evaluating on the train data itself)
result, text = model.eval_model('data/train.json')

print(result)
print(text)

print('-------------------')

# Making predictions using the model.
to_predict.append([{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What is this context?', 'id': '0'}]}])

to_predict.append([{"context": "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .",'qas': [{'question': 'Who was George Graham?', 'id': '1'}]}])

to_predict.append([{"context": "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .",'qas': [{'question': 'Where was George Graham born?', 'id': '2'}]}])

to_predict.append([{"context": "Cameron House, located on Loch Lomond near Balloch, Scotland, was first built in the mid-1700s, and later purchased by Sir James Smollett. The modern Baronial stone castle was built by William Spence in 1830 (rebuilt after a fire in 1865), with peaked gables and decorative turrets. The House is a Category B listed building. For three centuries, the land was part of the Smollett estate, now reduced to 44 hectares of wooded land that juts into the Loch. In 1985 Laird Patrick Telfer Smollett sold the House and land to De Vere Hotels.",'qas': [{'question': "How large is the Cameron House estate?", 'id': '3'}]}])

to_predict.append([{"context": "Cameron House, located on Loch Lomond near Balloch, Scotland, was first built in the mid-1700s, and later purchased by Sir James Smollett. The modern Baronial stone castle was built by William Spence in 1830 (rebuilt after a fire in 1865), with peaked gables and decorative turrets. The House is a Category B listed building. For three centuries, the land was part of the Smollett estate, now reduced to 44 hectares of wooded land that juts into the Loch. In 1985 Laird Patrick Telfer Smollett sold the House and land to De Vere Hotels.",'qas': [{'question': "When was the Cameron House Estate rebuilt?", 'id': '4'}]}])

to_predict.append([{"context": "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.",'qas': [{'question': "Who owns Hawaiian Airlines?", 'id': '5'}]}])

to_predict.append([{"context": "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.",'qas': [{'question': "What airline has never had a fatal accident?", 'id': '6'}]}])

to_predict.append([{"context": "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.",'qas': [{'question': "Where is the main airport for Hawaiian Airlines?", 'id': '7'}]}])

to_predict.append([{"context": "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.",'qas': [{'question': "Is Kahului airport the main hub for Hawaiian Airlines?", 'id': '8'}]}])

to_predict.append([{"context": "Much Ado About Nothing is a 2012 black and white American romantic comedy film adapted for the screen, produced, and directed by Joss Whedon, from William Shakespeare's play of the same name. The film stars Amy Acker, Alexis Denisof, Nathan Fillion, Clark Gregg, Reed Diamond, Fran Kranz, Sean Maher, and Jillian Morgese. To create the film, director Whedon established the production studio Bellwether Pictures. The film premiered at the 2012 Toronto International Film Festival and had its North American theatrical release on June 21, 2013.",'qas': [{'question': "Where did Much Ado About Nothing premiere?", 'id': '9'}]}])

to_predict.append([{"context": "Much Ado About Nothing is a 2012 black and white American romantic comedy film adapted for the screen, produced, and directed by Joss Whedon, from William Shakespeare's play of the same name. The film stars Amy Acker, Alexis Denisof, Nathan Fillion, Clark Gregg, Reed Diamond, Fran Kranz, Sean Maher, and Jillian Morgese. To create the film, director Whedon established the production studio Bellwether Pictures. The film premiered at the 2012 Toronto International Film Festival and had its North American theatrical release on June 21, 2013.",'qas': [{'question': "Did Quentin Tarantino direct Much Ado About Nothing?", 'id': '10'}]}])

to_predict.append([{"context": "Ondria Hardin is an American fashion model. Throughout her career she accrued controversy for her young age. Hardin was discovered at a pageant in her home state of North Carolina. She began her career in Tokyo, Japan. After being noticed by casting director Ashley Brokaw she was cast in a Prada campaign photographed by Steven Meisel by age 13. Designer Marc Jacobs was criticized for using Hardin, then aged 15, in his F/W 2012 fashion show, which he defended. She received further controversy for appearing in a Chanel campaign at age 15, and a Numéro editorial entitled African Queen.",'qas': [{'question': "Who is Ondria Harden?", 'id': '11'}]}])

to_predict.append([{"context": "Ondria Hardin is an American fashion model. Throughout her career she accrued controversy for her young age. Hardin was discovered at a pageant in her home state of North Carolina. She began her career in Tokyo, Japan. After being noticed by casting director Ashley Brokaw she was cast in a Prada campaign photographed by Steven Meisel by age 13. Designer Marc Jacobs was criticized for using Hardin, then aged 15, in his F/W 2012 fashion show, which he defended. She received further controversy for appearing in a Chanel campaign at age 15, and a Numéro editorial entitled African Queen.",'qas': [{'question': "Was Ondria Harden in a Chanel Campaign?", 'id': '12'}]}])

to_predict.append([{"context": "Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted \"blocks worlds\" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966. Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the \"patient\" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to \"My head hurts\" with \"Why do you say your head hurts?\".",'qas': [{'question': "When was ELIZA written?", 'id': '13'}]}])

to_predict.append([{"context": "Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted \"blocks worlds\" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966. Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the \"patient\" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to \"My head hurts\" with \"Why do you say your head hurts?\".",'qas': [{'question': "What did ELIZA do?", 'id': '14'}]}])

to_predict.append([{"context": "The Game & Watch brand is a series of handheld electronic games produced by Nintendo from 1980 to 1991. Created by game designer Gunpei Yokoi, each Game & Watch features a single game to be played on an LCD screen in addition to a clock, an alarm, or both. It was the earliest Nintendo video game product to gain major success.",'qas': [{'question': "Who created the Game & Watch brand?", 'id': '15'}]}])

to_predict.append([{"context": "The Game & Watch brand is a series of handheld electronic games produced by Nintendo from 1980 to 1991. Created by game designer Gunpei Yokoi, each Game & Watch features a single game to be played on an LCD screen in addition to a clock, an alarm, or both. It was the earliest Nintendo video game product to gain major success.",'qas': [{'question': "What did each Game & Watch feature?", 'id': '16'}]}])

to_predict.append([{"context": "Magnetism is a class of physical phenomena that are mediated by magnetic fields. Electric currents and the magnetic moments of elementary particles give rise to a magnetic field, which acts on other currents and magnetic moments. Magnetism is one aspect of the combined phenomenon of electromagnetism. The most familiar effects occur in ferromagnetic materials, which are strongly attracted by magnetic fields and can be magnetized to become permanent magnets, producing magnetic fields themselves. Demagnetizing a magnet is also possible. Only a few substances are ferromagnetic; the most common ones are iron, cobalt and nickel and their alloys.",'qas': [{'question': "Is aluminum ferromagnetic?", 'id': '17'}]}])

to_predict.append([{"context": "Magnetism is a class of physical phenomena that are mediated by magnetic fields. Electric currents and the magnetic moments of elementary particles give rise to a magnetic field, which acts on other currents and magnetic moments. Magnetism is one aspect of the combined phenomenon of electromagnetism. The most familiar effects occur in ferromagnetic materials, which are strongly attracted by magnetic fields and can be magnetized to become permanent magnets, producing magnetic fields themselves. Demagnetizing a magnet is also possible. Only a few substances are ferromagnetic; the most common ones are iron, cobalt and nickel and their alloys.",'qas': [{'question': "What substances are ferromagnetic?", 'id': '18'}]}])

to_predict.append([{"context": "Wendy's founded the fried chicken chain Sisters Chicken in 1978 and sold it to its largest franchiser in 1987. In 1979, the first European Wendy's opened in Munich. The same year, Wendy's became the first fast-food chain to introduce the salad bar. Wendy's entered the Asian market by opening its first restaurants in Japan in 1980, in Hong Kong in 1982, and in the Philippines and Singapore in 1983. In 1984, Wendy's opened its first restaurant in South Korea.",'qas': [{'question': "How did Wendy's enter the Asian market?", 'id': '19'}]}])

to_predict.append([{"context": "Wendy's founded the fried chicken chain Sisters Chicken in 1978 and sold it to its largest franchiser in 1987. In 1979, the first European Wendy's opened in Munich. The same year, Wendy's became the first fast-food chain to introduce the salad bar. Wendy's entered the Asian market by opening its first restaurants in Japan in 1980, in Hong Kong in 1982, and in the Philippines and Singapore in 1983. In 1984, Wendy's opened its first restaurant in South Korea.",'qas': [{'question': "When did Wendy's start a salad bar?", 'id': '20'}]}])

to_predict.append([{"context": "Wendy's founded the fried chicken chain Sisters Chicken in 1978 and sold it to its largest franchiser in 1987. In 1979, the first European Wendy's opened in Munich. The same year, Wendy's became the first fast-food chain to introduce the salad bar. Wendy's entered the Asian market by opening its first restaurants in Japan in 1980, in Hong Kong in 1982, and in the Philippines and Singapore in 1983. In 1984, Wendy's opened its first restaurant in South Korea.",'qas': [{'question': "Was the first European Wendy's in Paris?", 'id': '21'}]}])

to_predict.append([{"context": "The Market Theater Gum Wall is a brick wall covered in used chewing gum located in an alleyway in Post Alley under Pike Place Market in Downtown Seattle. Much like Bubblegum Alley in San Luis Obispo, California, the Market Theater Gum Wall is a local landmark. Parts of the wall are covered several inches thick, 15 feet (4.6 m) high along a 50-foot-long (15 m) section. The wall is by the box office for the Market Theater. The tradition began around 1993 when patrons of Unexpected Productions' Seattle Theatresports stuck gum to the wall and placed coins in the gum blobs. Theater workers scraped the gum away twice, but eventually gave up after market officials deemed the gum wall a tourist attraction around 1999. Some people created small works of art out of gum. It was named one of the top 5 germiest tourist attractions in 2009, second to the Blarney Stone. It is the location of the start of a ghost tour, and also a popular site with wedding photographers. The state governor, Jay Inslee, said it is his \"favorite thing about Seattle you can't find anywhere else\".",'qas': [{'question': "What happened in 1999?", 'id': '22'}]}])

to_predict.append([{"context": "The Market Theater Gum Wall is a brick wall covered in used chewing gum located in an alleyway in Post Alley under Pike Place Market in Downtown Seattle. Much like Bubblegum Alley in San Luis Obispo, California, the Market Theater Gum Wall is a local landmark. Parts of the wall are covered several inches thick, 15 feet (4.6 m) high along a 50-foot-long (15 m) section. The wall is by the box office for the Market Theater. The tradition began around 1993 when patrons of Unexpected Productions' Seattle Theatresports stuck gum to the wall and placed coins in the gum blobs. Theater workers scraped the gum away twice, but eventually gave up after market officials deemed the gum wall a tourist attraction around 1999. Some people created small works of art out of gum. It was named one of the top 5 germiest tourist attractions in 2009, second to the Blarney Stone. It is the location of the start of a ghost tour, and also a popular site with wedding photographers. The state governor, Jay Inslee, said it is his \"favorite thing about Seattle you can't find anywhere else\".",'qas': [{'question': "What did the state governor say?", 'id': '23'}]}])

to_predict.append([{"context": "Maya Dmitrievna Koveshnikova was a Russian painter, most known for her landscapes. In 1986, she was recognized as an Honored Artist of the Russian Soviet Federative Socialist Republic. She has paintings in galleries and museums throughout Russia and in both the Russo-Japanese House of Friendship in Sapporo, Japan and the Rijksmuseum of Amsterdam, as well as other international locations.",'qas': [{'question': "Does Maya Dmitrievna Koveshnikova have paintings in Japan?", 'id': '24'}]}])

to_predict.append([{"context": "Maya Dmitrievna Koveshnikova was a Russian painter, most known for her landscapes. In 1986, she was recognized as an Honored Artist of the Russian Soviet Federative Socialist Republic. She has paintings in galleries and museums throughout Russia and in both the Russo-Japanese House of Friendship in Sapporo, Japan and the Rijksmuseum of Amsterdam, as well as other international locations.",'qas': [{'question': "What is Maya Dmitrievna Koveshnikova known for?", 'id': '25'}]}])

to_predict.append([{"context": "Anthony Wayne Washington (born February 4, 1958 in San Francisco, California) is a former professional American football cornerback for the Washington Redskins and Pittsburgh Steelers of the National Football League (NFL). He played college football at Fresno State University and was drafted in the second round of the 1981 NFL Draft. Washington started for the Redskins in Super Bowl XVIII.",'qas': [{'question': "Where did Anthony Wayne Washington go to college?", 'id': '26'}]}])

to_predict.append([{"context": "Anthony Wayne Washington (born February 4, 1958 in San Francisco, California) is a former professional American football cornerback for the Washington Redskins and Pittsburgh Steelers of the National Football League (NFL). He played college football at Fresno State University and was drafted in the second round of the 1981 NFL Draft. Washington started for the Redskins in Super Bowl XVIII.",'qas': [{'question': "What Super Bowl was Anthony Wayne Washington in?", 'id': '27'}]}])

to_predict.append([{"context": "The Shulba Sutras or Śulbasūtras Sanskrit śulba: 'string, cord, rope' are sutra texts belonging to the Śrauta ritual and containing geometry related to fire-altar construction. The Shulba Sutras are part of the larger corpus of texts called the Shrauta Sutras, considered to be appendices to the Vedas. They are the only sources of knowledge of Indian mathematics from the Vedic period. Unique fire-altar shapes were associated with unique gifts from the Gods.",'qas': [{'question': "What period are the Shulba Sutras from?", 'id': '28'}]}])

to_predict.append([{"context": "The Shulba Sutras or Śulbasūtras Sanskrit śulba: 'string, cord, rope' are sutra texts belonging to the Śrauta ritual and containing geometry related to fire-altar construction. The Shulba Sutras are part of the larger corpus of texts called the Shrauta Sutras, considered to be appendices to the Vedas. They are the only sources of knowledge of Indian mathematics from the Vedic period. Unique fire-altar shapes were associated with unique gifts from the Gods.",'qas': [{'question': "What are the Shulba Sutras?", 'id': '29'}]}])

to_predict.append([{"context": "Sidewalk Labs is Alphabet Inc.'s urban innovation organization. Its goal is to improve urban infrastructure through technological solutions, and tackle issues such as cost of living, efficient transportation and energy usage. It is headed by Daniel L. Doctoroff, former deputy mayor of New York City for economic development and former chief executive of Bloomberg L.P. Other members include Craig Nevill-Manning, co-founder of Google's New York office and inventor of Froogle.",'qas': [{'question': "What does Sidewalk Labs do?", 'id': '30'}]}])

to_predict.append([{"context": "Sidewalk Labs is Alphabet Inc.'s urban innovation organization. Its goal is to improve urban infrastructure through technological solutions, and tackle issues such as cost of living, efficient transportation and energy usage. It is headed by Daniel L. Doctoroff, former deputy mayor of New York City for economic development and former chief executive of Bloomberg L.P. Other members include Craig Nevill-Manning, co-founder of Google's New York office and inventor of Froogle.",'qas': [{'question': "What does Daniel Doctoroff head?", 'id': '31'}]}])

to_predict.append([{"context": "Viking Olov Björk was a Swedish cardiac surgeon. In 1968, he collaborated with American engineer Donald Shiley to develop the Björk–Shiley valve, a mechanical prosthetic heart valve. It was the first 'tilting disc valve', used to replace the aortic or mitral valve. Many modifications followed, including the convexo-concave valve. The convexo-concave valve had defects in form of strut fractures. Therefore the monostrut valve was introduced to prevent outflow strut fractures.",'qas': [{'question': "What replaces the aortic valve?", 'id': '32'}]}])

to_predict.append([{"context": "Viking Olov Björk was a Swedish cardiac surgeon. In 1968, he collaborated with American engineer Donald Shiley to develop the Björk–Shiley valve, a mechanical prosthetic heart valve. It was the first 'tilting disc valve', used to replace the aortic or mitral valve. Many modifications followed, including the convexo-concave valve. The convexo-concave valve had defects in form of strut fractures. Therefore the monostrut valve was introduced to prevent outflow strut fractures.",'qas': [{'question': "Who do Donald Shiley collaborate with?", 'id': '33'}]}])

to_predict.append([{"context": "Henry Gross (born April 1, 1951) is an American singer-songwriter best known for his association with the group Sha Na Na and for his hit song, 'Shannon'. At age 18, while a student at Brooklyn College, Gross became a founding member of 1950's Rock & Roll revival group, Sha Na Na, playing guitar and wearing the greaser clothes he wore while a student at Midwood High School. He produced a single, 'Shannon', a song written about the death of former Beach Boys member Carl Wilson's dog, who was named Shannon.",'qas': [{'question': "What is Sha Na Na?", 'id': '34'}]}])

to_predict.append([{"context": "Henry Gross (born April 1, 1951) is an American singer-songwriter best known for his association with the group Sha Na Na and for his hit song, 'Shannon'. At age 18, while a student at Brooklyn College, Gross became a founding member of 1950's Rock & Roll revival group, Sha Na Na, playing guitar and wearing the greaser clothes he wore while a student at Midwood High School. He produced a single, 'Shannon', a song written about the death of former Beach Boys member Carl Wilson's dog, who was named Shannon.",'qas': [{'question': "What was Gross' hit song?", 'id': '35'}]}])

to_predict.append([{"context": "Henry Gross (born April 1, 1951) is an American singer-songwriter best known for his association with the group Sha Na Na and for his hit song, 'Shannon'. At age 18, while a student at Brooklyn College, Gross became a founding member of 1950's Rock & Roll revival group, Sha Na Na, playing guitar and wearing the greaser clothes he wore while a student at Midwood High School. He produced a single, 'Shannon', a song written about the death of former Beach Boys member Carl Wilson's dog, who was named Shannon.",'qas': [{'question': "Was there a song about a dog?", 'id': '36'}]}])

to_predict.append([{"context": "The Juniores are the Parma football team composed of footballers between 17 and 20 years old, which is the most senior youth category according to Italian football's hierarchy. Each season, the squad is trialled for promotion to the first team before the beginning of the season. Players deemed ready for first team football are registered. The team has competed in the Italian Campionato Nazionale Primavera, which has been known as the Trofeo Giacinto Facchetti since 2006, but has never won the title.",'qas': [{'question': "What's the age range for the Juniores?", 'id': '37'}]}])

to_predict.append([{"context": "The Juniores are the Parma football team composed of footballers between 17 and 20 years old, which is the most senior youth category according to Italian football's hierarchy. Each season, the squad is trialled for promotion to the first team before the beginning of the season. Players deemed ready for first team football are registered. The team has competed in the Italian Campionato Nazionale Primavera, which has been known as the Trofeo Giacinto Facchetti since 2006, but has never won the title.",'qas': [{'question': "Where have the Juniores competed?", 'id': '38'}]}])

to_predict.append([{"context": "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.",'qas': [{'question': "When did Michael Kmit die?", 'id': '39'}]}])

to_predict.append([{"context": "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.",'qas': [{'question': "Who won the Blake Prize in 1952?", 'id': '40'}]}])

to_predict.append([{"context": "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.",'qas': [{'question': "When did Michael Kmit win the Sulman Prize?", 'id': '41'}]}])

to_predict.append([{"context": "Michael Kmit (Ukrainian: Михайло Кміт) (25 July 1910 in Stryi, Lviv – 22 May 1981 in Sydney, Australia) was a Ukrainian painter who spent twenty-five years in Australia. He is notable for introducing a neo-Byzantine style of painting to Australia, and winning a number of major Australian art prizes including the Blake Prize (1952) and the Sulman Prize (in both 1957 and 1970). In 1969 the Australian artist and art critic James Gleeson described Kmit as 'one of the most sumptuous colourists of our time'.",'qas': [{'question': "What did Kmit bring to Australia?", 'id': '42'}]}])

to_predict.append([{"context": "Lindsay Ann Czarniak (born November 7, 1977), is an American sports anchor and reporter. She currently works for Fox Sports as a studio host for NASCAR coverage and a sideline reporter for NFL games. After spending six years with WRC-TV, the NBC owned-and-operated station in Washington, D.C., Czarniak joined ESPN as a SportsCenter anchor in August 2011 and left ESPN in 2017. Czarniak served as a host and sportsdesk reporter for NBC Sports coverage of the 2008 Summer Olympics in Beijing, China.",'qas': [{'question': "Is Lindsey Czarniak English?", 'id': '43'}]}])

to_predict.append([{"context": "Lindsay Ann Czarniak (born November 7, 1977), is an American sports anchor and reporter. She currently works for Fox Sports as a studio host for NASCAR coverage and a sideline reporter for NFL games. After spending six years with WRC-TV, the NBC owned-and-operated station in Washington, D.C., Czarniak joined ESPN as a SportsCenter anchor in August 2011 and left ESPN in 2017. Czarniak served as a host and sportsdesk reporter for NBC Sports coverage of the 2008 Summer Olympics in Beijing, China.",'qas': [{'question': "When did Lindsey Czarniak work at ESPN?", 'id': '44'}]}])

to_predict.append([{"context": "Doris Lo (Chinese: 盧素娟, Pinyin: Lú Sùjuān; 20 July 1952 – 22 July 2006) was a Hong Kong voice actor who was best known for voicing the character Nobita Nobi for the Hong Kong version of the anime along with Lam Pou-chuen who voices the character Doraemon. Lo died at the age of 54 from colorectal cancer at Shatin Hospital in Hong Kong.",'qas': [{'question': "Who voiced Doraemon?", 'id': '45'}]}])

to_predict.append([{"context": "Doris Lo (Chinese: 盧素娟, Pinyin: Lú Sùjuān; 20 July 1952 – 22 July 2006) was a Hong Kong voice actor who was best known for voicing the character Nobita Nobi for the Hong Kong version of the anime along with Lam Pou-chuen who voices the character Doraemon. Lo died at the age of 54 from colorectal cancer at Shatin Hospital in Hong Kong.",'qas': [{'question': "What character did Doris Lo voice?", 'id': '46'}]}])

to_predict.append([{"context": "Czech Republic has four main daily newspapers: Lidové noviny (former dissident publication); Mladá fronta DNES (with a centre-right orientation); Právo (with a centre-left political position) and Blesk, all based in Prague. Both Lidové noviny and Mladá fronta DNES are a part of the MAFRA publishing group, owned by Andrej Babiš, the current Prime Minister of the Czech Republic. As of 2018, the MAFRA group is a part of a trust fund along with other Babiš's companies.",'qas': [{'question': "What Czech Republic newspaper has a centre-left political position?", 'id': '47'}]}])

to_predict.append([{"context": "Czech Republic has four main daily newspapers: Lidové noviny (former dissident publication); Mladá fronta DNES (with a centre-right orientation); Právo (with a centre-left political position) and Blesk, all based in Prague. Both Lidové noviny and Mladá fronta DNES are a part of the MAFRA publishing group, owned by Andrej Babiš, the current Prime Minister of the Czech Republic. As of 2018, the MAFRA group is a part of a trust fund along with other Babiš's companies.",'qas': [{'question': "Where is Lidové noviny based?", 'id': '48'}]}])

iteratorInt = 0
tempPredictionList = []

for x in to_predict:
	f = open("QA-Results.csv", "a+")                                    # Open CSV document
	tempPredictionList.append(model.predict(x))
	print("\n Result "+str(iteratorInt)+": "+(tempPredictionList[iteratorInt])[0]['answer']+"\n")
	f.write("\""+str(iteratorInt)+"\", \""+(tempPredictionList[iteratorInt])[0]['answer']+"\";\n")
	f.close()                                                           # Close CSV document
	iteratorInt+=1

# print(model.predict(to_predict))
