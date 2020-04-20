import torch
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)

    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)

    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
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
            
    print("\""+question+"\", ")
    print("\""+answer+"\", \"NULL\"\n")                                 # By leaving the third column as "NULL", we create an easy search-index to be overwritten by the transformer-xl results, 
                                                                        # generated later.
    
                                                                        # Note: CSV files separare columns with commas (outside of strings), and rows with newline characters
    
    f= open("QA-Results.csv", "a+")                                     # Open Document
    f.write("\""+question+"\", ")                                       # Write: "Question", "Bert_Answer", "NULL"\n
    f.write("\""+answer+"\", \"NULL\"\n")                               #
    f.close()                                                           # Close Document


print("\"Question\", \"Bert Answer\", \"Transformer-XL Answer\"\n")     # Previews the first line of the CSV to the one executing the file

f= open("QA-Results.csv", "a+")                                         # Open Document
f.write("\"Question\", \"Bert Answer\", \"Transformer-XL Answer\"\n")   # Puts the category titles in the first line of the CSV
f.close()                                                               # Close Document

answer_question("Who was George Graham?", "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .")

answer_question("Where was George Graham born?", "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .")

answer_question("How large is the Cameron House estate?", "Cameron House, located on Loch Lomond near Balloch, Scotland, was first built in the mid-1700s, and later purchased by Sir James Smollett. The modern Baronial stone castle was built by William Spence in 1830 (rebuilt after a fire in 1865), with peaked gables and decorative turrets. The House is a Category B listed building. For three centuries, the land was part of the Smollett estate, now reduced to 44 hectares of wooded land that juts into the Loch. In 1985 Laird Patrick Telfer Smollett sold the House and land to De Vere Hotels.")

answer_question("Who owns Hawaiian Airlines?", "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.")

answer_question("What airline has never had a fatal accident?", "Hawaiian Airlines is the flag carrier and the largest airline in the U.S. state of Hawaii. It is the tenth-largest commercial airline in the US, and is based in Honolulu, Hawaii. The airline operates its main hub at Daniel K. Inouye International Airport on the island of Oʻahu and a secondary hub out of Kahului Airport on the island of Maui. Hawaiian Airlines is owned by Hawaiian Holdings, Inc. of which Peter R. Ingram is the current President and Chief Executive Officer. Hawaiian is the oldest US carrier that has never had a fatal accident or a hull loss throughout its history.")

answer_question("Where did Much Ado About Nothing premiere?", "Much Ado About Nothing is a 2012 black and white American romantic comedy film adapted for the screen, produced, and directed by Joss Whedon, from William Shakespeare's play of the same name. The film stars Amy Acker, Alexis Denisof, Nathan Fillion, Clark Gregg, Reed Diamond, Fran Kranz, Sean Maher, and Jillian Morgese. To create the film, director Whedon established the production studio Bellwether Pictures. The film premiered at the 2012 Toronto International Film Festival and had its North American theatrical release on June 21, 2013.")

answer_question("Did Quentin Tarantino direct Much Ado About Nothing?", "Much Ado About Nothing is a 2012 black and white American romantic comedy film adapted for the screen, produced, and directed by Joss Whedon, from William Shakespeare's play of the same name. The film stars Amy Acker, Alexis Denisof, Nathan Fillion, Clark Gregg, Reed Diamond, Fran Kranz, Sean Maher, and Jillian Morgese. To create the film, director Whedon established the production studio Bellwether Pictures. The film premiered at the 2012 Toronto International Film Festival and had its North American theatrical release on June 21, 2013.")

answer_question("Who is Ondria Harden?", "Ondria Hardin is an American fashion model. Throughout her career she accrued controversy for her young age. Hardin was discovered at a pageant in her home state of North Carolina. She began her career in Tokyo, Japan. After being noticed by casting director Ashley Brokaw she was cast in a Prada campaign photographed by Steven Meisel by age 13. Designer Marc Jacobs was criticized for using Hardin, then aged 15, in his F/W 2012 fashion show, which he defended. She received further controversy for appearing in a Chanel campaign at age 15, and a Numéro editorial entitled African Queen.")

answer_question("Was Ondria Harden in a Chanel Campaign?", "Ondria Hardin is an American fashion model. Throughout her career she accrued controversy for her young age. Hardin was discovered at a pageant in her home state of North Carolina. She began her career in Tokyo, Japan. After being noticed by casting director Ashley Brokaw she was cast in a Prada campaign photographed by Steven Meisel by age 13. Designer Marc Jacobs was criticized for using Hardin, then aged 15, in his F/W 2012 fashion show, which he defended. She received further controversy for appearing in a Chanel campaign at age 15, and a Numéro editorial entitled African Queen.")

answer_question("When was ELIZA written?","Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted \"blocks worlds\" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966. Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the \"patient\" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to \"My head hurts\" with \"Why do you say your head hurts?\".")

answer_question("What did ELIZA do?","Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted \"blocks worlds\" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966. Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the \"patient\" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to \"My head hurts\" with \"Why do you say your head hurts?\".")

answer_question("Who created the Game & Watch brand?","The Game & Watch brand is a series of handheld electronic games produced by Nintendo from 1980 to 1991. Created by game designer Gunpei Yokoi, each Game & Watch features a single game to be played on an LCD screen in addition to a clock, an alarm, or both. It was the earliest Nintendo video game product to gain major success.")

answer_question("What did each Game & Watch feature?","The Game & Watch brand is a series of handheld electronic games produced by Nintendo from 1980 to 1991. Created by game designer Gunpei Yokoi, each Game & Watch features a single game to be played on an LCD screen in addition to a clock, an alarm, or both. It was the earliest Nintendo video game product to gain major success.")

answer_question("What occurs in ferromagnetic materials?","Magnetism is a class of physical phenomena that are mediated by magnetic fields. Electric currents and the magnetic moments of elementary particles give rise to a magnetic field, which acts on other currents and magnetic moments. Magnetism is one aspect of the combined phenomenon of electromagnetism. The most familiar effects occur in ferromagnetic materials, which are strongly attracted by magnetic fields and can be magnetized to become permanent magnets, producing magnetic fields themselves. Demagnetizing a magnet is also possible. Only a few substances are ferromagnetic; the most common ones are iron, cobalt and nickel and their alloys.")

answer_question("What substances are ferromagnetic?","Magnetism is a class of physical phenomena that are mediated by magnetic fields. Electric currents and the magnetic moments of elementary particles give rise to a magnetic field, which acts on other currents and magnetic moments. Magnetism is one aspect of the combined phenomenon of electromagnetism. The most familiar effects occur in ferromagnetic materials, which are strongly attracted by magnetic fields and can be magnetized to become permanent magnets, producing magnetic fields themselves. Demagnetizing a magnet is also possible. Only a few substances are ferromagnetic; the most common ones are iron, cobalt and nickel and their alloys.")

answer_question("How did Wendy's enter the Asian market?","Wendy's founded the fried chicken chain Sisters Chicken in 1978 and sold it to its largest franchiser in 1987. In 1979, the first European Wendy's opened in Munich. The same year, Wendy's became the first fast-food chain to introduce the salad bar. Wendy's entered the Asian market by opening its first restaurants in Japan in 1980, in Hong Kong in 1982, and in the Philippines and Singapore in 1983. In 1984, Wendy's opened its first restaurant in South Korea.")

answer_question("What happened in 1999?","The Market Theater Gum Wall is a brick wall covered in used chewing gum located in an alleyway in Post Alley under Pike Place Market in Downtown Seattle. Much like Bubblegum Alley in San Luis Obispo, California, the Market Theater Gum Wall is a local landmark. Parts of the wall are covered several inches thick, 15 feet (4.6 m) high along a 50-foot-long (15 m) section. The wall is by the box office for the Market Theater. The tradition began around 1993 when patrons of Unexpected Productions' Seattle Theatresports stuck gum to the wall and placed coins in the gum blobs. Theater workers scraped the gum away twice, but eventually gave up after market officials deemed the gum wall a tourist attraction around 1999. Some people created small works of art out of gum. It was named one of the top 5 germiest tourist attractions in 2009, second to the Blarney Stone. It is the location of the start of a ghost tour, and also a popular site with wedding photographers. The state governor, Jay Inslee, said it is his \"favorite thing about Seattle you can't find anywhere else\".")

answer_question("What did the state governor say?","The Market Theater Gum Wall is a brick wall covered in used chewing gum located in an alleyway in Post Alley under Pike Place Market in Downtown Seattle. Much like Bubblegum Alley in San Luis Obispo, California, the Market Theater Gum Wall is a local landmark. Parts of the wall are covered several inches thick, 15 feet (4.6 m) high along a 50-foot-long (15 m) section. The wall is by the box office for the Market Theater. The tradition began around 1993 when patrons of Unexpected Productions' Seattle Theatresports stuck gum to the wall and placed coins in the gum blobs. Theater workers scraped the gum away twice, but eventually gave up after market officials deemed the gum wall a tourist attraction around 1999. Some people created small works of art out of gum. It was named one of the top 5 germiest tourist attractions in 2009, second to the Blarney Stone. It is the location of the start of a ghost tour, and also a popular site with wedding photographers. The state governor, Jay Inslee, said it is his \"favorite thing about Seattle you can't find anywhere else\".")

