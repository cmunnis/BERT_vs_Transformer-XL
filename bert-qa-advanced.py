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
    print('Question: "' + question + '"')
    print('Answer: "' + answer + '"')
    
    f= open("QA-Results.txt", "a+")
    f.write('Question:"' + question + '"\n')
    f.write('Answer"' + answer + ' "\n')
    f.close()

answer_question("Who was George Graham?", "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .")

answer_question("Where was George Graham born?", "George Graham was an English clockmaker, inventor, and geophysicist, and a Fellow of the Royal Society. He was born in Kirklinton, Cumberland. A Friend (Quaker) like his mentor Thomas Tompion, Graham left Cumberland in 1688 for London to work with Tompion. He later married Tompion's niece, Elizabeth Tompion.Plaque in Fleet Street, London, commemorating Thomas Tompion and George Graham Graham was partner to the influential English clockmaker Thomas Tompion during the last few years of Tompion's life .")

