# BERT_vs_Transformer-XL

## A Comparison of Two NLP Frameworks for General Research Purposes

The goal of Natural Language Processing (NLP) is to train computers to analyze human language. The widest-used versions of NLP are used in spell-check and grammar-check programs, but more advanced versions have been developed into tools used for much more than just identifying context within search queries. NLP is becoming increasingly more useful for researchers to summarize large amounts of data or long-form documents without the need for human supervision. Our project will examine two powerful NLP algorithms, BERT and Transformer-XL, in their abilities to extract and summarize data from chosen pieces of literature. Both have the attention model Transformer as their base. “[Transformer-XL] consists of a segment-level recurrence mechanism and a novel positional encoding scheme” (Dai, et al. 2019), meaning it takes segments of data and not only individually analyzes each segment, but also references segments against each other for increased accuracy regarding context. BERT focuses on working around the Transformer constraint of unidirectionality, where context is analyzed in only 1 direction, leaving room for error when the context from the other direction is needed. The strategy for its bidirectionality is “using a ‘masked language model’ (MLM) pre-training objective,” which “randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word using only the [left and right] context” (Devlin, et al. 2019). We will provide each algorithm with the same dataset and judge the results for each algorithm on its accuracy compared to its execution time.

### Works Cited

Dai, Zihang, et al. “Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.” ArXiv.org, Cornell University, 2 June 2019, arxiv.org/abs/1901.02860.

Devlin, Jacob, et al. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” ArXiv.org, Cornell University, 24 May 2019, arxiv.org/abs/1810.04805.


## Weekly Progress Reports

2/14/20: We've uploaded our abstract to the MassURC website and specified our needs for presentation. We created the GitHub repository for this project.

2/22/20: We updated the abstract to have quotes from the two research papers we're going to reference in our project. Ed has been working on setting up his company's server to run BERT, and will be providing access to Vincent and Connor soon. Vincent and Connor have been researching how to use/understand the results from both algorithms.

