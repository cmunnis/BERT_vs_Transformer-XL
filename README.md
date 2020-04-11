# BERT_vs_Transformer-XL

## A Comparison of Two NLP Frameworks for General Research Purposes

The goal of Natural Language Processing (NLP) is to train computers to analyze human language. The widest-used versions of NLP are used in spell-check and grammar-check programs, but more advanced versions have been developed into tools used for much more than just identifying context within search queries. NLP is becoming increasingly more useful for researchers to summarize large amounts of data or long-form documents without the need for human supervision. Our project will examine two powerful NLP algorithms, BERT and Transformer-XL, in their abilities to extract and summarize data from chosen pieces of literature. Both have the attention model Transformer as their base. “[Transformer-XL] consists of a segment-level recurrence mechanism and a novel positional encoding scheme” (Dai, et al. 2019), meaning it takes segments of data and not only individually analyzes each segment, but also references segments against each other for increased accuracy regarding context. BERT focuses on working around the Transformer constraint of unidirectionality, where context is analyzed in only 1 direction, leaving room for error when the context from the other direction is needed. The strategy for its bidirectionality is “using a ‘masked language model’ (MLM) pre-training objective,” which “randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word using only the [left and right] context” (Devlin, et al. 2019). We will provide each algorithm with the same dataset and judge the results for each algorithm on its accuracy compared to its execution time.

#### Works Cited

Dai, Zihang, et al. “Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.” ArXiv.org, Cornell University, 2 June 2019, arxiv.org/abs/1901.02860.

Devlin, Jacob, et al. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” ArXiv.org, Cornell University, 24 May 2019, arxiv.org/abs/1810.04805.


## Weekly Progress Reports

**2/14/20**: We've uploaded our abstract to the MassURC website and specified our needs for presentation. We created the GitHub repository for this project.

**2/22/20**: We updated the abstract to have quotes from the two research papers we're going to reference in our project. Ed has been working on setting up his company's server to run BERT, and will be providing access to Vincent and Connor soon. Vincent and Connor have been researching how to use/understand the results from both algorithms.

**2/28/20**: It was a bit close, but we have successfully been able to implement BERT onto our server, and are able to demonstrate it executing and working. With the structure of BERT implemented, our goal now shifts from the basics of BERT to changing its dataset manually.

**3/6/20**: Alongside the change in data set, we are also in the process of modifying the function to output, and save the generated results, the input, and the amount of time, in milliseconds into a text file with a similar syntax to that of a JSON file. Both this, and changing our bert's data set (from the IMDB Database to the Wikipedia Database), are still a work in progress, but we are getting closer to its completion.
   We've converted our TF BERT program to operate on Tensorflow 2, improving functionality and allowing use with Keras. 
   We've installed transformer-xl onto our server and are writing a keras script for building, finetuning and testing our transformer-xl model. 
   
**4/2/20**: Amongst other goals, scripts are being developed to significantly speed-up the testing and comparing process, to hopefully increase development efficiency.
- *Edward*:   
   Implemented entity-recognition.py for the Bert model, which can be viewed within our repository. This script is the counterpart to our entity recognition script for transformer xl. Both scripts read through a prepared script and identify entities that are containned within the scripts. We can use the combination of accuracy of the two models ability to find entities as well as the specificity of the entity categorization to judge the efficiency of the two models against each other. Our testing process will allow us to see how each model responds as the text used for entity analysis grows longer. 
   Additionally, background research will need to be done to see if bias is introduced due to the entity-recognition script utilizing pytorch rather than tensorflow-2, to see if it would be valuable to refactor the pipeline transformer-xl function to utilize pytorch as well.    
- *Vincent*:
   Currently in the process of creating intuitive testing and comparing scripts that will allow for increased productivity, as well as reformat the output of both BERT and Transformer-XL in a way that will allow both to be comparible to each other, and their output sent to different columns of a spreadsheet file, along with the base input, for ease of recording and comparing.
- *Connor*:
   Continued to: read up on Machine Learning; learn Python to be able to help make the necessary changes; and look up examples of how to find/understand output for each model.

**Goal for Next Week**:
We need to have a video or demonstration showing that we're able to run BERT.
