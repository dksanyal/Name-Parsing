# Person Name Segmentation Using Deep Learning

Personal names often need to be represented in a consistent format in an application. For example, in a library catalog or bibliography, author names are often represented as <LASTNAME, REMAININGNAME, SUFFIX>, or <LN, RN, SFX>. It is a particular challenge encountered while curating metadata in a digital library. A library may receive a resource like a book or a research paper from which it has to extract the relevant metadata. Identifying the name components is a big challenge owing to the diverse name writing conventions across the world. 

Example: 

"Raymond Lawrence Jr." => LN = "Lawrence", SFX = "Jr.", RN = "Raymond"


We devise a new machine learning-based tool to annotate a personal name into these components. This new technique helps us avoid the need of experts in framing rules for name annotation. Specifically, we use a BiLSTM network to learn name annotations. We compare our results with HMM as a baseline. The best results are obtained with character-level BiLSTM-CRF network.

## CODE

LSTM code: Present in CodeAndData/code/DL. The paths of training and test data in the python files must be updated.
HMM code: Present in CodeAndData/code/HMM. Files: hmmlearn.py (HMM with Laplace smoothing), hmmlearnabs.py (HMM with absolute discounting), hmmdecode.py (Viterbi algorithm)

Code to compare outputs with golden annotations: resultcompare.py 


## DATASET

We use metadata from the National Digital Library of India (NDLI) to train and test our parsers. It comprises author names from scholarly IEEE publications indexed by NDLI. The name components have been labeled by a rule-based system with some manual post-processing.

Training subset: train_names.txt, train_states.txt.  
Test subset: test_names.txt, test_states.txt

The code and dataset are freely available for research. We request you to kindly cite the following paper in case you use the code and/or data or just find the paper useful for your research.

Santosh T.Y.S.S., Sanyal D.K., Das P.P. (2020) Person Name Segmentation with Deep Neural Networks. In: B. R. P., Thenkanidiyoor V., Prasath R., Vanga O. (eds) Mining Intelligence and Knowledge Exploration. MIKE 2019. Lecture Notes in Computer Science, vol 11987. Springer, Cham.


