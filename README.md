# Person Name Segmentation Using Deep Learning

Personal names often need to be represented in a consistent format in an application. For example, in a library catalog or bibliography, author names are often represented as <LASTNAME, REMAININGNAME, SUFFIX>, or <LN, RN, SFX>. It is a particular challenge encountered while curating metadata in a digital library. A library may receive a resource like a book or a research paper from which it has to extract the relevant metadata. Part of the challenge in name annotation stems from the diverse nationalities of the authors. 

Example: 

"Raymond J. Lawrence Jr." => LN = "Lawrence", SFX = "Jr.", RN = "Raymond J."


We devise a new machine learning-based tool to annotate a personal name into these components. This new technique helps us avoid the need of experts in framing rules for name annotation. Specifically, we use an LSTM network to learn name annotations. We compare our results with HMM as a baseline. The best results are obtained with character-level Bidirectional LSTM network.

## CODE

LSTM code: wordBRLSTM.py, characterBRLSTM.py, seq2seq_utils.py (Note: It is trivial to change the models to unidirectional RNN.)

HMM code: hmmlearn.py (HMM with Laplace smoothing), hmmlearnabs.py (HMM with Absolute discounting), hmmdecode.py (Viterbi algorithm), hmmcompare.py (evaluate HMM by comparing output names with golden annotations)


## DATASET

We use labelled metadata from the National Digital Library of India (NDLI) to train and test our parsers. Currently the training data comprises author names from scholarly IEEE publications indexed by NDLI. The dataset is freely available for research purpose. 

Training subset: train.names, train.states. Test subset: test.names, test.states.




