# Person Name Segmentation Using Deep Learning

Personal names often need to be represented in a consistent format in an application. For example, in a library catalog or bibliography, author names are often represented as <LASTNAME, REMAININGNAME, SUFFIX>, or <LN, RN, SFX>. It is a particular challenge encountered while curating metadata in a digital library. A library may receive a resource like a book or a research paper from which it has to extract the relevant metadata. Part of the challenge in name annotation stems from the diverse nationalities of the authors. 

Example: 

"Raymond J. Lawrence Jr." => LN = "Lawrence", SFX = "Jr.", RN = "Raymond J."


We devise a new machine learning-based tool to annotate a personal name into these components. This new technique helps us avoid the need of experts in framing rules for name annotation. Specifically, we use an LSTM network to learn name annotations. We compare our results with HMM as a baseline. The best results are obtained with character-level Bidirectional LSTM network.
We use labelled metadata from the National Digital Library of India (NDLI) to train the LSTM. Currently the training data comprises author names from scholarly IEEE publications indexed by NDLI.


DATASET

An annotated corpus of personal names is prepared to test the performance of our tool. The dataset is freely available for research purpose.




