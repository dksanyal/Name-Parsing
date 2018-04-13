# Name-Annotation

Personal names often need to be represented in a consistent format in an application. For example, in a library catalog or bibliography, author names are often represented as <LASTNAME, SUFFIX, FORENAME>. It is a particular challenge encountered while curating metadata in a digital library. A library may receive a resource like a book or a research paper from which it has to extract the relevant metadata. Part of the challenge in name annotation stems from the diverse nationalities of the authors. 

Examples: 

"Rena Torres Cacoullos"   => LASTNAME = "Torres Cacoullos", FORENAME = "Rena"

"Raymond J. Lawrence Jr." => LASTNAME = "Lawrence", SUFFIX = "Jr.", FORENAME = "Raymond J."


We devise a new machine learning-based tool to annotate a personal name precisely into these 3 components. This new technique helps us avoid the need of experts in framing rules for name annotation. Specifically, we use bidirectional LSTM to learn name annotations.
We use labelled metadata from the National Digital Library of India (NDLI) to train the LSTM. Most of the training dataset comprises author names from scholarly papers in PubMed and SpringerLink.


TEST SET

A manually-annotated corpus of personal name is prepared to test the performance of our tool. The test set if freely available.


TEST RESULTS

We achieved an accuracy of 86\% on the above test set.


DEMO

An online demo is available for public use.

