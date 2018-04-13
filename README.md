# Name-Annotation
This is part of a project on automatic annotation of personal names.

Personal names often need to be represented in a consistent format in an application. For example, in a library author names are often represented as <LASTNAME, SUFFIX, FORENAME>. It is a particular challenge encountered while curating metadata in a digital library. Part of the challenge stems from the diverse nationalities of the authors. 

We devise a new machine learning-based tool to annotate a personal name precisely into these 3 components. This new technique helps us avoid the need of experts in framing rules for name annotation. We use bidirectional LSTM to learn name annotations.
We use metadata from the National Digital Library of India (NDLI) to train the LSTM. Most of the training dataset comprises author names from scholarly papers in PubMed and SpringerLink.


TEST SET

A manually-annotated corpus of personal name is prepared to test the performance of our tool. The test set if freely available.


TEST RESULTS

We achieved an accuracy of 86\% on the above test set.


DEMO

An online demo is available for public use.

