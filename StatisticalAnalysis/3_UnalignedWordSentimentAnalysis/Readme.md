# Analysis for Unaligned Words (Cont.)

## Purpose
Discover how the machine-translated and the human-translated dialogue texts are different

## Background
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated, and human-translated (culture-aware translated) texts.
We get unaligned words using GIZA++ from them.
We would like to discover differences between machine-translated and human-translated texts to reduce negative impacts on relationships in cross-cultural conversation.
However, we do not know which words are important for analysis.
Therefore, we expect to extract important words by LIWC and to do a more valuable analysis for finding the differences in the machine-translated and the human-translated dialogue texts.

## [LIWC](http://liwc.wpengine.com/) Dictionaty
+ [CLIWC](https://cliwc.weebly.com/)
+ [JIWC](https://sociocom.naist.jp/jiwc-dictionary/)

## What we do 
In /LIWC_Analysis/
+ Get unaligned word counts from machine-translated and human-translated data with LIWCs
+ Test them and get the emotion categories with statistically significant differences in their word counts
+ Analyze the words in each emotion category 

Also,
+ Annotate pragmatic marker labels on each the words (reference: Blum(1984))
    Annotate them according to the change which each word bring about from the machine-translated text to the human-translated text
    + more/less direct
    + more/less intence
        + more/less upgrader
        + more/less downgrader
        + more/less specific / 
        + more/less respectful /  
        + more/less humble /  
    + add/rmv expect_sth_in_return
    + add/rmv irony or rhetorical question
    + speaker/listener/spearker&listener/impersonal oriented

Furthermore,
In /LIWC_Analysis/PragmaticMarkerClassification/
+ Classify them by BERT

## Reference
@article{article,
author = {BLUM-KULKA, SHOSHANA and Olshtain, Elite},
year = {1984},
month = {03},
pages = {},
title = {Requests and Apologies: A Cross-Cultural Study of Speech Act Realization Patterns (CCSARP)1},
volume = {5},
journal = {Applied Linguistics},
doi = {10.1093/applin/5.3.196}
}
