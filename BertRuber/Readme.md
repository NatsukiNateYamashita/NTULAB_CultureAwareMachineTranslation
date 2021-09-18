# BertRuber for Measuring Cultural-Unnaturalness for CH2JA and JA2CH Machine Translation in Dialogue

## Purpose
Verify that the current machine translators do a culturally unnatural translation.

## Back-ground
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated and human-translated (culture-aware translated) texts.
We get unaligned words using GIZA++ from them.
We would like to discover differences between machine-translated and human-translated texts for reduce negative impacts on relationships in cross-cultural conversation.

Many empirical studies in cross-cultural communication or applied pragmatics mention the problems caused by cultural differences.
When translating one language to another by human, even by machine, the translated utterances could not fit the conversation culture of the target language.
In other words, in a relevant situation, the machine-translated utterance would be contrary to the pragmatic rules of the target language. 
As result, it has a negative impact on the relationships between the people in the conversation.
Therefore, we attempt to verify whether this problem exists by measuring Cultural-Unnaturalness for CH2JA and JA2CH Machine Translation in Dialogue.

## Premise
Current machine translators are enough good to translate documents in terms of the meanings

## Hypothesis
When you respond to the other in a certain situation, the response would have the likeness of the language.
For example, the likeness of Japanese could be that response for gratitude is tend to be gratitude.

## Goal
Using the dialogue of a native speaker of language A as a reference, compare the dialogue of a native speaker of language B translated by a machine into language A with that translated by a human(culture-aware translation).

Then make sure the following:
The human-translated version more closely resembles the exchange made by a native speaker of language A.
The opportunity translation does not resemble the interaction conducted by the native speaker of language A.

## Data
### Corpora
+ [MPDD](http://nlg.csie.ntu.edu.tw/nlpresource/MPDD/)
+ [CEJC](https://www2.ninjal.ac.jp/conversation/cejc.html) 

### Data Processing
#### Data Cleaning
for CECJ

+ Remove tags
+ Concat consequent utterances by the same speaker into one utterance

#### Data Selection
We only use the utterances in certain situations: Apology, Request, and Thanks.
These utterances have situation labels which we annotated.

#### Translation Data Creation
1. Machine Translated Data by [DeepL](https://www.deepl.com/en/translator)
2. Human Translated Data by 1 Japanese and 1 Taiwanese (These data are surely culture-aware translated.)

## Method
### Metric
[BertRuber](https://www.researchgate.net/publication/334600238_Better_Automatic_Evaluation_of_Open-Domain_Dialogue_Systems_with_Contextualized_Embeddings)

When you measure the likeness of Japanese:
+ Train model on the pair of situational utterances and the response by native Japanese.
+ Measure the pair of them which is translated by machine/human.

## Model
Multiligual Bert + Semi-hard negative sampling

## 
Data should be prepared like:
- data
    - corpora
        - situations
            - original_neg.csv
            - original_query.csv
            - original_res.csv
            - machine_neg.csv
            - machine_query.csv
            - machine_res.csv
            - human_neg.csv
            - human_query.csv
            - human_res.csv
                ...
```
python create_dataset.py args
python get_embedding.py args
python main.py args
```

## Reference
@inproceedings{inproceedings,
author = {Ghazarian, Sarik and Wei, Johnny and Galstyan, Aram and Peng, Nanyun},
year = {2019},
month = {01},
pages = {82-89},
title = {Better Automatic Evaluation of Open-Domain Dialogue Systems with Contextualized Embeddings},
doi = {10.18653/v1/W19-2310}
}

[github.com/gmftbyGMFTBY/RUBER-and-Bert-RUBER](https://github.com/gmftbyGMFTBY/RUBER-and-Bert-RUBER)