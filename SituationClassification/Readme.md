#  Classification of Situations in Dialogue for CH2JA and JA2CH Culture-Aware Machine Translation

## Purpose
Verify that a model can be aware of "appropriate" language uses in a relevant situation in a certain culture.

## Back-ground
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated, and human-translated (culture-aware translated) texts.

Many empirical studies in cross-cultural communication or applied pragmatics mention the problems caused by cultural differences.
When translating one language to another by human, even by machine, the translated utterances could not fit the conversation culture of the target language.
In other words, in a relevant situation, the machine-translated utterance would be contrary to the pragmatic rules of the target language. 
As result, it has a negative impact on the relationships between the people in the conversation.

Therefore, we attempt to verify whether a model can be aware of "appropriate" language uses in the relevant situations in Japanese and Chinese.

## Premise
Current machine translators are enough good to translate documents in terms of the meanings

## Hypothesis
When you utter something to the other in a certain situation, the utterance would have the likeness of the language.
For example, the likeness of Japanese could be the tendencies that gratitude or apology is frequently used (for the maintenance of relationships).

The score of classification on human-translated texts (culture-aware translation) is higher than that on machine-translated texts because a human can be aware of pragmatic rules which define "appropriate" language uses in the relevant contexts.
Awareness of contextual information is helpful for an improvement of classification because the appropriate language use in a given culture depends on the pragmatic rules (contexts). 

## Goal
Verify our hypothesis.

## Method
To ensure that the sequence-to-sequence model recognizes appropriate language uses in the relevant dialogue cultures, we classify situations using our dataset and compare each classification score.

### Contextual Information Options
+ interpersonal relationships
+ previous utterances
+ corpus name
+ language name
+ the utterance type (the situational utterance or the response to it)

### Classification
+ binary classification for each situation: apology, request, and thanks
+ train models for each contextual information option
+ test the models and compare the result

## Model
+ Multilingual BERT
+ Multilingual BERT + semi-hard negative sampling 
+ Multilingual BERT + semi-hard negative sampling + Domain Adaption
+ Multilingual T5

## Reference

https://github.com/fungtion/DANN
https://github.com/huggingface/transformers
https://github.com/ThilinaRajapakse/simpletransformers

