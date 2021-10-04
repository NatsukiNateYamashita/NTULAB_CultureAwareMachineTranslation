#  Culture-Aware Machine Translation for CH2JA and JA2CH

## Purpose
In order to reduce the problem caused by cultural differences in cross-cultural communication, generate culture-aware translation which has more "appropriate" language uses in the relevant situation for the relevant culture

## Back-ground
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated, and human-translated (culture-aware translated) texts.

Many empirical studies in cross-cultural communication or applied pragmatics mention the problems caused by cultural differences.
When translating one language to another by human, even by machine, the translated utterances could not fit the conversation culture of the target language.
In other words, in a relevant situation, the machine-translated utterance would be contrary to the pragmatic rules of the target language.
As result, it has a negative impact on the relationships between the people in the conversation.

Therefore, we attempt to generate culture-aware translation which has more "appropriate" language uses in the relevant situation for the relevant culture

## Hypothesis
When you utter something to the other in a certain situation, the utterance would have the likeness of the language.
For example, the likeness of Japanese could be the tendencies that gratitude or apology is frequently used (for the maintenance of relationships).

The score of classification on human-translated texts (culture-aware translation) is higher than that on machine-translated texts because a human can be aware of pragmatic rules which define "appropriate" language uses in the relevant contexts.
Awareness of contextual information is helpful for an improvement of classification because the appropriate language use in a given culture depends on the pragmatic rules (contexts).

## Goal
Generate culture-aware translation which has more "appropriate" language uses in the relevant situation for the relevant culture

## Method
To ensure that the sequence-to-sequence model recognizes appropriate language uses in the relevant dialogue cultures, we classify situations using our dataset and compare each classification score.

### Contextual Information Options
+ interpersonal relationship
+ previous utterances
+ situation name
+ corpus name
+ language name
+ the utterance type (the situational utterance or the response to it)

### Metric
F1-BERTScore
Our grand truth data, which is human-translated text data, is culture-aware translated.
The meanings could be different between the pre-translated text and the human-translated text.

BERTScore which measures the similarity of contextualized embeddings infers the similarity of meanings.
Therefore, it is suitable for our experiment to measuring the similarity between generated translation and grand truth (human-translated text).

### Model
+ Multilingual T5

### Input Format
\[contextual information\]: query:\[text to be translated\] context: \[previous utterance(s)\]

## Usage
+ Run all 'translate*.ipynb' or 'styletransfer*.ipynb' which you would like to run.
  Note:
  + translate*.ipynb:
    - input: pre-translated texts
    - output: texts in the target language
  + styletransfer*.ipynb:
    - input: machine translated texts into the target language
    - output: texts in the target language
  + styletransfer_t5out*.ipynb:
    - input: machine translated texts into the target language by 'translate*.ipynb'
    - output: texts in the target language
+ Run all 'analyze_scores.ipynb'
+ Run all 'selfbleu.ipynb'
## Reference
https://github.com/ThilinaRajapakse/simpletransformers
