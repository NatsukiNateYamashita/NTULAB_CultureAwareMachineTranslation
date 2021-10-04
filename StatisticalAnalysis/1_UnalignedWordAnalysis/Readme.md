# Analysis for Unaligned Words

## Purpose
Discover how the machine-translated and the human-translated dialogue texts are different

## Background
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated and human-translated (culture-aware translated) texts.
We get unaligned words using GIZA++ from them.
We would like to discover differences between machine-translated and human-translated texts for reduce negative impacts on relationships in cross-cultural conversation.
We expect that the differences in the language uses of certain situations between the machine-translated and the human-translated dialogue texts potentially exist in the unaligned words.

## What we do
Analysis for the unaligned words and their POS

## Usage
```
python create_datable
```

+ Run all 'visualize_ranking_of_unalignedword_pos.ipynb'
+ Run all 'visualize_ranking_unalignedword_pos_by_relation.ipynb'
+ Run all 'visualize_num_of_unalignedword_by_each_category.ipynb'
+ Run all 'measure_sim_sentences_by_contextualized_embeddings.ipynb'

Run other programs if you need
