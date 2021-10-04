# Analysis for Unaligned Words

## Purpose
Discover how the machine-translated and the human-translated dialogue texts are different

## Background
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated, and human-translated (culture-aware translated) texts.

## What we do

+ Statistical analysis
    + words, their POS, cooccurrence
    + unaligned words using GIZA++
    + meaningful unaligned words using LIWC
    + pragmatic markers on sentences include meaningful unaligned words
+ Pragmatic markers classification
    + directness
    + intensity
    + perspective
