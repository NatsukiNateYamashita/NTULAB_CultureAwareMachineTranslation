# Get Unaligned Words using GIZA++

## Purpose
Get unaligned words using GIZA++

## Background
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated and human-translated (culture-aware translated) texts.
We would like to discover differences between machine-translated and human-translated texts for reduce negative impacts on relationships in cross-cultural conversation.
We expect that the differences in the language uses of certain situations between the machine-translated and the human-translated dialogue texts potentially exist in the unaligned words.

## What We Do

+ Align words using [Giza++](https://github.com/moses-smt/giza-pp/tree/master/GIZA%2B%2B-v2) for
    + pre-translated - machine-translated
    + pre-translated - human-translated
+ Create aligned words matrix using [github.com/Deep1994/align_sym_plot](https://github.com/Deep1994/align_sym_plot) for them
+ Get unaligned words of each


## Reference
[Giza++](https://github.com/moses-smt/giza-pp/tree/master/GIZA%2B%2B-v2)
[github.com/Deep1994/align_sym_plot](https://github.com/Deep1994/align_sym_plot)