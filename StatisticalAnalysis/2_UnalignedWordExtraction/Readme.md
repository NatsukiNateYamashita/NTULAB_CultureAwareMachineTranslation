# Get Unaligned Words using GIZA++

## Purpose
Get unaligned words using GIZA++

## Background
We created a parallel dialogue dataset in Chinese and Japanese in apology, request, thanks situations.
The parallel dataset contains pre-translated, machine-translated and human-translated (culture-aware translated) texts.
We would like to discover differences between machine-translated and human-translated texts for reduce negative impacts on relationships in cross-cultural conversation.
We expect that the differences in the language uses of certain situations between the machine-translated and the human-translated dialogue texts potentially exist in the unaligned words.

## What we do

+ Align words using [Giza++](https://github.com/moses-smt/giza-pp/tree/master/GIZA%2B%2B-v2) for
    + pre-translated - machine-translated
    + pre-translated - human-translated
+ Create aligned words matrix using [github.com/Deep1994/align_sym_plot](https://github.com/Deep1994/align_sym_plot) for them
+ Get unaligned words of each

## Usage
+ Install and run [Giza++](https://github.com/moses-smt/giza-pp/tree/master/GIZA%2B%2B-v2) with dataset. The usage of [Giza++](https://github.com/moses-smt/giza-pp/tree/master/GIZA%2B%2B-v2) is [HERE](https://github.com/moses-smt/giza-pp/tree/master/GIZA%2B%2B-v2).
```
python 0_DataShaping/run_all.py
```

+ Install [github.com/Deep1994/align_sym_plot](https://github.com/Deep1994/align_sym_plot) in '1_FastAlign'.
+ Compile it.
```
python 1_FastAlign/run_all.py
```

```
python 2_GetUnalignedWord/run_all_get_unaligned.py
```

## Reference
[Giza++](https://github.com/moses-smt/giza-pp/tree/master/GIZA%2B%2B-v2)
[github.com/Deep1994/align_sym_plot](https://github.com/Deep1994/align_sym_plot)
