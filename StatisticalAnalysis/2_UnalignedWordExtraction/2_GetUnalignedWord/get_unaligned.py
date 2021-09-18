import os
import re
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="get index of unaligned words")
    parser.add_argument(
        "--in_A3_final_file", "-i", default="../GIZA++-v2/gizaed_data/cejc/apology/original_query_rewrited_query.A3.final", help="input A3.final file"
    )
    parser.add_argument(
        "--in_aligned_file", "-a", default="aligned_data/cejc/apology/rewrited_query.out", help="input aligned data file"
    )
    parser.add_argument(
        "--out_del_file", "-od", default="unaligned_index/cejc/apology/rewrited_query.del", help="output file"
    )
    parser.add_argument(
        "--out_add_file", "-oa", default="unaligned_index/cejc/apology/rewrited_query.add", help="output file"
    )

    args = parser.parse_args()
    src_len = []
    tgt_len = []
    with open(args.in_A3_final_file, "r", encoding="utf-8") as f:
        for line in f:
            if line[0] == "#":
                temp = re.findall(r"\d+", line)[1:3]
                # print(result)     
                try:
                    src_len.append(temp[0])
                    tgt_len.append(temp[1])
                except:
                    print(line,'skipped to process this line')
    del_index = []
    add_index = []
    with open(args.in_aligned_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            pairs = line.split(' ')
            src_index = []
            tgt_index = []
            for pair in pairs:
                s, t = pair.split("-")
                src_index.append(int(s))
                tgt_index.append(int(t))
            del_tmp = []
            add_tmp = []
            for idx in range(int(src_len[i])):
                if idx not in src_index:
                    del_tmp.append(str(idx))
            for idx in range(int(tgt_len[i])):
                if idx not in tgt_index:
                    add_tmp.append(str(idx))
            del_index.append(" ".join(del_tmp))
            add_index.append(" ".join(add_tmp))

    os.makedirs(os.path.dirname(args.out_del_file), exist_ok=True)
    with open(args.out_del_file, "w", encoding="utf-8") as f:
        for x in del_index:
            f.write(x)
            f.write("\n")
    with open(args.out_add_file, "w", encoding="utf-8") as f:
        for x in add_index:
            f.write(x)
            f.write("\n")
