import subprocess
import os

# def run(corpus, situation, src, tgt, sentence_type):
def run( in_A3_final_file, in_aligned_file,  out_del_file, out_add_file, in_A3_dir="" ,in_aligned_dir="", out_dir="",):
    # make vocab files and rearranged files
    in_A3_final_file = in_A3_dir + in_A3_final_file
    in_aligned_file = in_aligned_dir + in_aligned_file
    out_del_file = out_dir + out_del_file
    out_add_file = out_dir + out_add_file
    command = ['python3', 'get_unaligned.py', 
                "-i", in_A3_final_file, 
                "-a", in_aligned_file, 
                "-od", out_del_file, 
                "-oa", out_add_file]
    proc = subprocess.Popen(command)    #->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()
  

if __name__ == "__main__":
    run(
        "cejc/apology/original_query_rewrited_query.A3.final",

        "cejc/apology/rewrited_query.out",
        
        "cejc/apology/rewrited_query.del",
        "cejc/apology/rewrited_query.add",

        "../GIZA++-v2/gizaed_data/",
        "aligned_data/",
        "unaligned_index/"
        )

