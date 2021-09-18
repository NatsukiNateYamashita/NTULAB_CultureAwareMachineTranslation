import subprocess
import os

# def run(corpus, situation, src, tgt, sentence_type):
def run(src_f_path, tgt_f_path, data_dir=''):
    # make vocab files and rearranged files
    src_path = data_dir + src_f_path
    tgt_path = data_dir + tgt_f_path
    command = ['./plain2snt.out', src_path, tgt_path]
    proc = subprocess.Popen(command)    #->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()
    

    # run GIZA++
    src_vcb_path = src_path + ".vcb"
    tgt_vcb_path = tgt_path + ".vcb"
    bitext_f_path = src_path + "_" + os.path.basename(tgt_path) + ".snt"
    save_dir = "gizaed_data/"
    save_f_prefix = save_dir + src_f_path + "_" + os.path.basename(tgt_f_path)
    os.makedirs(os.path.dirname(save_f_prefix), exist_ok=True)
    command = ["./GIZA++", "-s", src_vcb_path, "-t", tgt_vcb_path, "-c", bitext_f_path, "-o", save_f_prefix]
    proc = subprocess.Popen(command)    #->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()
    # For reversed
    bitext_f_path = tgt_path + "_" + os.path.basename(src_path) + ".snt"
    save_f_prefix = save_dir + tgt_f_path + "_" + os.path.basename(src_path)
    command = ["./GIZA++", "-s", tgt_vcb_path, "-t", src_vcb_path, "-c", bitext_f_path, "-o", save_f_prefix]
    proc = subprocess.Popen(command)    #->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()


    # run extract.py and get aligned_words files 
    input_f_path = save_dir + src_f_path + "_" + os.path.basename(tgt_f_path) + ".A3.final"
    output_dir = "extracted_data/"
    output_f_path = output_dir + tgt_f_path + ".extract"
    os.makedirs(os.path.dirname(output_f_path), exist_ok=True)
    command = ["python", "extract.py", "-i", input_f_path, "-o", output_f_path]
    proc = subprocess.Popen(command)    #->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()
    # For reversed 
    input_f_path = save_dir + tgt_f_path + "_" + os.path.basename(src_path) + ".A3.final"
    output_f_path = output_dir + tgt_f_path + ".rev" +".extract"
    command = ["python", "extract.py", "-i", input_f_path, "-o", output_f_path, "-r"]
    proc = subprocess.Popen(command)    #->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()

if __name__ == "__main__":
    run("cejc/apology/original_query","cejc/apology/rewrited_query","../mrphdata/")

