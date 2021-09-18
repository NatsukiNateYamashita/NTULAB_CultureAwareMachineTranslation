import subprocess
import os

def run(file1_name, file2_name, data_dir="", method="grow-diag-final-and"):
    
    file1_path = data_dir + file1_name
    file2_path = data_dir + file2_name
    save_dir = "aligned_data/"
    save_fpath = save_dir + file1_name[:-8] + ".out"

    os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
    command = ["fast_align/build/atools",
            "-i", file1_path,
            "-j", file2_path,
            "-c", method]
    with open(save_fpath, 'w') as f:
        proc = subprocess.Popen(command,stdout=f)  #->コマンドが実行される(処理の終了は待たない)
        result = proc.communicate()

if __name__ == "__main__":
    run("cejc/apology/rewrited_query.extract","cejc/apology/rewrited_query.rev.extract","extracted_data/")

