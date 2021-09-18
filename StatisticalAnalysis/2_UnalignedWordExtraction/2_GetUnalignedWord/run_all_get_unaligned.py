from run_get_unaligned import run

corpora = ["cejc","mpdd"]
situations = ["request","apology","thanksgiving"]
methods = ["rewrited","translated"]
types = ["query", "res"]
in_A3_dir = "../GIZA++-v2/gizaed_data/"
in_aligned_dir = "aligned_data/"
out_aligned_dir = "unaligned_index/"

for corpus in corpora:
    for situation in situations:
        for method in methods:
            for typ in types:
                run(f"{corpus}/{situation}/original_{typ}_{method}_{typ}.A3.final",
                    f"{corpus}/{situation}/{method}_{typ}.out", 
                    f"{corpus}/{situation}/{method}_{typ}.del", 
                    f"{corpus}/{situation}/{method}_{typ}.add",
                    in_A3_dir,
                    in_aligned_dir,
                    out_aligned_dir )
                print(f'finish running: {corpus} {situation} {method} {typ}')