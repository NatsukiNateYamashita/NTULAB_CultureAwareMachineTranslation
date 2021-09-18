from run import run

corpora = ["cejc","mpdd"]
situations = ["request","apology","thanksgiving"]
methods = ["rewrited","translated"]
types = ["query", "res"]
data_dir = "../mrphdata/"

for corpus in corpora:
    for situation in situations:
        for method in methods:
            for typ in types:
                run(f"{corpus}/{situation}/original_{typ}",f"{corpus}/{situation}/{method}_{typ}", data_dir)
                print(f'finish running: {corpus} {situation} {method} {typ}')