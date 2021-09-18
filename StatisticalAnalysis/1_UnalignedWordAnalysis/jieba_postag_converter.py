def get_upperclass_postag_convdict(dic):
    for key, value in dic.items():
        if len(key)>=2:
            dic[key] = dic[key[0]]

    return dic

def get_postag_dict(upperclass=False):
    dic = {}
    with open("jieba_postag_list", "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.split()[1:]
            if len(line) == 1:
                line.append(line[0])
            if len(line) == 3:
                line[1] = line[1] + " " + line[2]
            
            dic[line[0]] = line[1]
    
    if upperclass == True:
        dic = get_upperclass_postag_convdict(dic)
    
    return dic

if __name__=="__main__":
    dic = get_postag_dict(upperclass=False)
    print(dic)