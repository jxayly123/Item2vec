#-*-coding:utf8-*-
"""
author:ly
date:2020-2-18
produce train data for item2vec
"""
import os
from collections import defaultdict
def produce_train_data(input_file, out_file):
    """
    :param input_file: user behavior file
    :param out_file: output file
    :return:
    """
    if not os.path.exists(input_file):
        print("there's no such file in this directory")
        return
    linenum = 0
    record = defaultdict(list)
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if rating < score_thr:
            continue
        # if userid not in record:
            # record[userid] = []
        record[userid].append(itemid)
    fp.close()
    fw = open(out_file, 'w+')
    for userid in record:
        fw.write(" ".join(record[userid]) + "\n")
    fw.close()

if __name__ == "__main__":
    produce_train_data("./ml-latest-small/ratings.csv", "./ml-latest-small/train_data.txt")
