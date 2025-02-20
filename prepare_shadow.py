import pickle as pkl
import os
import numpy as np
import shutil
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon-book")
    args = parser.parse_args()

    dataset = args.dataset
    original_loc = "KGAT_new/datasets/{}/".format(dataset)
    train_file = "KGAT_new/datasets/{}/train.txt".format(dataset)
    # partial + shadow (same domain)
    partial_ratio = 0.2
    shadow_ratio = 0.4
    lines = open(train_file, 'r').readlines()
    lines_shadow = []
    lines_TP = []
    lines_Tnp = []
    for l in lines:
        decision = np.random.choice(3, 1, p=[0.3, 0.56, 0.14])[0] #shadow, Training not P, training but P
        if decision == 0:
            lines_shadow.append(l)
        elif decision == 1:
            lines_Tnp.append(l)
        else:
            lines_Tnp.append(l)
            lines_TP.append(l)
            lines_shadow.append(l)
    # write shadow dataset
    shadow_folder = "KGAT_new/datasets/{}-shadow/".format(dataset)
    if not os.path.exists(shadow_folder):
        os.makedirs(shadow_folder)

    # clear files before writing
    files = glob.glob(shadow_folder + '*.txt')
    for f in files:
        try:
            os.remove(f)
            print("Removed " + f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    shadow_file = shadow_folder + "train_shadow.txt"
    target_file = shadow_folder + "train.txt"
    with open(shadow_file, 'w') as fout:
        for l in lines_shadow:
            fout.write(l)
    fout.close()
    with open(target_file, 'w') as fout:
        for l in lines_Tnp:
            fout.write(l)
    fout.close()

    for other_file in ['entity_list.txt', 'item_list.txt', 'kg_final.txt', 'relation_list.txt', 'test.txt', 'user_list.txt']:
        if other_file == 'item_list.txt' and dataset == "yelp2018":
            lines = open(original_loc + other_file, 'r').readlines()
            id_count = -1
            new_lines = []
            for l in lines:
                if id_count < 0:
                    id_count += 1
                    new_lines.append(l)
                    continue
                tmp = l.strip().split(" ")
                len_id = len(str(id_count))
                if tmp[1][:len_id] != str(id_count):
                    print("Somthing is wrong with line: {}\n the id should be {}".format(l, id_count))
                    exit(1)
                new_line = " ".join([tmp[0], str(id_count), tmp[1][len_id+1:]]) + '\n'
                new_lines.append(new_line)
                id_count += 1
            with open(shadow_folder + other_file, 'w') as fout:
                for l in new_lines:
                    fout.write(l)
            fout.close()
        else:
            shutil.copy2(original_loc + other_file, shadow_folder + other_file)

