import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from tqdm import tqdm
from torchmetrics.functional import pairwise_cosine_similarity
import time

similarity_list = [cosine, euclidean, correlation, chebyshev,
                           braycurtis, canberra, cityblock, sqeuclidean]

sim_name_list = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                 'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
sim_ind = [0,4,6,7]
similarity_list = [similarity_list[i] for i in sim_ind]
sim_name_list = [sim_name_list[i] for i in sim_ind]
#import numpy as np
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon-book-shadow")
    parser.add_argument("--shadow_dataset", type=str, default="am-la")
    parser.add_argument("--model", type=str, default="NFM")
    parser.add_argument("--shadow_model", type=str, default="NFM")
    parser.add_argument("--training", type=int, default=1)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--partial_option", type=float, default=0)
    parser.add_argument("--random_rec", type=float, default=0)
    parser.add_argument("--random_embedding", type=float, default=0)
    parser.add_argument("--target_emb", type=int, default=0)
    parser.add_argument("--target_dim", type=int, default=64)
    parser.add_argument("--shadow_dim", type=int, default=64)
    args = parser.parse_args()

    dataset = args.dataset
    shadow_dataset = args.shadow_dataset
    shadow_model = args.shadow_model
    train_test = "shadow" if args.training else "target"
    model = args.model
    K = args.K

    original_loc = "KGAT_new/datasets/{}/".format(dataset)
    shadow_loc = "KGAT_new/datasets/{}/".format(shadow_dataset)
    user_list = shadow_loc + 'user_list.txt'
    item_list = shadow_loc + 'item_list.txt'
    user_array = []
    lines = open(user_list, 'r').readlines()
    skip = True
    for l in lines:
        tmp = l.strip()
        if skip:
            skip = False
            continue
        if len(tmp) == 0:
            continue
        line = tmp.split(" ")
        user_array.append(int(line[1]))

    item_array = []
    lines = open(item_list, 'r').readlines()
    skip = True
    for l in lines:
        tmp = l.strip()
        if skip:
            skip = False
            continue
        line = tmp.split(" ")
        item_array.append(int(line[1]))
        pass

    num_user = max(user_array) + 1
    num_item = max(item_array) + 1
    short_name_dict = {"am": "amazon-book-shadow",
                       "la": "last-fm-shadow",
                       "ye": "yelp2018-shadow"}
    if shadow_dataset not in ["amazon-book-shadow", "last-fm-shadow", "yelp2018-shadow"]:
        # check target data match or not
        if short_name_dict[shadow_dataset[:2]] != dataset and not args.training:
            print("Target dataset should be {} instead of {}".format(short_name_dict[shadow_dataset[:2]], dataset))
            exit(1)
        if shadow_dataset[3:5] in short_name_dict:
            item_source = short_name_dict[shadow_dataset[3:5]]
            item_pre_loc = "KGAT_new/datasets/{}/".format(item_source)
            print("change item list loc to: " + item_pre_loc)
            item_list = item_pre_loc + 'item_list.txt'

            item_array_pre = []
            lines = open(item_list, 'r').readlines()
            skip = True
            for l in lines:
                tmp = l.strip()
                if skip:
                    skip = False
                    continue
                line = tmp.split(" ")
                item_array_pre.append(int(line[1]))
                pass

            num_item_pre = max(item_array_pre) + 1
        else:
            num_item_pre = 0
    else:
        num_item_pre = 0

    # load item embeddings and top K recommendations for users
    embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, shadow_dataset)

    ranked_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model if args.training else model,
                                                          shadow_dataset if args.training else dataset)
    ranked_file = "{}_recommendations.csv".format(train_test)
    if args.partial_option and train_test == "shadow":
        ranked_file = "{}_recommendations({}).csv".format(train_test, args.partial_option)
    embedding_file = "shadow_embeddings({}).csv".format(args.partial_option) if args.partial_option else "shadow_embeddings.csv"
    if args.partial_option == 0.001 and args.shadow_dim != 64:
        embedding_file = embedding_file.replace(".csv", "dim_{}.csv".format(args.shadow_dim))
    if args.partial_option == 0.001:
        if train_test == "shadow" and args.shadow_dim != 64:
            ranked_file = ranked_file.replace(".csv", "dim_{}.csv".format(args.shadow_dim))
        elif train_test == "target" and args.target_dim != 64:
            ranked_file = ranked_file.replace(".csv", "dim_{}.csv".format(args.target_dim))
    if args.target_emb and not args.training:
        print("Warning! Load target embedding")
        embedding_file = "target_embeddings.csv"

    if args.random_rec > 0 and train_test == "target":
        ranked_file = ranked_file.replace(".csv", f"random_rec-{args.random_rec}.csv")

    if args.random_embedding > 0 and train_test == "target":
        ranked_file = ranked_file.replace(".csv", f"random_embd-{args.random_embedding}.csv")

    print("Load embedding from:" + embeddings_loc + embedding_file)
    print("Load recommendation from:" + ranked_loc + ranked_file)

    df_topk = pd.read_csv(ranked_loc + ranked_file)
    embeddings = pd.read_csv(embeddings_loc + embedding_file).to_numpy()
    len_all_embedding = len(embeddings)
    # if generating testing, embedding us second part only
    if train_test == "target":
        embeddings = embeddings[num_item_pre:]
        len_now = len(embeddings)
        print("The embeddings is cut into {} from {} for MIA-testing generating".format(len_now, len_all_embedding))

    df_topk.columns = ["user"] + ["top_{}".format(i) for i in range(df_topk.shape[1]-1)]

    # load MIA-sets:
    if not args.training:
        sub_name = "test"
    else:
        sub_name = "train({})".format(args.partial_option) if args.partial_option else "train"
    MIA_train_inds = pd.read_csv(shadow_loc  + "MIA-{}.csv".format(sub_name) if args.training else original_loc + "MIA-{}.csv".format(sub_name))

    df_join = pd.merge(MIA_train_inds, df_topk, left_on='user', right_on='user')
    #       join MIA-train with recommendations
    topK_columns = [col_name for col_name in df_join.columns if "top" in col_name]

    res = []
    print("{} records to go!".format(len(df_join)))
    v2 = True
    sample = 0
    if not v2:
        for index, row in tqdm(df_join.iterrows()):
            candidate_item_vec = embeddings[row['item']]
            items_vec = embeddings[row[topK_columns].to_numpy()]
            items_vec = items_vec[:K]
            if row['item'] in row[topK_columns].to_numpy():
                continue
            dist_row = [sim(candidate_item_vec, i_vec) for sim in similarity_list for i_vec in items_vec]
            ui_id = [row["user"], row["item"]]
            label = [row['label']]
            res.append(ui_id + dist_row + label)
            if sample and index >= len(df_join) * sample:
                break
    else:
        #torch.set_num_threads(32)
        # init distance rows
        tensor_emb = torch.Tensor(embeddings)

        can_items = df_join['item'].to_numpy()
        top_items = df_join[topK_columns[:K]].to_numpy()
        ui_id = df_join[["user", "item"]].to_numpy()
        labels = df_join["label"].to_numpy()
        print("Getting L1 matrix")
        L1_m = torch.cdist(tensor_emb, tensor_emb, p=1)
        res_l1 = []
        for i in tqdm(range(len(can_items))):
            row = np.concatenate([ui_id[i],
                                  L1_m[can_items[i], top_items[i]].numpy()])
            res_l1.append(row)
        #del L1_m
        print("BC M plus")
        L1_m_plus = torch.cdist(tensor_emb, -tensor_emb, p=1)
        print("BC M")
        bc_m = L1_m.div(L1_m_plus)
        res_bm = []
        for i in tqdm(range(len(can_items))):
            row = np.concatenate([bc_m[can_items[i], top_items[i]].numpy(),
                                  [labels[i]]])
            res_bm.append(row)
        del bc_m, L1_m, L1_m_plus

        print("Getting L2 matrix")
        L2_m = torch.cdist(tensor_emb, tensor_emb, p=2)
        res_l2 = []
        for i in tqdm(range(len(can_items))):
            row = np.concatenate([L2_m[can_items[i], top_items[i]].numpy()])
            res_l2.append(row)
        del L2_m

        print("Getting cosine matrix")
        cos_m = 1 - pairwise_cosine_similarity(tensor_emb, tensor_emb)
        res_cos = []
        for i in tqdm(range(len(can_items))):
            row = np.concatenate([cos_m[can_items[i], top_items[i]].numpy()])
            res_cos.append(row)
        del cos_m
        time.sleep(15)

        print("Get total inputs")
        res = np.hstack([res_l1, res_l2, res_cos, res_bm])
        print("Finished and delete matrix")

    df_MIA_input = pd.DataFrame(res)
    df_columns = ["user", "item"] + ["{}_{}".format(sim, i) for sim in sim_name_list for i in range(K)] + ['label']
    df_MIA_input.columns = df_columns
    if not args.training:
        if shadow_model != model:
            sub_save_name = "test({})".format(model)
        else:
            sub_save_name = "test"
    else:
        sub_save_name = "train"

    if args.partial_option:
        sub_save_name += "({})".format(args.partial_option)
    file_name = embeddings_loc + "MIA_input_{}.csv".format(sub_save_name)
    if args.target_dim != 64 or args.shadow_dim != 64:
        file_name = file_name.replace(".csv", "{}-{}.csv".format(args.target_dim, args.shadow_dim))
    if "shadow" not in dataset:
        if ("amazon" in dataset) or ("last-fm" in dataset) or ("yelp" in dataset):
            file_name = file_name.replace(shadow_dataset, dataset)
    print("Writing file")
    if args.target_emb and not args.training:
        file_name = file_name.replace(".csv", "(target).csv")
    if args.random_rec > 0 and train_test == "target":
        file_name = file_name.replace(".csv", f"-random_rec-{args.random_rec}.csv")

    if args.random_embedding > 0 and train_test == "target":
        file_name = file_name.replace(".csv", f"-random_embd-{args.random_embedding}.csv")
    df_MIA_input.to_csv(file_name, header=False, index=False)

    print("Done, saved to {}".format(file_name))


