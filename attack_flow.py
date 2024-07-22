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

from sklearn.preprocessing import StandardScaler
from attack import attack_k, MIA_plus_k


def prepare_MIA_input(dataset, shadow_dataset, model, shadow_model, training, K,
                      partial_option, kg_only, balance, defense: int=0, def_method='mean'):

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

    num_item_pre = 0

    # load item embeddings and top K recommendations for users
    embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, shadow_dataset)

    ranked_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model if training else model,
                                                          shadow_dataset if training else dataset)
    ranked_file = "{}_recommendations.csv".format('shadow' if training else 'target')
    if not training and defense > 0:
        ranked_file = "{}_recommendations(t={}, {}).csv".format('target', defense, def_method)
    if partial_option and training == 1:
        ranked_file = "{}_recommendations({}).csv".format('shadow', partial_option)
        if balance:
            ranked_file = "{}_recommendations({}).csv".format('shadow', partial_option + 1)
    embedding_file = "shadow_embeddings({}).csv".format(partial_option) if partial_option else "shadow_embeddings.csv"
    if balance and training == 1:
        # ranked_file = "shadow_recommendations(2.0).csv" DO NOT use balanced shadow recommendation
        if partial_option == 1.001:
            embedding_file = "shadow_embeddings(2.0).csv"
        else:
            embedding_file = f"shadow_embeddings({1 + partial_option}).csv"

    if kg_only > 0:
        embedding_file = embedding_file.replace("shadow", "shadow-kg")

    print("Load embedding from:" + embeddings_loc + embedding_file)
    print("Load recommendation from:" + ranked_loc + ranked_file)

    df_topk = pd.read_csv(ranked_loc + ranked_file)
    embeddings = pd.read_csv(embeddings_loc + embedding_file).to_numpy()
    len_all_embedding = len(embeddings)
    # if generating testing, embedding us second part only
    if training == "target":
        embeddings = embeddings[num_item_pre:]
        len_now = len(embeddings)
        print("The embeddings is cut into {} from {} for MIA-testing generating".format(len_now, len_all_embedding))

    df_topk.columns = ["user"] + ["top_{}".format(i) for i in range(df_topk.shape[1]-1)]

    # load MIA-sets:
    if not training:
        sub_name = "test"
    else:
        sub_name = "train({})".format(partial_option if partial_option < 5 else 3.2) if partial_option else "train"
        if balance:
            if partial_option == 1.001:
                sub_name = "train(1.001)"
            elif partial_option >= 5:
                sub_name = "train({})".format(3.2)
            else:
                sub_name = "train({})".format(partial_option)
    mia_train_set = shadow_loc + "MIA-{}.csv".format(sub_name) if training else original_loc + "MIA-{}.csv".format(sub_name)
    MIA_train_inds = pd.read_csv(mia_train_set)
    print("loaded MIA set from {}".format(mia_train_set))
    print(f"loaded {MIA_train_inds['label'].sum()} members and {(1 - MIA_train_inds['label']).sum()} non-members")

    # change labels according to real membership (balanced)
    if balance and training == 1:
        mia_train_set = "KGAT_new/datasets/{}/train_shadow({}).txt".format(dataset,
                                                                        1 + partial_option if partial_option <5 else 4.2)
        print("preparing mia set from {}".format(mia_train_set))
        lines = open(mia_train_set, 'r').readlines()
        dfs = []
        for line in lines:
            content = line.split()
            u = content[0]
            i_list = content[1:]
            df_cf = pd.DataFrame(i_list, columns=['item'])
            df_cf['user'] = u
            df_cf = df_cf.astype(int)
            dfs.append(df_cf)
        df_cf_balanced = pd.concat(dfs, ignore_index=True)
        df_cf_balanced = df_cf_balanced.drop_duplicates()
        MIA_train_inds = pd.merge(MIA_train_inds, df_cf_balanced,
                                  left_on=['user', 'item'], right_on=['user', 'item'],
                                  how='left', indicator=True)
        MIA_train_inds.loc[MIA_train_inds['_merge'] == 'left_only', 'label'] = 0
        MIA_train_inds.drop(columns=['_merge'])
        print(f"After correction: {MIA_train_inds['label'].sum()} members and {(1 - MIA_train_inds['label']).sum()} non-members")

    df_join = pd.merge(MIA_train_inds, df_topk, left_on='user', right_on='user')
    #       join MIA-train with recommendations
    if training == 0:
        hh = 1
        pass
        #print("subsample 10% of testing data")
        #df_join = df_join.sample(frac=0.1, random_state=1, replace=False)
        # split by item frequency and sample with same number
        df_item_freq = df_join.groupby('item').size().reset_index()
        df_item_freq.columns = ['item', 'count']
        try:
            df_item_freq['rank'] = pd.qcut(df_item_freq['count'], 10, labels=range(10))
        except:
            print("Not using qcut")
            df_item_freq = df_item_freq.sort_values('count').copy()
            range_lenth = len(df_item_freq) // 10
            ranks = []
            for i in range(10):
                ranks += [i] * range_lenth
            if len(ranks) < len(df_item_freq):
                ranks += [9] * (len(df_item_freq) - len(ranks))
            df_item_freq['rank'] = ranks

        #num_sample = df_item_freq.groupby('rank').size().min()
        #num_sample = df_item_freq.loc[df_item_freq['rank'] == 0, 'count'].sum()
        df_join = df_join.merge(df_item_freq[['item', 'rank']], on='item', how='left')
        num_sample = df_join.groupby('rank').size().min()
        num_before_sample = len(df_join)
        if hh != 1:
            df_join = df_join.groupby('rank').sample(n=num_sample, replace=False)
        else:
            print("Same 20/80 head tail with evenly split.")
            freqs = [1, 1, 1, 1, 1, 1, 1, 1, 4, 4]
            df_list = []
            for i in range(10):
                df_join_sub = df_join.loc[df_join['rank'] == i].copy()
                df_join_sub = df_join_sub.sample(n=freqs[i] * num_sample, replace=False)
                df_list.append(df_join_sub)
            df_join = pd.concat(df_list, ignore_index=True)
        num_after_sample = len(df_join)
        print(f"sampled {num_after_sample} from {num_before_sample} testing data")
        if num_after_sample > 100000:
            df_join = df_join.sample(frac=0.1, random_state=1, replace=False)
            print(f"subsample 10% of {num_after_sample} testing data")
    topK_columns = [col_name for col_name in df_join.columns if "top" in col_name]

    res = []
    print("{} records to go!".format(len(df_join)))

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

    print("Finished generation")
    if training:
        return df_MIA_input
    else:
        return df_MIA_input, df_item_freq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon-book-shadow")
    parser.add_argument("--shadow_dataset", type=str, default="am-la")
    parser.add_argument("--model", type=str, default="NFM")
    parser.add_argument("--shadow_model", type=str, default="NFM")
    parser.add_argument("--training", type=int, default=1)
    parser.add_argument("--partial_option", type=float, default=0)
    parser.add_argument("--random_rec", type=float, default=0)
    parser.add_argument("--random_embedding", type=float, default=0)
    parser.add_argument("--target_emb", type=int, default=0)
    parser.add_argument("--target_dim", type=int, default=64)
    parser.add_argument("--shadow_dim", type=int, default=64)
    parser.add_argument("--kg_only", type=int, default=0)
    parser.add_argument("--gpu_ind", type=int, default=0)
    parser.add_argument("--balance", type=bool, default=True)
    parser.add_argument('--Ks', type=int, nargs='+')
    parser.add_argument('--gammas', type=int, nargs='+')
    parser.add_argument('--epochs', type=int, nargs='+')
    parser.add_argument("--weight_adjust", type=float, default=0)
    parser.add_argument("--defense", type=int, default=0)
    parser.add_argument("--def_method", type=str, default='mean')
    args = parser.parse_args()

    dataset = args.dataset
    shadow_dataset = args.shadow_dataset
    shadow_model = args.shadow_model
    model = args.model
    K = 100


    embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, shadow_dataset)

    MIA_input_train_B = prepare_MIA_input(dataset, shadow_dataset, model, shadow_model, 1, K,
                                          args.partial_option, args.kg_only, balance=True)
    MIA_input_train_L = prepare_MIA_input(dataset, shadow_dataset, model, shadow_model, 1, K,
                                          args.partial_option, args.kg_only, balance=False)
    MIA_input_test, item_freq_rank = prepare_MIA_input(dataset, shadow_dataset, model, shadow_model, 0, K,
                                        args.partial_option, args.kg_only, balance=False,
                                                       defense=args.defense, def_method=args.def_method)
    '''common_inds = MIA_input_train_B['label'] == MIA_input_train_L['label']
    diff_inds = MIA_input_train_B['label'] != MIA_input_train_L['label']
    mem_b_number = int(MIA_input_train_B.loc[common_inds, 'label'].sum())
    MIA_input_train_B = pd.concat([MIA_input_train_B.loc[common_inds].groupby('label').sample(n=mem_b_number, replace=False),
                                   MIA_input_train_B.loc[diff_inds]])
    MIA_input_train_L = MIA_input_train_L.loc[MIA_input_train_B.index]'''

    MIA_input_train_B = MIA_input_train_B.values
    MIA_input_train_L = MIA_input_train_L.values
    MIA_input_test = MIA_input_test.values

    ss = StandardScaler()
    MIA_input_train_B[:, 2:-1] = ss.fit_transform(MIA_input_train_B[:, 2:-1])
    MIA_input_train_L[:, 2:-1] = ss.fit_transform(MIA_input_train_L[:, 2:-1])
    MIA_input_test[:, 2:-1] = ss.fit_transform(MIA_input_test[:, 2:-1])

    device = torch.device("cuda:{}".format(args.gpu_ind) if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    metrics_list = []
    pop_res_list = []
    print("Applying multiple K attack")
    #for gamma in [1.0, 2.0, 3.0, 4.0, 5.0]:
    for gamma in [int(x) for x in args.gammas]:
        for epoch in args.epochs:
            for K in args.Ks:

                for t in range(5):
                    metric, pop_res = MIA_plus_k(MIA_input_train_L, MIA_input_train_B, MIA_input_test,
                                                 device=device, gamma=gamma, epoch=epoch, topK=K,
                                                 weight_adjust=args.weight_adjust, testing_rank=item_freq_rank)
                    pop_res = [[gamma, epoch] + x for x in pop_res]
                    pop_res_list += pop_res
                    metrics_list.append([gamma, epoch, K] + metric)
                df_metric = pd.DataFrame(metrics_list,
                                         columns=["gamma", "epoch", "Top K", "Accuracy", "Precesion", "Recall", "AUC",
                                                  "F1"])
                save_loc = embeddings_loc + "MIA_results({}-{}).csv".format(model, shadow_model)
                if args.partial_option:
                    save_loc = save_loc.replace(".csv", "({}).csv".format(args.partial_option))
                if args.balance:
                    save_loc = save_loc.replace(".csv", "(plus-gamma).csv")
                if args.weight_adjust > 0:
                    save_loc = save_loc.replace(".csv", f"(wa={args.weight_adjust}).csv")
                if args.defense > 0:
                    save_loc = save_loc.replace(".csv", f"(t={args.defense}, {args.def_method}).csv")
                df_metric.to_csv(save_loc)
                print("saved to " + save_loc)

                df_pop = pd.DataFrame(pop_res_list,
                                      columns=['gamma', "epoch", 'Top K popular', "Accuracy", "Precesion", "Recall",
                                               "AUC",
                                               "F1", "K", "Category"])
                df_pop = df_pop.groupby(['gamma', "epoch", "Top K popular", "K", "Category"]).mean()
                df_pop.to_csv(save_loc.replace("MIA_results", "Popular_results"))
                print("saved to " + save_loc.replace("MIA_results", "Popular_results"))