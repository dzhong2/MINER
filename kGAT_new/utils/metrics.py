import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits = hits[hits.sum(axis=1) > 0]
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        if u not in train_user_dict:
            train_user_dict[u] = []
            test_user_dict[u] = []
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall']    = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg']      = ndcg_at_k_batch(binary_hit, k)
    return metrics_dict


def calc_metrics_at_k_torch(cf_scores, batch_tests, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    _, rank_indices = torch.sort(cf_scores, descending=True)
    #rank_indices = rank_indices.cpu()

    for i in range(batch_tests.shape[0]):
        batch_tests[i] = batch_tests[i][rank_indices[i]]
    binary_hit = np.array(batch_tests.cpu(), dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall']    = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg']      = ndcg_at_k_batch(binary_hit, k)
    # max k = 100
    batch_ranks = rank_indices[:, :100].cpu().numpy()
    return metrics_dict, batch_ranks


def get_score_df_batch(cf_scores, train_user_dict, test_user_dict, user_ids):
    train_score_list = []
    train_label_list = []
    test_score_list = []
    test_label_list = []
    train_users = []
    train_items = []
    test_users = []
    test_items = []
    for idx, u in enumerate(user_ids):
        if u not in train_user_dict:
            train_user_dict[u] = []
            test_user_dict[u] = []
            continue
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        if len(train_pos_item_list) == 0 or len(test_pos_item_list) == 0:
            continue
        # train
        labels = np.concatenate([np.ones(len(train_pos_item_list)), np.zeros(len(train_pos_item_list))])
        prob = np.ones(cf_scores.shape[1])
        prob[train_pos_item_list] = 0
        prob[test_pos_item_list] = 0
        prob = prob / prob.sum()
        non_member_ind = np.random.choice(cf_scores.shape[1], len(train_pos_item_list), p=prob)
        scores = cf_scores[idx][np.concatenate([train_pos_item_list, non_member_ind])]
        train_score_list.append(scores)
        train_label_list.append(labels)
        train_items.append(np.concatenate([train_pos_item_list, non_member_ind]))
        train_users.append([idx] * len(scores))

        # test
        labels = np.concatenate([np.ones(len(test_pos_item_list)), np.zeros(len(test_pos_item_list))])
        non_member_ind = np.random.choice(cf_scores.shape[1], len(test_pos_item_list), p=prob)
        scores = cf_scores[idx][np.concatenate([test_pos_item_list, non_member_ind])]
        test_score_list.append(scores)
        test_label_list.append(labels)
        test_items.append(np.concatenate([test_pos_item_list, non_member_ind]))
        test_users.append([idx] * len(scores))

    train_users_all = np.concatenate(train_users)
    train_items_all = np.concatenate(train_items)
    train_score_all = np.concatenate(train_score_list)
    train_label_all = np.concatenate(train_label_list)

    test_users_all = np.concatenate(test_users)
    test_items_all = np.concatenate(test_items)
    test_score_all = np.concatenate(test_score_list)
    test_label_all = np.concatenate(test_label_list)

    df_train = pd.DataFrame(np.array([train_users_all,
                                      train_items_all,
                                      train_score_all,
                                      train_label_all]).T, columns=["user", "item", "score", "label"])

    df_test = pd.DataFrame(np.array([test_users_all,
                                     test_items_all,
                                     test_score_all,
                                     test_label_all]).T, columns=["user", "item", "score", "label"])

    df_train["train"] = 1
    df_test["train"] = 0

    df_all = pd.concat([df_train, df_test])

    return df_all



