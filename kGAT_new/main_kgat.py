import os
import sys
import random
from time import time

import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    ranks = []
    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)
            #batch_scores = batch_scores.cpu()
            #batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict,
            #                                  batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
            batch_trains = torch.ones(batch_scores.shape)
            batch_tests = torch.zeros(batch_scores.shape)
            for i in range(len(batch_user_ids)):
                uid = batch_user_ids[i].cpu().item()
                if uid in train_user_dict:
                    batch_trains[i][train_user_dict[uid]] = - torch.inf
                    batch_tests[i][test_user_dict[uid]] = 1
            batch_trains = batch_trains.to(device)
            batch_tests = batch_tests.to(device)

            batch_scores = batch_scores * batch_trains
            batch_metrics, batch_ranks = calc_metrics_at_k_torch(batch_scores, batch_tests, Ks)

            cf_scores.append(batch_scores.cpu().numpy())
            ranks.append(batch_ranks)
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    ranks = np.concatenate(ranks, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return metrics_dict, ranks


def get_score_df(model, dataloader, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    df_all = []
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)
            batch_scores = batch_scores.cpu()
            df_batch = get_score_df_batch(batch_scores, train_user_dict, test_user_dict,
                                    batch_user_ids.cpu().numpy(), item_ids.cpu().numpy())
            df_all.append(df_batch)
            pbar.update(1)
    print("{} Dataframes in total".format(len(df_all)))
    df = pd.concat(df_all)
    return df


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda:{}".format(args.gpu_ind) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and args.gpu_ind > torch.cuda.device_count() - 1:
        device = torch.device("cuda:0")
    print("Using device: {}".format(device))

    # load data
    data = DataLoaderKGAT(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            embedding_item = model.entity_user_embed.weight.detach()[: data.n_items]
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')
            if args.www > 0:
                embedding_item_new = model.entity_user_embed.weight.detach()[: data.n_items]
                embedding_loss = (embedding_item - embedding_item_new).abs().sum() * args.www
                kg_batch_loss += embedding_loss.to(device)

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        #model(h_list, t_list, r_list, relations, mode='update_att')
        with torch.no_grad():
            model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            metrics_dict, _ = evaluate(model, data, Ks, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.csv', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))
    save_best_model(model, args.save_dir, best_epoch)


def predict(args, data=None):
    # GPU / CPU
    device = torch.device("cuda:{}".format(args.gpu_ind) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and args.gpu_ind > torch.cuda.device_count() - 1:
        device = torch.device("cuda:0")
    print("Using device: {}".format(device))

    # load data
    if data is None:
        print("Load data because data is None")
        data = DataLoaderKGAT(args, logging)
    else:
        print("Use data from input (training is run before)")

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    #cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    metrics_dict, top_items = evaluate(model, data, Ks, device)
    #np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
    metrics_df = []
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append([k, m, metrics_dict[k][m]])
    metrics_df = pd.DataFrame(metrics_df, columns=['K', 'metric', 'value'])
    metric_df_name = '/metrics_final.csv'

    if args.random_rec > 0:
        metric_df_name = metric_df_name.replace(".csv", f"-random_rec-{args.random_rec}.csv")
    if args.random_embedding > 0:
        metric_df_name = metric_df_name.replace(".csv", f"-random_embd-{args.random_embedding}.csv")
    metrics_df.to_csv(args.save_dir + metric_df_name, index=False)

    #ranked_ind = (-cf_scores).argsort(axis=1)[:, :args.topK]

    df = pd.DataFrame(top_items)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'user'})
    rec_name = args.save_dir + '{}_recommendations.csv'.format(args.data_name)
    if args.topK > 100:
        rec_name = rec_name.replace(".csv", "({}).csv".format(args.topK))
    if args.random_rec > 0:
        rec_name = rec_name.replace(".csv", f"random_rec-{args.random_rec}.csv")
    if args.random_embedding > 0:
        rec_name = rec_name.replace(".csv", f"random_embd-{args.random_embedding}.csv")

    df.to_csv(rec_name,
              index=False)
    print("Recommendation saved to:" + rec_name)

    if args.train_shadow:
        sub_name = "shadow"
    else:
        sub_name = "target"
    if args.partial_option:
        tail_name = "({}).csv".format(args.partial_option)
    else:
        tail_name = ".csv"

    if args.embed_dim != 64:
        tail_name = tail_name.replace(".csv", "dim_{}.csv".format(args.embed_dim))
    rec_name = args.save_dir + '../{}_recommendations{}'.format(sub_name, tail_name)
    if args.topK > 100:
        rec_name = rec_name.replace(".csv", "({}).csv".format(args.topK))
    if args.random_rec > 0:
        rec_name = rec_name.replace(".csv", f"random_rec-{args.random_rec}.csv")
    if args.random_embedding > 0:
        rec_name = rec_name.replace(".csv", f"random_embd-{args.random_embedding}.csv")

    df.to_csv(rec_name, index=False)
    print("Recommendation saved to:" + rec_name)

    embeddings_all = model.entity_user_embed.weight.detach().cpu().numpy()
    embeddings_items = embeddings_all[: data.n_items]
    df_item_embd = pd.DataFrame(embeddings_items)
    embedding_name = args.save_dir + '{}_embeddings.csv'.format(args.data_name)
    if args.topK > 100:
        embedding_name = embedding_name.replace(".csv", "({}).csv".format(args.topK))
    df_item_embd.to_csv(embedding_name,
                        index=False)

    print("embedding saved to:" + embedding_name)

    embedding_name = args.save_dir + '../{}_embeddings{}'.format(sub_name, tail_name)
    if args.topK > 100:
        embedding_name = embedding_name.replace(".csv", "({}).csv".format(args.topK))
    df_item_embd.to_csv(embedding_name,
                        index=False)
    print("embedding saved to:" + embedding_name)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'],
        metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))


def get_auc(args):
    # GPU / CPU

    device = torch.device("cuda:{}".format(args.gpu_ind) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and args.gpu_ind > torch.cuda.device_count() - 1:
        device = torch.device("cuda:0")
    print("Using device: {}".format(device))

    # load data
    data = DataLoaderKGAT(args, logging)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    df = get_score_df(model, data, device)
    score_name = args.save_dir + '../target_score.csv'
    if args.topK > 100:
        score_name = score_name.replace(".csv", "({}).csv".format(args.topK))

    df.to_csv(score_name, index=False)
    print("Score saved to:" + score_name)



if __name__ == '__main__':
    args = parse_kgat_args()
    if args.get_score_only:
        get_auc(args)
        exit(0)
    if not args.get_output_only:
        train(args)

    predict(args)

