import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os
import shutil


class MLP(nn.Module):
    def __init__(self, num_latent):
        self.nl = num_latent
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_latent, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda:0")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)

        return loss.cpu().detach().item(), acc.cpu().detach().item()


    def all_metrics(self, X_target, y_target, verbos=True):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        if verbos:
            print("Accuracy = {:.2%}\n Precision = {:.2%} \n Recall = {:.2%}\n AUC = {:.4}\n F1{:.4}".format(acc_target,
                                                                                               prec_target,
                                                                                               recall_target,
                                                                                               auc_target,
                                                                                                             f1_target))
        return [acc_target, prec_target, recall_target, auc_target, f1_target]


class KMIA_plus(nn.Module):
    def __init__(self, num_latent):
        self.nl = num_latent
        super(KMIA_plus, self).__init__()
        self.fc11 = nn.Linear(num_latent, 64)
        self.fc12 = nn.Linear(64, 64)
        self.fc13 = nn.Linear(64, 2) # for MSL
        self.fc21 = nn.Linear(num_latent, 64)
        self.fc22 = nn.Linear(64, 64)# for MSB
        self.fc23 = nn.Linear(64, 2)
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda:0")

    def forward(self, x1, x2, at):
        x1 = F.relu(self.fc11(x1))
        x1 = F.relu(self.fc12(x1))
        x1 = torch.sigmoid(self.fc13(x1))
        x2 = F.relu(self.fc21(x2))
        x2 = F.relu(self.fc22(x2))
        x2 = torch.sigmoid(self.fc23(x2))

        x = at * x1 + (1 - at) * x2
        x = F.softmax(x, dim=1)
        return x

    def output_test(self, x):
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.softmax(self.fc13(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain1, ytrain1, Xtrain2, ytrain2, at):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain1).to(self.device),
                       torch.Tensor(Xtrain2).to(self.device),
                       at)
        loss = at * self.criterion(outputs, torch.LongTensor(ytrain1).to(self.device)) + (1 - at) * self.criterion(outputs, torch.LongTensor(ytrain2).to(self.device))
        loss.backward()
        self.optimizer.step()

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self.output_test(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)

        return loss.cpu().detach().item(), acc.cpu().detach().item()


    def all_metrics(self, X_target, y_target, verbos=True):
        outputs_target = self.output_test(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        if verbos:
            print("Accuracy = {:.2%}\n Precision = {:.2%} \n Recall = {:.2%}\n AUC = {:.4}\n F1{:.4}".format(acc_target,
                                                                                               prec_target,
                                                                                               recall_target,
                                                                                               auc_target,
                                                                                                             f1_target))
        return [acc_target, prec_target, recall_target, auc_target, f1_target]


def mia_train(mia_X, mia_y, device=torch.device("cuda:0"), lr=0.1):
    Xtrain, Xval, ytrain, yval = train_test_split(mia_X, mia_y, test_size=0.2)
    # train MIA
    mlp = MLP(Xval.shape[1])
    mlp.to(device)
    mlp.device = device

    mlp.criterion = torch.nn.CrossEntropyLoss()
    mlp.optimizer = torch.optim.SGD(mlp.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    pbar = tqdm(range(2000))
    opt_loss = 1e10
    patient = 50
    for i in pbar:
        mlp.train_one_epoch(Xtrain, ytrain)

        train_loss, train_acc = mlp.loss_acc(Xtrain, ytrain)
        val_loss, val_acc = mlp.loss_acc(Xval, yval)

        pbar.set_postfix({'Loss': train_loss,
                          'Acc': train_acc,
                          'Val Loss': val_loss,
                          'Val Acc': val_acc})
        if opt_loss / 1.001 > val_loss:
            opt_loss = val_loss
            patient = 50
        else:
            patient = patient - 1

        if patient == 0:
            pbar.close()
            print("Early break at epoch {}".format(i))

            break
    print("Training End")
    return mlp


def MIA_plus_train(mia_X1, mia_X2, mia_y1, mia_y2, device=torch.device("cuda:0"), lr=0.1, gamma=1.0, epoch=500):
    inds = np.arange(len(mia_X1))
    Xtrain1, Xval1, ytrain1, yval1, indtrain, indval = train_test_split(mia_X1, mia_y1, inds, test_size=0.2)
    Xtrain2 = mia_X2[indtrain]
    Xval2 = mia_X2[indval]
    ytrain2 = mia_y2[indtrain]
    yval2 = mia_y2[indval]


    # train MIA
    mlp = KMIA_plus(Xtrain1.shape[1])
    mlp.to(device)
    mlp.device = device

    mlp.criterion = torch.nn.CrossEntropyLoss()
    mlp.optimizer = torch.optim.SGD(mlp.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    pbar = tqdm(range(epoch))
    for i in pbar:
        at = 1 - ((i + 1)/epoch/gamma) ** 2
        mlp.train_one_epoch(Xtrain1, ytrain1, Xtrain2, ytrain2, at)

        train_loss1, train_acc1 = mlp.loss_acc(Xtrain1, ytrain1)
        train_loss2, train_acc2 = mlp.loss_acc(Xtrain2, ytrain2)
        val_loss1, val_acc1 = mlp.loss_acc(Xval1, yval1)
        val_loss2, val_acc2 = mlp.loss_acc(Xval2, yval2)

        pbar.set_postfix({'Loss1': train_loss1,
                          'Acc1': train_acc1,
                          'Loss2': train_loss2,
                          'Acc2': train_acc2,
                          'Val Loss1': val_loss1,
                          'Val Acc1': val_acc1,
                          'Val Loss2': val_loss2,
                          'Val Acc2': val_acc2})
    print("Training End")
    return mlp


def test_popular_item(mlp, mia_X_test, mia_y_test, items, K, gamma=-1):
    pop_list = [0.1, 0.2, 0.3, 0.4]
    item_set, item_count = np.unique(items, return_counts=True)
    sorted_item = item_set[item_count.argsort()[::-1]]
    item_size = len(sorted_item)
    res = []
    for pop in pop_list:
        pop_item_number = int(item_size * pop)
        pop_item_start = max(int(item_size * (pop - 0.1)), 0)
        pop_items = sorted_item[pop_item_start: pop_item_number]
        print("{} items out of {} in totall".format(pop_item_number, item_size))
        test_cases_ind = [i for i in range(len(items)) if items[i] in pop_items]

        mia_X_test_sub = mia_X_test[test_cases_ind]
        mia_y_test_sub = mia_y_test[test_cases_ind]
        metrics = mlp.all_metrics(mia_X_test_sub, mia_y_test_sub)
        res.append([pop] + metrics + [K, "popular Item"])

    return res


def test_popular_item_v2(mlp, mia_X_test, mia_y_test, items, K, testing_rank=None):
    pop_list = [0.1, 0.2, 0.3, 0.4]
    res = []
    for i in range(4):
        pop_str = pop_list[i]
        pop_items = set(testing_rank.loc[testing_rank['rank'] == 9-i, 'item'].values)
        pop_size = len(pop_items)
        item_size = testing_rank['item'].nunique()
        print("{} items out of {} in total".format(pop_size, item_size))
        test_cases_ind = [i for i in range(len(items)) if items[i] in pop_items]

        mia_X_test_sub = mia_X_test[test_cases_ind]
        mia_y_test_sub = mia_y_test[test_cases_ind]
        metrics = mlp.all_metrics(mia_X_test_sub, mia_y_test_sub)
        res.append([pop_str] + metrics + [K, "popular Item"])

    return res


def test_unpopular_item(mlp, mia_X_test, mia_y_test, items, K):
    pop_list = [0.1, 0.2, 0.3, 0.4]
    item_set, item_count = np.unique(items, return_counts=True)
    sorted_item = item_set[item_count.argsort()]
    item_size = len(sorted_item)
    res = []
    for pop in pop_list:
        pop_item_number = int(item_size * pop)
        pop_item_start = max(int(item_size * (pop - 0.1)), 0)
        pop_items = sorted_item[pop_item_start: pop_item_number]
        print("{} items out of {} in total".format(pop_item_number, item_size))
        test_cases_ind = [i for i in range(len(items)) if items[i] in pop_items]

        mia_X_test_sub = mia_X_test[test_cases_ind]
        mia_y_test_sub = mia_y_test[test_cases_ind]
        metrics = mlp.all_metrics(mia_X_test_sub, mia_y_test_sub)
        res.append([pop] + metrics + [K, "Unpopular Item"])

    return res


def test_unpopular_item_v2(mlp, mia_X_test, mia_y_test, items, K, testing_rank=None):
    pop_list = [0.1, 0.2, 0.3, 0.4]
    res = []
    for i in range(4):
        pop_str = pop_list[i]
        pop_items = set(testing_rank.loc[testing_rank['rank'] == i, 'item'].values)
        pop_size = len(pop_items)
        item_size = testing_rank['item'].nunique()
        print("{} items out of {} in total".format(pop_size, item_size))
        test_cases_ind = [i for i in range(len(items)) if items[i] in pop_items]

        mia_X_test_sub = mia_X_test[test_cases_ind]
        mia_y_test_sub = mia_y_test[test_cases_ind]
        metrics = mlp.all_metrics(mia_X_test_sub, mia_y_test_sub)
        res.append([pop_str] + metrics + [K, "Unpopular Item"])

    return res


def test_inactive_user(mlp, mia_X_test, mia_y_test, users, K):
    pop_list = [0.1, 0.2, 0.3, 0.4]
    user_set, user_count = np.unique(users, return_counts=True)
    sorted_user = user_set[user_count.argsort()]
    user_size = len(sorted_user)
    res = []
    for pop in pop_list:
        pop_user_number = int(user_size * pop)
        pop_user_start = int(user_size * (pop - 0.1))
        pop_users = sorted_user[pop_user_start: pop_user_number]
        print("{} items out of {} in totall".format(pop_user_number, user_size))
        test_cases_ind = [i for i in range(len(users)) if users[i] in pop_users]

        mia_X_test_sub = mia_X_test[test_cases_ind]
        mia_y_test_sub = mia_y_test[test_cases_ind]
        metrics = mlp.all_metrics(mia_X_test_sub, mia_y_test_sub)
        res.append([pop] + metrics + [K, "Inactive User"])

    return res

def test_active_user(mlp, mia_X_test, mia_y_test, users, K):
    pop_list = [0.1, 0.2, 0.3, 0.4]
    user_set, user_count = np.unique(users, return_counts=True)
    sorted_user = user_set[user_count.argsort()[::-1]]
    user_size = len(sorted_user)
    res = []
    for pop in pop_list:
        pop_user_number = int(user_size * pop)
        pop_user_start = int(user_size * (pop - 0.1))
        pop_users = sorted_user[pop_user_start: pop_user_number]
        print("{} items out of {} in totall".format(pop_user_number, user_size))
        test_cases_ind = [i for i in range(len(users)) if users[i] in pop_users]

        mia_X_test_sub = mia_X_test[test_cases_ind]
        mia_y_test_sub = mia_y_test[test_cases_ind]
        metrics = mlp.all_metrics(mia_X_test_sub, mia_y_test_sub)
        res.append([pop] + metrics + [K, "Active User"])

    return res


def judge_popular_item(predict, mia_y, items):
    pop_list = [0.5, 0.1, 0.2, 0.3, 0.4]
    item_set, item_count = np.unique(items, return_counts=True)
    sorted_item = item_set[item_count.argsort()[::-1]]
    item_size = len(sorted_item)
    res = []
    for pop in pop_list:
        pop_item_number = int(item_size * pop)
        pop_items = sorted_item[: pop_item_number]
        print("{} items out of {} in totall".format(pop_item_number, item_size))
        test_cases_ind = [i for i in range(len(items)) if items[i] in pop_items]

        predict_sub = predict[test_cases_ind]
        y_sub = mia_y[test_cases_ind]
        tmp_pop = [metrics.accuracy_score(y_sub, predict_sub),
                   metrics.precision_score(y_sub, predict_sub),
                   metrics.recall_score(y_sub, predict_sub),
                   metrics.f1_score(y_sub, predict_sub)]
        res.append([pop] + tmp_pop)

    return np.array(res)


def attack_k(array_train, array_test, topK, total_k=100, device=torch.device("cuda")):
    sim_list = list(range(4))
    select_columns = [2 + i + sim * total_k for i in range(topK) for sim in sim_list]
    select_columns = [0, 1] + select_columns + [array_train.shape[1] - 1]
    array_train_tmp = array_train[:, select_columns]
    array_test_tmp = array_test[:, select_columns]
    print("Loading {} MIA training interactions and {} MIA testing interactions".format(len(array_train),
                                                                                        len(array_test)))
    print("Total input of X is {} dimension".format(array_train_tmp[:, 2:-1].shape[1]))
    mia_x = array_train_tmp[:, 2:-1]
    mia_y = array_train_tmp[:, -1]

    mlp = mia_train(mia_x, mia_y, device, lr=0.1)

    mia_X_test = array_test_tmp[:, 2:-1]
    mia_y_test = array_test_tmp[:, -1]
    metrics = mlp.all_metrics(mia_X_test, mia_y_test)

    if topK in [20, 40, 60, 80, 100]:
        items = array_test_tmp[:, 1]
        users = array_test_tmp[:, 0]
        pop_res = test_popular_item(mlp, mia_X_test, mia_y_test, items, topK)
        unpop_res = test_unpopular_item(mlp, mia_X_test, mia_y_test, items, topK)
        pop_res_u = test_active_user(mlp, mia_X_test, mia_y_test, users, topK)
        unpop_res_u = test_inactive_user(mlp, mia_X_test, mia_y_test, users, topK)
        #print("{} items in pop_res".format(len(pop_res[0])))
        pop_res = pop_res + unpop_res + pop_res_u + unpop_res_u
    else:
        pop_res = []
    return metrics, pop_res


def MIA_plus_k(array_trainL, array_trainB, array_test, device=torch.device("cuda"), gamma=1.0,
               epoch=500, topK=100, weight_adjust=0.1, testing_rank=None):
    total_k = 100
    sim_list = list(range(4))
    select_columns = [2 + i + sim * total_k for i in range(topK) for sim in sim_list]
    select_columns = [0, 1] + select_columns + [array_trainL.shape[1] - 1]
    array_train_tmpL = array_trainL[:, select_columns]
    array_train_tmpB = array_trainB[:, select_columns]
    array_test_tmp = array_test[:, select_columns]
    print("Loading {} MIA training interactions (L), {} MIA training interactions (B) and {} MIA testing interactions".format(len(array_train_tmpL),
                                                                                        len(array_train_tmpB),
                                                                                        len(array_test)))
    print("Total input of X is {} dimension".format(array_train_tmpL[:, 2:-1].shape[1]))
    mia_x1 = array_train_tmpL[:, 2:-1]
    mia_x2 = array_train_tmpB[:, 2:-1]
    mia_y1 = array_train_tmpL[:, -1]
    mia_y2 = array_train_tmpB[:, -1]
    if weight_adjust == 1:
        input_shape = mia_x2.shape[1]
        #weights = 1 - weight_adjust * np.arange(-input_shape / 2, input_shape / 2) / input_shape * 2
        weights = 1 / (np.concatenate([np.arange(mia_x2.shape[1]/4) for _ in range(4)]) + 1)
        mia_x1 = mia_x1 * weights
        mia_x2 = mia_x2 * weights
    elif weight_adjust == 2:
        weights = 1 / np.log2(np.concatenate([np.arange(mia_x2.shape[1]/4) for _ in range(4)]) + 2)
        mia_x1 = mia_x1 * weights
        mia_x2 = mia_x2 * weights

    mlp = MIA_plus_train(mia_x1, mia_x2, mia_y1, mia_y2, device, lr=0.05, gamma=gamma, epoch=epoch)

    mia_X_test = array_test_tmp[:, 2:-1]
    mia_y_test = array_test_tmp[:, -1]
    if weight_adjust > 0:
        mia_X_test = weights * mia_X_test
    metrics = mlp.all_metrics(mia_X_test, mia_y_test)

    items = array_test_tmp[:, 1]
    users = array_test_tmp[:, 0]
    pop_res = test_popular_item_v2(mlp, mia_X_test, mia_y_test, items, topK, testing_rank=testing_rank)
    unpop_res = test_unpopular_item_v2(mlp, mia_X_test, mia_y_test, items, topK, testing_rank=testing_rank)
    #pop_res_u = test_active_user(mlp, mia_X_test, mia_y_test, users, topK)
    #unpop_res_u = test_inactive_user(mlp, mia_X_test, mia_y_test, users, topK)
    #print("{} items in pop_res".format(len(pop_res[0])))
    pop_res = pop_res + unpop_res# + pop_res_u + unpop_res_u
    return metrics, pop_res


def attack_average_k(array_train, array_test, topK, total_k=100, device=torch.device("cuda")):
    sim_list = list(range(4))
    select_columns = [2 + i + sim * total_k for i in range(topK) for sim in sim_list]
    select_columns = [0, 1] + select_columns + [array_train.shape[1] - 1]
    array_train_tmp = array_train[:, select_columns]
    array_test_tmp = array_test[:, select_columns]
    print("Loading {} MIA training interactions and {} MIA testing interactions".format(len(array_train),
                                                                                        len(array_test)))
    print("averaging")

    train_arr_list = [array_train_tmp[:, :2]]
    for i in range(4):
        train_arr_list.append(array_train_tmp[:, 2 + i * topK: 2 + (i+1) * topK].mean(axis=1).reshape(-1, 1))
    train_arr_list.append(array_train_tmp[:, -1].reshape(-1, 1))

    train_arr = np.hstack(train_arr_list)

    test_arr_list = [array_test_tmp[:, :2]]
    for i in range(4):
        test_arr_list.append(array_test_tmp[:, 2 + i * topK: 2 + (i + 1) * topK].mean(axis=1).reshape(-1, 1))
    test_arr_list.append(array_test_tmp[:, -1].reshape(-1, 1))

    test_arr = np.hstack(test_arr_list)

    print("Total input of X is {} dimension".format(train_arr[:, 2:-1].shape[1]))
    ss = StandardScaler()
    mia_x = ss.fit_transform(train_arr[:, 2:-1])
    mia_y = train_arr[:, -1]

    mlp = mia_train(mia_x, mia_y, device, lr=args.lr)

    mia_X_test = ss.fit_transform(test_arr[:, 2:-1])
    mia_y_test = test_arr[:, -1]
    metrics = mlp.all_metrics(mia_X_test, mia_y_test)

    if topK == 20:
        items = test_arr[:, 1]
        pop_res = test_popular_item(mlp, mia_X_test, mia_y_test, items, topK)
        print("{} items in pop_res".format(len(pop_res[0])))
    else:
        pop_res = []
    return metrics, pop_res


def search_best_threshold(input_x, input_y):
    mint = input_x.min()
    maxt = input_x.max()
    best_performance = 0
    best_choice = 1
    best_t = -1
    for i in range(100):
        tmp_t = i * (maxt - mint)/100 + mint
        performances = judge_with_threshold(input_x, input_y, tmp_t)
        if performances[0] > best_performance:
            best_performance = performances[0]
            best_choice = 0
            best_t = tmp_t
        if performances[1] > best_performance:
            best_performance = performances[1]
            best_choice = 1
            best_t = tmp_t
    return best_t, best_choice, best_performance


def judge_with_threshold(input_x, input_y, t):
    pos_loss = abs((input_x > t) * 1 - input_y).sum() / len(input_x)
    neg_loss = abs((input_x < t) * 1 - input_y).sum() / len(input_x)
    return [1 - pos_loss, 1 - neg_loss]


def threshold_attack(array_train, array_test, topK, total_k=100, device=torch.device("cuda")):
    sim_list = list(range(4))
    select_columns = [2 + i + sim * total_k for i in range(topK) for sim in sim_list]
    select_columns = [0, 1] + select_columns + [array_train.shape[1] - 1]
    array_train_tmp = array_train[:, select_columns]
    array_test_tmp = array_test[:, select_columns]
    print("Loading {} MIA training interactions and {} MIA testing interactions".format(len(array_train),
                                                                                        len(array_test)))
    print("Total input of X is {} dimension".format(array_train_tmp[:, 2:-1].shape[1]))
    ss = StandardScaler()
    mia_x = ss.fit_transform(array_train_tmp[:, 2:-1])
    mia_y = array_train_tmp[:, -1]
    bt, bc, bp, bi = 0,0,0, 0

    for i in tqdm(range(mia_x.shape[1])):
        input_x = mia_x[:, i]
        input_y = mia_y
        best_t, best_choice, best_performance = search_best_threshold(input_x, input_y)
        if best_performance > bp:
            bt, bc, bp = best_t, best_choice, best_performance
            bi = i
    print("best_t, best_choice, best_performance is {} {} {} for column {}".format(bt, bc, bp, bi))


    mia_X_test = ss.fit_transform(array_test_tmp[:, 2:-1])
    mia_y_test = array_test_tmp[:, -1]
    # inference
    items = array_test_tmp[:, 1]
    input_x_test = mia_X_test[:, bi]
    ps = judge_with_threshold(input_x_test, mia_y_test, bt)
    predict = [(input_x_test > bt) * 1, (input_x_test < bt) * 1][bc]
    performance = ps[bc]
    res = judge_popular_item(predict, mia_y_test, items)
    df = pd.DataFrame(res, columns=["Accuracy","Precision", "Recall", "F1", "Popularity"])
    return df



def MIA_input_loader(attack_type, target_data, shadow_data, target_model, shadow_model, random_rec=0,
                                               random_embedding=0):
    if attack_type == 1: # Partial only, shadow_data == target_data
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
        lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, target_data)
        if not os.path.exists(lab_loc):
            os.makedirs(lab_loc)
            print("Init lab loc: " + lab_loc)
        train_sub_name = "MIA_input_train.csv" if not args.partial_option else "MIA_input_train({}).csv".format(
            args.partial_option)
        test_sub_name = "MIA_input_test.csv" if not args.partial_option else "MIA_input_test({}).csv".format(
            args.partial_option)

        if random_rec > 0:
            test_sub_name = test_sub_name.replace(".csv", f"-random_rec-{args.random_rec}.csv")

        if random_embedding > 0:
            test_sub_name = test_sub_name.replace(".csv", f"-random_embd-{args.random_embedding}.csv")

        if os.path.exists(embeddings_loc + train_sub_name):
            print("Moving {}\n"
                  "    into {}".format(embeddings_loc + train_sub_name, lab_loc + train_sub_name)
                  )
            shutil.copyfile(embeddings_loc + train_sub_name,
                            lab_loc + train_sub_name)
            # check and delete source
            if os.path.exists(lab_loc + train_sub_name):
                os.remove(embeddings_loc + train_sub_name)
                print("removed " + embeddings_loc + train_sub_name)
        else:
            print("Skip moving file from " + embeddings_loc + train_sub_name)

        print("loading training from:" + lab_loc + train_sub_name)
        array_train = pd.read_csv(lab_loc + train_sub_name, header=None).to_numpy()
        if model != shadow_model:
            test_sub_name = test_sub_name.replace("test", "test({})".format(model))

        if os.path.exists(embeddings_loc + test_sub_name):
            print("Moving {}\n"
                  "    into {}".format(embeddings_loc + test_sub_name, lab_loc + test_sub_name)
                  )
            shutil.copyfile(embeddings_loc + test_sub_name,
                            lab_loc + test_sub_name)
            # check and delete source
            if os.path.exists(lab_loc + test_sub_name):
                os.remove(embeddings_loc + test_sub_name)
                print("removed " + embeddings_loc + test_sub_name)
        else:
            print("Skip moving file from " + embeddings_loc + test_sub_name)

        testing_file = lab_loc + test_sub_name
        print("loading testing from:" + testing_file)
        array_test = pd.read_csv(testing_file, header=None).dropna().to_numpy()

    elif attack_type == 2: # Shadow only, shadow_data == target_data
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, shadow_data)
        lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, shadow_data)
        if not os.path.exists(lab_loc):
            os.makedirs(lab_loc)
            print("Init lab loc: " + lab_loc)
        train_sub_name = "MIA_input_train(0.001).csv"
        if args.target_dim != 64 or args.shadow_dim != 64:
            assert shadow_data == target_data
            train_sub_name = train_sub_name.replace(".csv", "{}-{}.csv".format(args.target_dim, args.shadow_dim))
        if os.path.exists(embeddings_loc + train_sub_name):
            print("Moving {}\n"
                  "    into {}".format(embeddings_loc + train_sub_name, lab_loc + train_sub_name)
                  )
            shutil.copyfile(embeddings_loc + train_sub_name,
                            lab_loc + train_sub_name)
            # check and delete source
            if os.path.exists(lab_loc + train_sub_name):
                os.remove(embeddings_loc + train_sub_name)
                print("removed " + embeddings_loc + train_sub_name)
        else:
            print("Skip moving file from " + embeddings_loc + train_sub_name)

        print("loading training from:" + lab_loc + train_sub_name)
        array_train = pd.read_csv(lab_loc + train_sub_name, header=None).to_numpy()

        embeddings_loc = "./KGAT_new/trained_model/{}/{}-{}/".format(shadow_model, target_data[:2], target_model)
        lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}-{}/".format(shadow_model, target_data[:2], target_model)
        if target_data == shadow_data:
            embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
            lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, target_data)
        if "shadow" not in target_data:
            if ("amazon" in target_data) or ("last-fm" in target_data) or ("yelp" in target_data):
                embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
                lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, target_data)
        if not os.path.exists(lab_loc):
            os.makedirs(lab_loc)
            print("Init lab loc: " + lab_loc)
        test_sub_name = "MIA_input_test(0.001).csv"
        if args.target_dim != 64 or args.shadow_dim != 64:
            assert shadow_data == target_data
            embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
            lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, target_data)
            test_sub_name = test_sub_name.replace(".csv", "{}-{}.csv".format(args.target_dim, args.shadow_dim))
        if os.path.exists(embeddings_loc + test_sub_name):
            print("Moving {}\n"
                  "    into {}".format(embeddings_loc + test_sub_name, lab_loc + test_sub_name)
                  )
            shutil.copyfile(embeddings_loc + test_sub_name,
                            lab_loc + test_sub_name)
            # check and delete source
            if os.path.exists(lab_loc + test_sub_name):
                os.remove(embeddings_loc + test_sub_name)
                print("removed " + embeddings_loc + test_sub_name)
        else:
            print("Skip moving file from " + embeddings_loc + test_sub_name)

        testing_file = lab_loc + test_sub_name
        print("loading testing from:" + testing_file)
        array_test = pd.read_csv(testing_file, header=None).dropna().to_numpy()

    else:
        # load P
        ratio_p_dict = {0: 1/6,
                         1: 1/4,
                         2: 1/2,
                         3: 3/4,
                         4: 5/6}
        ratio_p = ratio_p_dict[args.ratio_p]
        print("Launching attack 3 with ratio_p=", ratio_p)
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
        lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, target_data)
        if not os.path.exists(lab_loc):
            os.makedirs(lab_loc)
            print("Init lab loc: " + lab_loc)
        Ptrain_sub_name = "MIA_input_train(1.001).csv"
        if os.path.exists(embeddings_loc + Ptrain_sub_name):
            print("Moving {}\n"
                  "    into {}".format(embeddings_loc + Ptrain_sub_name, lab_loc + Ptrain_sub_name)
                  )
            shutil.copyfile(embeddings_loc + Ptrain_sub_name,
                            lab_loc + Ptrain_sub_name)
            # check and delete source
            if os.path.exists(lab_loc + Ptrain_sub_name):
                os.remove(embeddings_loc + Ptrain_sub_name)
                print("removed " + embeddings_loc + Ptrain_sub_name)
        else:
            print("Skip moving file from " + embeddings_loc + Ptrain_sub_name)
        Parray_train = pd.read_csv(lab_loc + Ptrain_sub_name, header=None).to_numpy()
        P_size = Parray_train.shape[0]
        S_size = int(Parray_train.shape[0]/ratio_p * (1-ratio_p))

        print("{} partial data and {} shadow data from {} total data".format(P_size, S_size, P_size + S_size))
        # load S
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, shadow_data)
        lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, shadow_data)
        if not os.path.exists(lab_loc):
            os.makedirs(lab_loc)
            print("Init lab loc: " + lab_loc)
        Strain_sub_name = "MIA_input_train(0.001).csv"
        if os.path.exists(embeddings_loc + Strain_sub_name):
            print("Moving {}\n"
                  "    into {}".format(embeddings_loc + Strain_sub_name, lab_loc + Strain_sub_name)
                  )
            shutil.copyfile(embeddings_loc + Strain_sub_name,
                            lab_loc + Strain_sub_name)
            # check and delete source
            if os.path.exists(lab_loc + Strain_sub_name):
                os.remove(embeddings_loc + Strain_sub_name)
                print("removed " + embeddings_loc + Strain_sub_name)
        else:
            print("Skip moving file from " + embeddings_loc + Strain_sub_name)
        Sarray_train = pd.read_csv(lab_loc + Strain_sub_name, header=None)
        size_S_total = Sarray_train.shape[0]
        if size_S_total <= S_size:
            Sarray_train = Sarray_train.sample(S_size, replace=True).to_numpy()
        else:
            Sarray_train = Sarray_train.sample(S_size, replace=True).to_numpy()

        # normalize in advance
        ss = StandardScaler()
        Parray_train[:, 2:-1] = ss.fit_transform(Parray_train[:, 2:-1])
        Sarray_train[:, 2:-1] = ss.fit_transform(Sarray_train[:, 2:-1])

        array_train = np.vstack([Parray_train, Sarray_train])
        del Parray_train, Sarray_train

        #load testing from T(1.001)
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
        lab_loc = "../../../../Wang-ds/dzhong2/MIA_against_KBRec/{}/{}/".format(shadow_model, target_data)
        test_sub_name = "MIA_input_test(1.001).csv"
        if target_model != shadow_model:
            test_sub_name = test_sub_name.replace("test", "test({})".format(target_model))

        if os.path.exists(embeddings_loc + test_sub_name):
            print("Moving {}\n"
                  "    into {}".format(embeddings_loc + test_sub_name, lab_loc + test_sub_name)
                  )
            shutil.copyfile(embeddings_loc + test_sub_name,
                            lab_loc + test_sub_name)
            # check and delete source
            if os.path.exists(lab_loc + test_sub_name):
                os.remove(embeddings_loc + test_sub_name)
                print("removed " + embeddings_loc + test_sub_name)
        else:
            print("Skip moving file from " + embeddings_loc + test_sub_name)

        testing_file = lab_loc + test_sub_name
        print("loading testing from:" + testing_file)
        array_test = pd.read_csv(testing_file, header=None).dropna().to_numpy()

    return array_train, array_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon-book-shadow")
    parser.add_argument("--model", type=str, default="NFM")
    parser.add_argument("--shadow_model", type=str, default="NFM")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--total_k", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--ratio_p", type=float, default=0)
    parser.add_argument('--gpu_ind', type=int, default=2, help='index of gpu')
    parser.add_argument('--partial_option', type=float, default=0, help='partial option in 0.01 or 0.05')
    parser.add_argument("--multi_attack", type=int, default=0)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--target_dim", type=int, default=64)
    parser.add_argument("--shadow_dim", type=int, default=64)
    parser.add_argument("--random_rec", type=float, default=0)
    parser.add_argument("--random_embedding", type=float, default=0)

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    shadow_model = args.shadow_model
    topK = args.topk
    total_k = args.total_k
    short_name_dict = {"am": "amazon-book-shadow",
                       "la": "last-fm-shadow",
                       "ye": "yelp2018-shadow"}
    if args.partial_option == 0:
        attack_type = 1
        target_data = dataset
        shadow_data = dataset
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
    elif args.partial_option > 1:
        attack_type = 1
        target_data = dataset
        shadow_data = dataset
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data)
    elif args.partial_option == 0.001: #dataset like am-la or yelp
        target_data = short_name_dict[dataset[:2]]
        if "shadow" not in dataset:
            if ("amazon" in dataset) or ("last-fm" in dataset) or ("yelp" in dataset):
                target_data = dataset
        if dataset[3:5] in short_name_dict:
            shadow_data = short_name_dict[dataset[3:5]]
            attack_type = 2
            embeddings_loc = "./KGAT_new/trained_model/{}/{}-{}/".format(shadow_model, target_data[:2], model)
        else:
            shadow_data = short_name_dict[dataset[:2]]
            attack_type = 2
            embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(shadow_model, target_data, model)
    else: #dataset like am-la-ps
        target_data = short_name_dict[dataset[:2]]
        shadow_data = short_name_dict[dataset[3:5]]
        attack_type = 3
        embeddings_loc = "./KGAT_new/trained_model/{}/{}/".format(model, target_data)
    if not os.path.exists(embeddings_loc):
        print("Init directory: "+ embeddings_loc)
        os.makedirs(embeddings_loc)

    array_train, array_test = MIA_input_loader(attack_type=attack_type,
                                               target_data=target_data,
                                               shadow_data=shadow_data,
                                               target_model=model,
                                               shadow_model=shadow_model,
                                               random_rec=args.random_rec,
                                               random_embedding = args.random_embedding,
                                               )

    ss = StandardScaler()
    array_train[:, 2:-1] = ss.fit_transform(array_train[:, 2:-1])

    array_test[:, 2:-1] = ss.fit_transform(array_test[:, 2:-1])

    device = torch.device("cuda:{}".format(args.gpu_ind) if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    if args.baseline == 1: # baseline1 threshold
        pop_df = threshold_attack(array_train, array_test, topK, total_k)
        save_loc = embeddings_loc + "Baseline1_popular_results({}-{}).csv".format(model, shadow_model)
        if args.partial_option:
            save_loc = save_loc.replace(".csv", "({}).csv".format(args.partial_option))
        pop_df.to_csv(save_loc)
        print("saved to " + save_loc)
        exit(0)
    elif args.baseline == 2: # baseline2 average
        metrics_list = []
        pop_res_list = []
        print("Applying multiple K attack")
        for K in [20, 40, 60, 80, 100]:
            for t in range(5):
                metric, pop_res = attack_average_k(array_train, array_test, K, device=device)
                pop_res_list += pop_res
                metrics_list.append([K] + metric)
        df_metric = pd.DataFrame(metrics_list, columns=["Top K", "Accuracy", "Precesion", "Recall", "F1","AUC"])
        df_metric = df_metric.groupby("Top K").mean()
        save_loc = embeddings_loc + "Baseline2_results({}-{}).csv".format(model, shadow_model)
        if args.partial_option:
            save_loc = save_loc.replace(".csv", "({}).csv".format(args.partial_option))
        df_metric.to_csv(save_loc)
        print("saved to " + save_loc)

        df_pop = pd.DataFrame(pop_res_list, columns=["Top K popular", "Accuracy", "Precesion", "Recall", "AUC","F1", "K", "Category"])
        df_pop = df_pop.groupby(["Top K popular", "K", "Category"]).mean()
        df_pop.to_csv(save_loc.replace("Baseline2_results", "Baseline2_popular_results"))
        print("saved to " + save_loc.replace("Baseline2_results", "Baseline2_popular_results"))
        exit(0)
    else:
        pass

    if args.multi_attack:
        metrics_list = []
        pop_res_list = []
        print("Applying multiple K attack")
        for K in [20, 40, 60, 80, 100]:
            for t in range(5):
                metric, pop_res = attack_k(array_train, array_test, K, device=device)
                pop_res_list += pop_res
                metrics_list.append([K] + metric)
        df_metric = pd.DataFrame(metrics_list, columns=["Top K", "Accuracy", "Precesion", "Recall", "AUC", "F1"])
        df_metric = df_metric.groupby("Top K").mean()
        save_loc = embeddings_loc + "MIA_results({}-{}).csv".format(model, shadow_model)
        if args.partial_option:
            save_loc = save_loc.replace(".csv", "({}).csv".format(args.partial_option))

        if args.random_rec > 0:
            save_loc = save_loc.replace(".csv", f"-random_rec-{args.random_rec}.csv")

        if args.random_embedding > 0:
            save_loc = save_loc.replace(".csv", f"-random_embd-{args.random_embedding}.csv")
        df_metric.to_csv(save_loc)
        print("saved to " + save_loc)

        df_pop = pd.DataFrame(pop_res_list, columns=["Top K popular", "Accuracy", "Precesion", "Recall", "AUC", "F1", "K", "Category"])
        df_pop = df_pop.groupby(["Top K popular", "K", "Category"]).mean()
        df_pop.to_csv(save_loc.replace("MIA_results", "Popular_results"))
        print("saved to " + save_loc.replace("MIA_results", "Popular_results"))
    else:
        pop_df_list = []
        for t in range(5):
            _, df_pop = attack_k(array_train, array_test, 20, device=device)
            pop_df_list += df_pop
        save_loc = embeddings_loc + "MIA_results({}-{}).csv".format(model, shadow_model)
        if args.partial_option:
            save_loc = save_loc.replace(".csv", "({}).csv".format(args.partial_option))
        if target_data != shadow_data:
            save_loc = save_loc.replace(".csv", "({}-{}).csv".format(target_data[:2], shadow_data[:2]))
        #print(pop_df_list)
        df_pop = pd.DataFrame(pop_df_list, columns=["Top K popular", "Accuracy", "Precesion", "Recall", "AUC","F1", "K", "Category"])
        df_pop = df_pop.groupby(["Top K popular", "K", "Category"]).mean()
        if args.target_dim != 64 or args.shadow_dim != 64:
            save_loc = save_loc.replace(".csv", "{}-{}.csv".format(args.target_dim, args.shadow_dim))

        if args.random_rec > 0:
            save_loc = save_loc.replace(".csv", f"-random_rec-{args.random_rec}.csv")

        if args.random_embedding > 0:
            save_loc = save_loc.replace(".csv", f"-random_embd-{args.random_embedding}.csv")
        print("saved to " + save_loc)
        if attack_type == 3:
            df_pop.to_csv(save_loc.replace("MIA_results", "Popular_results_{}".format(args.ratio_p)))
            print("saved to " + save_loc.replace("MIA_results", "Popular_results_{}".format(args.ratio_p)))
        else:
            df_pop.to_csv(save_loc.replace("MIA_results", "Popular_results"))
            print("saved to " + save_loc.replace("MIA_results", "Popular_results"))



