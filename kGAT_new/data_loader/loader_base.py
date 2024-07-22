import os
import time
import random
import collections

import torch
import numpy as np
import pandas as pd
import shutil
from collections import defaultdict


class DataLoaderBase(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        if args.train_shadow:
            print("Using Shadow Graph instead of Target Graph")
            if args.partial_option > 0:
                self.train_file = os.path.join(self.data_dir, 'train_shadow({}).txt'.format(args.partial_option))
            else:
                self.train_file = os.path.join(self.data_dir, 'train_shadow.txt')
            print("Loading training-shadow from " + self.train_file)
        else:
            self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        if args.kg_partial > 0:
            print("Loading kg-shadow from " + 'kg_final({}).txt'.format(args.kg_partial))
            self.kg_file = os.path.join(self.data_dir, 'kg_final({}).txt'.format(args.kg_partial))
        else:
            print("Loading kg from " + 'kg_final.txt')
            self.kg_file = os.path.join(self.data_dir, 'kg_final.txt')
        if not args.kg_only:
            self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
            self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
            self.statistic_cf()
        else:
            self.cf_train_data, self.train_user_dict = ([], []), {}
            self.cf_test_data, self.test_user_dict = ([], []), {}
            self.n_users = 0
            self.n_items = 0
            item_list_file = self.data_dir + "/item_list.txt"
            lines = open(item_list_file, 'r').readlines()
            count = 0
            for l in lines:
                if count==0:
                    count += 1
                    continue
                tmp = l.strip()
                item_id = int(tmp.split()[1])
                self.n_items = max(self.n_items, item_id + 1)

            self.n_cf_train = 0
            self.n_cf_test = 0

            pass

        if self.use_pretrain == 1:
            self.load_pretrained_data()


    def load_cf(self, filename):
        if self.args.rm > 0:
            ints = os.path.join(self.data_dir, 'ints_to_rm.csv')
            df_rm = pd.read_csv(ints)
            np.random.seed(None)
            seed = int(np.random.random_sample() * 100)
            print(seed)
            df_rm = df_rm.groupby('label').sample(n=1, replace=False, random_state=seed)
            rm_dict = defaultdict(list)
            for i, row in df_rm.iterrows():
                if row['label'] == self.args.rm - 1:
                    rm_dict[row['user']].append(row['item'])
        else:
            rm_dict = {}
        np.random.seed(self.args.seed)
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))
                if user_id in rm_dict:
                    rm_id_list = rm_dict[user_id]
                    for item_id in item_ids:
                        if item_id in rm_id_list:
                            print(f"removed int {user_id}-{item_id}")
                            continue
                        user.append(user_id)
                        item.append(item_id)
                    user_dict[user_id] = [x for x in item_ids if x not in rm_id_list]
                else:
                    for item_id in item_ids:
                        user.append(user_id)
                        item.append(item_id)
                    user_dict[user_id] = item_ids
        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)

        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1

        if self.data_name in ["am-la", 'la-am', 'ye-am', 'am-ye', 'ye-la', 'la-ye']:
            print("add complete item list to item")
            item_list_file = self.data_dir + "/item_list.txt"
            lines = open(item_list_file, 'r').readlines()
            count = 0
            for l in lines:
                if count==0:
                    count += 1
                    continue
                tmp = l.strip()
                item_id = int(tmp.split()[1])
                self.n_items = max(self.n_items, item_id + 1)
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data


    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break
            if n_pos_items == 0:
                print('Wait! should not be zero??')
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items


    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items


    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = user_dict.keys()
        exist_users = [key for key in exist_users if len(user_dict[key]) > 0]
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(list(exist_heads), batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim
        assert self.item_pre_embed.shape[1] == self.args.embed_dim


