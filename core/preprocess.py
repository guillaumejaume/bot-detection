#!/usr/bin/env python3
"""
Script for preprocessing user graphs
"""
import os
import csv
import torch
import time 
import tqdm 
import pandas as pd
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, timezone
import dgl 
from dgl.data.utils import save_graphs


# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
BATCH_SIZE = 128
FEATS_OUT_PATH = '/dataP/gja/bot_detection/features'
GRAPHS_OUT_PATH = '/dataP/gja/bot_detection/graphs'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='path to the raw csv.',
        default='',
        required=False
    )
    return parser.parse_args()


def load_csv(fname):
    data = pd.read_csv(fname, lineterminator='\n')
    return data 


def build_topology(df, df_mentioned):

    def build_edge_list(row):
        user_id = row['id']
        edge_list = []
        mentioned_users = df_mentioned[df_mentioned['id']==user_id]['mentions']
        mentioned_users = mentioned_users.tolist()
        mentioned_users = [list(map(int, entry.replace('[', '').replace(']', '').replace(' ', '').split(','))) for entry in mentioned_users]
        for mentioned_per_tweet in mentioned_users:
            for src in mentioned_per_tweet:
                if user_id != src:
                    edge_list.append([user_id, src])
                    edge_list.append([src, user_id])
            for dst in mentioned_per_tweet:
                if src != dst:
                    edge_list.append([src, dst])
        return edge_list

    return df.apply(build_edge_list, axis=1)


def quantize(df, q=10, with_one_hot=True):
    df = pd.qcut(df, q=q)
    if with_one_hot:
        return pd.get_dummies(df)
    return df 


def extract_bert_embeddings(df):
  
    # build tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)

    # get tokens 
    tokens = [tokenizer.encode(str(entry), padding='max_length', max_length=20, truncation=True) for entry in df.tolist()]
    tokens = np.array(tokens)

    # batch processing  
    description_features = []

    batched_array = np.array_split(tokens, int(tokens.shape[0] / BATCH_SIZE))

    for batch in tqdm.tqdm(batched_array):
        batch = torch.tensor(batch).to(DEVICE)
        outputs = model(batch)
        embs = outputs[0][:, 0, :]
        description_features.append(embs.cpu().detach().numpy())

    description_features = np.concatenate(description_features)

    description_features = pd.DataFrame(
        description_features,
        columns=['bert_' + str(i) for i in range(description_features.shape[1])]
    )

    return description_features


def build_node_features(df):
    # 10x features for followers count
    followers_count_features = quantize(df['followers_count'])

    # 10x features for friend count
    friends_count_features = quantize(df['friends_count'])

    # 10x features for number of days accounted created 
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].apply(lambda x: x.replace(tzinfo=None))
    df['created_days'] = (pd.datetime.now() - df['created_at']).dt.days
    date_account_created_features = quantize(df['created_days'])

    # 768 features for description embedding
    description_features = extract_bert_embeddings(df['descrition'])

    # 1x feature for profile pic
    has_profile_pic = df['default_profile_image']

    # 1x features for profile + background pic 
    has_profile_and_bg_pic = df['default_profile']

    # stitch all the features into one array: resulting in 800 features per node. 
    node_features = pd.concat([
        df['id'],
        followers_count_features,
        friends_count_features,
        date_account_created_features,
        has_profile_pic,
        has_profile_and_bg_pic,
        description_features
    ],
    axis=1,
    )
    return node_features


def build_dgl_graphs(features, mentioned_features):

    def build_graph(row):

        g = dgl.DGLGraph()
        node_ids = np.unique(np.array([row['id']] + [item for sublist in row['edge_list'] for item in sublist]))

        all_feats = []
        valid_node_ids = []
        for user_id in node_ids:
            feats = mentioned_features[mentioned_features['id']==user_id][:1].to_numpy().astype(float)  #.tolist()[1:]
            feats = torch.from_numpy(feats).squeeze()
            feats = feats[1:]
            if feats.shape[0] == 0:
                feats = row[1:-2].to_numpy().astype(float)
                feats = torch.from_numpy(feats).squeeze()
            if feats.shape[0] != 0:
                all_feats.append(feats)
                valid_node_ids.append(user_id)

        g.add_nodes(len(valid_node_ids))
        if len(valid_node_ids) > 0:
            g.ndata['feats'] = torch.stack(all_feats)
            g.ndata['user_id'] = torch.LongTensor(valid_node_ids)

        edge_list = row['edge_list']
        valid_edge_list = [edge for edge in edge_list if edge[0] in valid_node_ids and edge[1] in valid_node_ids]
        src = [x[0] for x in valid_edge_list]
        dst = [x[1] for x in valid_edge_list]
        mapping = {user_id:i for i, user_id in enumerate(valid_node_ids)}
        g.add_edges([mapping[x] for x in src], [mapping[x] for x in dst])

        # print('Graph', g.number_of_nodes(), g.number_of_edges())
        if g.number_of_edges() == 0:
            print(row['label'])
        fname = str(row['id']) + '.bin'
        # save_graphs(os.path.join(GRAPHS_OUT_PATH, fname), [g], {'label': torch.LongTensor([row['label']])})

        return g

    return features.apply(build_graph, axis=1)

        

def main(args):
    """
    Process. 
    Args:
        args (Namespace): parsed arguments.
    """

    pre_load = True
    if not pre_load:

        # 1.  Load all the raw csv 
        light_polluters = load_csv(os.path.join(args.data_path, 'light_content_polluters.csv'))
        polluters_mentions = load_csv(os.path.join(args.data_path, 'content_polluters_mentions.csv'))
        polluters_info = load_csv(os.path.join(args.data_path, 'content_polluters_info.csv'))
        polluters_info['label'] = np.ones(polluters_info.shape[0]) # 1 is polluter 0 is legit 

        light_legit = load_csv(os.path.join(args.data_path, 'light_legit.csv'))
        legits_mentions = load_csv(os.path.join(args.data_path, 'legit_mentions.csv'))
        legit_info = load_csv(os.path.join(args.data_path, 'legit_info.csv'))
        legit_info['label'] = np.zeros(legit_info.shape[0]) # 1 is polluter 0 is legit 

        # 2. Stitch the databases
        users_info = pd.concat([
                polluters_info,
                legit_info
            ],
            axis=0,
            ignore_index=True
        )

        mentioned = pd.concat([
                light_polluters,
                light_legit
            ],
            axis=0,
            ignore_index=True
        )

        mentioned_info = pd.concat([
                polluters_mentions,
                legits_mentions
            ],
            axis=0,
            ignore_index=True
        )

        # 3. Get user-level features 
        users_features = build_node_features(users_info)

        # 4. Add label to the set of features 
        users_features['label'] = users_info['label']

        # 5. Add edge list to the set of features and save
        users_features['edge_list'] = build_topology(users_info, mentioned) 
        users_features.to_pickle(os.path.join(FEATS_OUT_PATH, 'users_features.pickle'))

        # 6. Get mentioned user-level features and save 
        mentioned_users_features = build_node_features(mentioned_info)
        mentioned_users_features.to_pickle(os.path.join(FEATS_OUT_PATH, 'mentioned_users_features.pickle'))

    else:
        users_features = pd.read_pickle(os.path.join(FEATS_OUT_PATH, 'users_features.pickle'))
        mentioned_users_features = pd.read_pickle(os.path.join(FEATS_OUT_PATH, 'mentioned_users_features.pickle'))

    # 4. Build DGL graphs
    build_dgl_graphs(users_features, mentioned_users_features)


if __name__ == "__main__":
    main(args=parse_arguments())
