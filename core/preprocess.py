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
import numpy as np
from transformers import AutoTokenizer, AutoModel


# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
BATCH_SIZE = 128


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


def build_edge_list(row):
  user_id = row['id']
  edge_list = []
  mentioned_users = light_polluters[light_polluters['id']==user_id]['mentions']
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


def quantize(df, q=10, with_one_hot=True):
    df = pd.qcut(df, q=q)
    if with_one_hot:
        return pd.get_dummies(df)
    return df 


def extract_bert_embeddings(df):
  
    # build tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # get tokens 
    tokens = [tokenizer.encode(str(entry), padding='max_length', max_length=20, truncation=True) for entry in df.tolist()]
    tokens = np.array(tokens)

    # batch processing  
    description_features = []

    batched_array = np.array_split(tokens, int(tokens.shape[0] / batch_size))

    for batch in tqdm.tqdm(batched_array):
        batch = torch.tensor(batch).to(device)
        outputs = model(batch)
        embs = outputs['last_hidden_state'][:, 0, :]
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
        followers_count_features,
        friends_count_features,
        date_account_created_features,
        has_profile_pic,
        has_profile_and_bg_pic,
        description_features
    ],
    axis=1
    )
    return node_features
        

def main(args):
    """
    Process. 
    Args:
        args (Namespace): parsed arguments.
    """

    light_polluters = load_csv(os.path.join(args.base_path, 'light_content_polluters.csv'))
    mention_polluters = load_csv(os.path.join(args.base_path, 'content_polluters_mentions.csv'))
    polluters_info = load_csv(os.path.join(args.base_path, 'content_polluters_info.csv'))

    light_legit = load_csv(os.path.join(args.base_path, 'light_legit.csv'))
    legit_mentions = load_csv(os.path.join(args.base_path, 'legit_mentions.csv'))
    legit_info = load_csv(os.path.join(args.base_path, 'legit_info.csv'))

    polluters_info_node_features = build_node_features(polluters_info)
    legit_info_node_features = build_node_features(legit_info)
    

if __name__ == "__main__":
    main(args=parse_arguments())
