import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 计算accept rate
def get_metrics(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ['n_predict','draft_len','encode_speed','decode_speed','itt','final_ttft','accept_rate']
    return df['encode_speed'].mean(), df['decode_speed'].mean(), df['itt'].mean(), df['final_ttft'].mean(), df['accept_rate'].mean()

n_predicts = [32, 64, 128]
drafts = [4, 8]
for n_predict in n_predicts:
    for draft in drafts:
        csv_path = f'./datasets/{n_predict}_{draft}_result.csv'
        encode_speed, decode_speed, itt, final_ttft, accept_rate = get_metrics(csv_path)
        print(f'n_predict={n_predict}, draft={draft}, encode_speed={encode_speed}, decode_speed={decode_speed}, itt={itt}, final_ttft={final_ttft}, accept_rate={accept_rate}')