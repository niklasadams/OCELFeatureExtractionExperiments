from ocpa.algo.filtering.log import time_filtering
import ocpa.algo.feature_extraction.factory as feature_extraction
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
def construct_table(feature_storage, index_list = "all", exclude_events = []):
    if index_list == "all":
        index_list = list(range(0,len(feature_storage.feature_graphs)))
    features = feature_storage.event_features
    df = pd.DataFrame(columns=features)
    print(df)
    dict_list = []
    for g in tqdm([feature_storage.feature_graphs[i] for i in index_list]):
        for node in g.nodes:
            if node.event_id not in exclude_events:
                dict_list.append(node.attributes)
            #print(node.attributes)
    df = pd.DataFrame(dict_list)
    print(df)
    return df
