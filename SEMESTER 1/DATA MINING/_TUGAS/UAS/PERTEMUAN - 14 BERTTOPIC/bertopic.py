import pandas as pd 
import numpy as np
from bertopic import BERTopic

df = pd.read_csv("Indonesian_.csv", engine='python')

print(df.head)

# from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups

# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))["data"]

# seed_topic_list = [["drug", "cancer", "drugs", "doctor"],
#                    ["windows", "drive", "dos", "file"],
#                    ["space", "launch", "orbit", "lunar"]]

# topic_model = BERTopic(seed_topic_list=seed_topic_list)
# topics, probs = topic_model.fit_transform(docs)
