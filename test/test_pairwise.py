import pandas as pd
import sys
sys.path.append("/home/nick/Pairwise-Metrics/src")
from pairwise import distances


data = pd.read_csv("/home/nick/Pairwise-Metrics/test/LungCap.csv")

metrics = distances(data)

metrics.correlation

metrics.covariance

metrics.euclidean

metrics.cosine
