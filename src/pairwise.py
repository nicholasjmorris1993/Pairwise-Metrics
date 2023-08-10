import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def distances(df):
    model = Distances()
    model.correlation_distance(df)
    model.covariance_distance(df)
    model.euclidean_distance(df)
    model.cosine_distance(df)

    return model

class Distances:
    def correlation_distance(self, df):
        df = df.copy()
        self.correlation = df.corr()

        # group columns together with hierarchical clustering
        X = self.correlation.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the correlation matrix for the received dataframe
        self.correlation = df.corr()
        
        # plot the correlation matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(self.correlation, cmap="RdYlGn")
        plt.xticks(range(len(self.correlation.columns)), self.correlation.columns, rotation=90);
        plt.yticks(range(len(self.correlation.columns)), self.correlation.columns);
        
        # add the colorbar legend
        fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

        title = "Correlation Plot"
        fig.suptitle(title, y=1.08)
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")

    def covariance_distance(self, df):
        df = df.copy()
        self.covariance = df.cov()

        # group columns together with hierarchical clustering
        X = self.covariance.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the covariance matrix for the received dataframe
        self.covariance = df.cov()
        
        # plot the covariance matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(self.covariance, cmap="RdYlGn")
        plt.xticks(range(len(self.covariance.columns)), self.covariance.columns, rotation=90);
        plt.yticks(range(len(self.covariance.columns)), self.covariance.columns);
        
        # add the colorbar legend
        fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

        title = "Covariance Plot"
        fig.suptitle(title, y=1.08)
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")

    def euclidean_distance(self, df):
        df = df.copy()
        self.euclidean = self._euclidean_distance(df)

        # group columns together with hierarchical clustering
        X = self.euclidean.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the euclidean matrix for the received dataframe
        self.euclidean = self._euclidean_distance(df)
        
        # plot the euclidean matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(self.euclidean, cmap="RdYlGn")
        plt.xticks(range(len(self.euclidean.columns)), self.euclidean.columns, rotation=90);
        plt.yticks(range(len(self.euclidean.columns)), self.euclidean.columns);
        
        # add the colorbar legend
        fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

        title = "Euclidean Distance Plot"
        fig.suptitle(title, y=1.08)
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")

    def cosine_distance(self, df):
        df = df.copy()
        self.cosine = self._cosine_distance(df)

        # group columns together with hierarchical clustering
        X = self.cosine.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the cosine matrix for the received dataframe
        self.cosine = self._cosine_distance(df)
        
        # plot the cosine matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(self.cosine, cmap="RdYlGn")
        plt.xticks(range(len(self.cosine.columns)), self.cosine.columns, rotation=90);
        plt.yticks(range(len(self.cosine.columns)), self.cosine.columns);
        
        # add the colorbar legend
        fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

        title = "Cosine Distance Plot"
        fig.suptitle(title, y=1.08)
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")

    def _euclidean_distance(self, df):
        euclidean = euclidean_distances(df.T)
        euclidean = pd.DataFrame(
            euclidean, 
            columns=df.columns, 
            index=df.columns,
        )

        return euclidean

    def _cosine_distance(self, df):
        cosine = cosine_distances(df.T)
        cosine = pd.DataFrame(
            cosine, 
            columns=df.columns, 
            index=df.columns,
        )

        return cosine
