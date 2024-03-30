import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


class FeatureGenerator:
    def __init__(self, train, songs):
        """
        Creates following features:
        1) target encoding features
        2) conditional probabilities
        3) svd features from user_item matrix
        :param train:
        """
        self.train = train.copy()
        self.songs = songs

        self.artist_target = None
        self.composer_target = None
        self.lyricist_target = None

        self.total_counts = None
        self.genre_counts = None
        self.tab_counts = None
        self.screen_counts = None

        self.user_features = None
        self.item_features = None

    def conditional_probabilities(self, predictor, condition, data):
        total_counts = data.groupby(condition).size().reset_index(name='count')
        predictor_counts = data.groupby([condition, predictor]).size().reset_index(name=f'{predictor}_count')
        predictor_counts = pd.merge(predictor_counts, total_counts, on=condition)
        predictor_counts[f'{predictor}_probability'] = predictor_counts[f'{predictor}_count'] / predictor_counts[
            'count']
        predictor_counts = predictor_counts.drop([f'{predictor}_count', 'count'], axis=1)
        return predictor_counts

    def svd_features(self):
        user_item = self.train.groupby(['msno', 'song_id']).size().reset_index(name='count')
        user_item['count'] = user_item['count'].astype(float)

        # unique users and songs
        idx_user = user_item['msno'].unique()
        idx_item = user_item['song_id'].unique()

        # hash-map to indexes
        user_idx = {user_id: idx for idx, user_id in enumerate(idx_user)}
        item_idx = {item_id: idx for idx, item_id in enumerate(idx_item)}

        # replace ids with indexes
        user_item['msno'] = user_item['msno'].map(lambda x: user_idx[x])
        user_item['song_id'] = user_item['song_id'].map(lambda x: item_idx[x])

        # sparse matrix
        user_item_matrix = coo_matrix(
            (user_item['count'], (user_item['msno'], user_item['song_id']))
        )
        user_item_matrix = user_item_matrix.tocsr()

        # compute svd features
        u, s, vt = svds(user_item_matrix, k=20)
        user_features = np.dot(u, np.diag(s))
        item_features = vt.T

        user_features = pd.DataFrame(user_features).reset_index().rename(columns={'index': 'msno'})
        user_features['msno'] = user_features['msno'].map(lambda x: idx_user[x])

        item_features = pd.DataFrame(item_features).reset_index().rename(columns={'index': 'song_id'})
        item_features['song_id'] = item_features['song_id'].map(lambda x: idx_item[x])

        self.user_features = user_features
        self.item_features = item_features
        return

    def fit_transform(self, df):
        df.copy()
        df = pd.merge(df, self.songs, on='song_id', how='left')

        # target features
        self.artist_target = df.groupby('artist_name').agg({'target': 'mean'}) \
            .rename(columns={'target': 'target_artist'}).reset_index()
        self.composer_target = df.groupby('composer').agg({'target': 'mean'}) \
            .rename(columns={'target': 'target_composer'}).reset_index()
        self.lyricist_target = df.groupby('lyricist').agg({'target': 'mean'}) \
            .rename(columns={'target': 'target_lyricist'}).reset_index()

        df = pd.merge(df, self.artist_target, on='artist_name', how='left')
        df = pd.merge(df, self.composer_target, on='composer', how='left')
        df = pd.merge(df, self.lyricist_target, on='lyricist', how='left')

        # conditional probabilities features
        predictors = ['source_system_tab', 'source_screen_name', 'source_type',
                      'genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
        for predictor in predictors:
            predictor_counts = self.conditional_probabilities(predictor, 'msno', df)
            df = pd.merge(df, predictor_counts, on=['msno', predictor], how='left')
            df[f'{predictor}_probability'] = df[f'{predictor}_probability'].fillna(0)

        predictors = ['source_system_tab', 'source_screen_name', 'source_type']
        for predictor in predictors:
            predictor_counts = self.conditional_probabilities(predictor, 'song_id', df)
            df = pd.merge(df, predictor_counts, on=['song_id', predictor], how='left', suffixes=('', '_song'))
            df[f'{predictor}_probability_song'] = df[f'{predictor}_probability_song'].fillna(0)

        # svd features
        self.svd_features()
        df = pd.merge(df, self.user_features, on='msno', how='left')
        df = pd.merge(df, self.item_features, on='song_id', how='left')

        return df

    def transform(self, df):
        df = pd.merge(df, self.songs, on='song_id', how='left')
        df = pd.merge(df, self.artist_target, on='artist_name', how='left')
        df = pd.merge(df, self.composer_target, on='composer', how='left')
        df = pd.merge(df, self.lyricist_target, on='lyricist', how='left')

        predictors = ['source_system_tab', 'source_screen_name', 'source_type',
                      'genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
        for predictor in predictors:
            predictor_counts = self.conditional_probabilities(predictor, 'msno', pd.concat((df, self.train)))
            df = pd.merge(df, predictor_counts, on=['msno', predictor], how='left')
            df[f'{predictor}_probability'] = df[f'{predictor}_probability'].fillna(0)

        predictors = ['source_system_tab', 'source_screen_name', 'source_type']
        for predictor in predictors:
            predictor_counts = self.conditional_probabilities(predictor, 'song_id', pd.concat((df, self.train)))
            df = pd.merge(df, predictor_counts, on=['song_id', predictor], how='left', suffixes=('', '_song'))
            df[f'{predictor}_probability_song'] = df[f'{predictor}_probability_song'].fillna(0)

        df = pd.merge(df, self.user_features, on='msno', how='left')
        df = pd.merge(df, self.item_features, on='song_id', how='left')
        return df
