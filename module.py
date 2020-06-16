import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
import os



class TsvParser:
    """
    Class for tsv parsing, returns iterable by chunks.
    There are 2 methods of reading - incremental and chunks.

    :param path: string
        Any valid string path is acceptable.
    :param needed_feature_codes: 'all' or object, default = 'all'
        Which feature codes are needed. By default all codes are used.
    :param reading_type: 'incremental', 'chunks', default='incremental'
        How to perform reading - incrementally (1 by 1) or in chunks
    :param chunksize: int
        Should be set if :param reading_type set 'chunks'.
        Otherwise an error is raised.

    """
    def __init__(self,
                 path:str,
                 reading_type:str = 'incremental',
                 chunksize: int = 1):



        self.chunksize = chunksize
        self.path = path
        self.reading_type = reading_type
        pass

    def get_iterable(self):
        if self.chunksize > 1 and self.reading_type == 'incremental':
            raise ValueError("reading_type = 'incremental' is only supported "
                                 "if chunksize = 1")

        if not isinstance(self.path, str):
            raise ValueError("path=%r must be of type str" % self.path)

        if not self.reading_type in ['chunks', 'incremental']:
            raise ValueError("Unrecognized reading_type=%r. Available types are"
                             "['incremental', 'chunks']"
                             % self.reading_type)

        return pd.read_csv(self.path, sep='\t', chunksize=self.chunksize)

class z_standardizer(TransformerMixin, BaseEstimator):
    """
    An alternative to sklearn's StandardScaler but with Welford's online algorithm
    Implemented as an sklearn transformer

    :param needed_feature_codes: 'all' or object, default = 'all'
    """
    def __init__(self):
        self._mean = None
        self._variance = None

    def fit(self, X, y=None):
        # при новом вызове фита - удаляем имеющиеся значения
        self._variance = None
        self._mean = None

        for elem in X:
            self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            X = X.values[0]

        # если не инициализированы среднее и дисперсия -
        # все сбрасываем
        if (self._mean is None) and (self._variance is None):
            self._mean = np.zeros(X.shape)
            self._M2 = np.zeros(X.shape)
            self._count = 0


        self._count += 1
        delta = X - self._mean
        self._mean += delta / self._count
        delta2 = X - self._mean
        self._M2 += delta * delta2

        # используем несмещенную дисперсию - ну, на всякий случай
        self._variance =  self._M2 / (self._count - 1)
        return self

    def transform(self, X, copy=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        return (X - self._mean)/np.sqrt(self._variance)


class ParseVacations:
    """
    A class that joins reading and standardization as a pipeline.

    :param needed_feature_codes: 'all' or object, default = 'all'
        Which feature codes are needed. By default all codes are used.
    """

    def __init__(self,
                 needed_feature_codes):
        self.needed_feature_codes = needed_feature_codes
        self.feature_names = ['feature_' + str(count) for count in range(256)]
        self.is_fitted = False

    def prepare_chunk(self, X):
        # именуем колонки, разбиваем фичи вакансии
        features = ['feature_code'] + ['feature_' + str(count) for count in range(256)]
        X[features] = X['features'].str.split(',', expand=True).astype('int32')
        X.columns = ['id_job', 'features'] + features
        return X

    def get_code_features(self, X, feature_code):
        # создана для удобства - чтобі не писать одно и то же несколько раз
        # возвращает фичи вакансии по нужному фича-коду
        return X[X['feature_code'] == feature_code][self.feature_names]

    def train_scalers(self,
                      tsv_parser: TsvParser,
                      standartizer):
        # метод для обучения скейлеров

        self.tsv_parser = tsv_parser
        self.code_scaler = {}
        # итерируемся по чанкам. В инкрементальном случае, чанк = 1
        for chunk in tsv_parser.get_iterable():
            chunk = self.prepare_chunk(chunk)
            # Для каждого фичер кода фиттим скейлер отдельно
            for feature_code in self.needed_feature_codes:
                if not feature_code in self.code_scaler:
                    self.code_scaler[feature_code] = standartizer()
                self.code_scaler[feature_code].partial_fit(
                    self.get_code_features(chunk[chunk.feature_code == feature_code], feature_code))

        self.is_fitted = True
        return self

    def transform_file(self,
                      path):
        # Еще раз проходимся по файлу и трансформируем, попутно записывая в новый
        if not self.is_fitted:
            raise NotFittedError('this instance is not fitted yet')

        # итерируемся по чанкам. В инкрементальном случае, чанк = 1
        for chunk in self.tsv_parser.get_iterable():
            chunk = self.prepare_chunk(chunk)

            # итерирование по фичер кодам
            for feature_code in self.needed_feature_codes:

                mask = chunk[chunk['feature_code'] == feature_code].index


                chunk[f'max_feature_{feature_code}_index'] = 0
                chunk[f'max_feature_{feature_code}_abs_mean_diff'] = 0

                # индекс  самого большого значения среди фич
                chunk.loc[mask, f'max_feature_{feature_code}_index'] = \
                    chunk[self.feature_names].apply(lambda s: self.feature_names.index(max(zip(s, s.keys()))[1]), 1)

                # значение самой большой фичи минус среднее
                chunk.loc[mask, [f'max_feature_{feature_code}_abs_mean_diff']] = \
                chunk[self.feature_names].apply(lambda s: max(zip(s, s.keys()))[0] - \
                        self.code_scaler[feature_code]._mean[self.feature_names.index(max(zip(s, s.keys()))[1])], 1)

                # теперь стандартизируем фичи
                chunk[self.feature_names] = self.code_scaler[feature_code].transform(
                                                        self.get_code_features(chunk, feature_code))

            # если файла еще нет, тогда записываем с хидером
            if not os.path.isfile('./path_of_file'):
                chunk.to_csv(path, sep='\t',  index=False, mode='a')

            chunk.to_csv(path, sep='\t',  index=False, header=None, mode='a')
        return self
