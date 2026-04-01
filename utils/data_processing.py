import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
import torchvision.transforms as transforms

import util
import webmercator


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGERY_DIR = os.path.join(DATA_DIR, 'imagery')
IMAGERY_NPZ_DIR = os.path.join(DATA_DIR, 'imagery_npz')
ZIP_CODES_PATH = os.path.join(DATA_DIR, 'ziplatlon.csv')
CENSUS_PATH = os.path.join(DATA_DIR, '16zpallnoagi.csv')
MAPPING_PATH = os.path.join(DATA_DIR, 'mapping.json')
MAPPING_CACHE_VERSION = 2
DEFAULT_TEST_SIZE = 0.2


def get_all_tiles(min_x, max_x, min_y, max_y):
    return [
        (x, y)
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
    ]


def filter_zip_coordinate_rows(zip_codes_data, census_data):
    valid_zip_codes = set(census_data['ZIPCODE']).intersection(set(zip_codes_data['zip']))
    filtered_zip_codes = (
        zip_codes_data[zip_codes_data['zip'].isin(valid_zip_codes)]
        .drop_duplicates(subset='zip', keep='first')
        .reset_index(drop=True)
    )
    return filtered_zip_codes, valid_zip_codes


def build_zip_codes_to_tiles(zip_codes_data, zoom=14):
    zip_codes_to_tiles = {}
    for row in zip_codes_data.itertuples(index=False):
        x, y = webmercator.xy(row.latitude, row.longitude, z=zoom)
        zip_codes_to_tiles[int(row.zip)] = (int(x), int(y))
    return zip_codes_to_tiles


def assign_tiles_to_zipcodes(all_tiles, zip_codes_to_tiles):
    if not zip_codes_to_tiles:
        raise ValueError('No ZIP coordinates available for tile assignment.')

    mapping = defaultdict(list)
    remaining_images = set(range(len(all_tiles)))
    tiles_to_indexes = {tile: index for index, tile in enumerate(all_tiles)}

    for zip_code, tile in zip_codes_to_tiles.items():
        index = tiles_to_indexes.get(tile)
        if index is None:
            continue
        mapping[zip_code].append(index)
        remaining_images.discard(index)

    for index in sorted(remaining_images):
        curr_x, curr_y = all_tiles[index]
        nearest_zip = min(
            zip_codes_to_tiles,
            key=lambda zip_code: (
                (curr_x - zip_codes_to_tiles[zip_code][0]) ** 2
                + (curr_y - zip_codes_to_tiles[zip_code][1]) ** 2
            ),
        )
        mapping[nearest_zip].append(index)

    return mapping


def build_groups_from_mapping(mapping, num_images):
    groups = np.full(num_images, -1, dtype=np.int64)
    for zip_code, indexes in mapping.items():
        groups[np.asarray(indexes, dtype=np.int64)] = int(zip_code)

    if np.any(groups < 0):
        raise ValueError('Every image must be assigned to exactly one ZIP group.')
    return groups


def grouped_train_test_split(X, y, groups, test_size=DEFAULT_TEST_SIZE, random_state=0):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        groups[train_idx],
        groups[test_idx],
    )


class Mapping(object):

    MIN_X = 2794
    MAX_X = 2838
    MIN_Y = 6528
    MAX_Y = 6571

    def __init__(self):
        self.allTiles = get_all_tiles(self.MIN_X, self.MAX_X, self.MIN_Y, self.MAX_Y)
        self.convertImagesToNpz(range(self.MIN_X, self.MAX_X + 1),
                                range(self.MIN_Y, self.MAX_Y + 1))
        self.loadImages()
        self.readZipCodesData()
        self.readCensusData()

        self.mappingSaved = False
        self.mapping = defaultdict(list)
        num_images = self.getNumImages()
        self.labels = np.zeros(num_images, dtype=np.float32)
        self.income = np.zeros(num_images, dtype=np.float64)
        self.population = np.zeros(num_images, dtype=np.float64)

    def convertImagesToNpz(self, range_x, range_y):
        os.makedirs(IMAGERY_NPZ_DIR, exist_ok=True)
        for x in range_x:
            for y in range_y:
                jpg = os.path.join(IMAGERY_DIR, '14_{}_{}.jpg'.format(x, y))
                npz = os.path.join(IMAGERY_NPZ_DIR, '14_{}_{}.npz'.format(x, y))
                if not os.path.isfile(npz):
                    util.jpg_to_npz(jpg, npz)

    def getTile(self, coordinates):
        x, y = coordinates
        npz = os.path.join(IMAGERY_NPZ_DIR, '14_{}_{}.npz'.format(x, y))
        return np.load(npz)['arr_0']

    def loadImages(self):
        result = [self.getTile(tile) for tile in self.allTiles]
        self.images = np.array(result)

    def readZipCodesData(self):
        self.zipCodesData = pd.read_csv(
            ZIP_CODES_PATH,
            delimiter=';',
            usecols=['zip', 'latitude', 'longitude'],
        ).dropna(subset=['zip', 'latitude', 'longitude'])

    def readCensusData(self):
        census_data = pd.read_csv(
            CENSUS_PATH,
            usecols=['ZIPCODE', 'N1', 'A02650'],
        ).dropna(subset=['ZIPCODE', 'N1', 'A02650'])

        self.censusData = (
            census_data.groupby('ZIPCODE', as_index=False)[['N1', 'A02650']]
            .sum()
        )
        self.censusByZip = self.censusData.set_index('ZIPCODE')
        self.zipCodesData, valid_zip_codes = filter_zip_coordinate_rows(
            self.zipCodesData,
            self.censusData,
        )
        self.zipCodes = sorted(int(zip_code) for zip_code in valid_zip_codes)

    def createMapping(self):
        zip_codes_to_tiles = build_zip_codes_to_tiles(self.zipCodesData)
        self.mapping = assign_tiles_to_zipcodes(self.allTiles, zip_codes_to_tiles)
        self.mappingSaved = True

    def saveMapping(self, mapping_path=MAPPING_PATH):
        payload = {
            'version': MAPPING_CACHE_VERSION,
            'bounds': {
                'min_x': self.MIN_X,
                'max_x': self.MAX_X,
                'min_y': self.MIN_Y,
                'max_y': self.MAX_Y,
            },
            'mapping': {str(key): value for key, value in self.mapping.items()},
        }
        with open(mapping_path, 'w') as fp:
            json.dump(payload, fp)

    def loadMapping(self, mapping_path=MAPPING_PATH):
        with open(mapping_path) as fp:
            payload = json.load(fp)

        if not isinstance(payload, dict) or 'mapping' not in payload:
            raise ValueError('Legacy mapping cache detected; regenerate mapping.')
        if payload.get('version') != MAPPING_CACHE_VERSION:
            raise ValueError('Unsupported mapping cache version: {}'.format(payload.get('version')))

        bounds = payload.get('bounds', {})
        expected_bounds = {
            'min_x': self.MIN_X,
            'max_x': self.MAX_X,
            'min_y': self.MIN_Y,
            'max_y': self.MAX_Y,
        }
        if bounds != expected_bounds:
            raise ValueError('Mapping cache bounds do not match current grid.')

        self.mapping = defaultdict(
            list,
            {int(key): value for key, value in payload['mapping'].items()},
        )

    def createLabels(self):
        for zip_code, indexes in self.mapping.items():
            if zip_code not in self.censusByZip.index:
                raise KeyError('Missing census data for ZIP {}'.format(zip_code))

            row = self.censusByZip.loc[zip_code]
            total_population = float(row['N1'])
            total_income = float(row['A02650'])
            if not indexes:
                continue

            population = total_population / len(indexes)
            income = total_income / len(indexes)
            for index in indexes:
                self.population[index] += population
                self.income[index] += income

        zero_population = self.population <= 0
        if np.any(zero_population):
            warnings.warn(
                'Encountered tiles with zero population after label generation; assigning zero labels.',
                RuntimeWarning,
            )

        self.labels = np.divide(
            self.income,
            self.population,
            out=np.zeros_like(self.income, dtype=np.float32),
            where=self.population > 0,
        )

    def getLabeledData(self):
        return self.images, self.labels

    def getGroups(self):
        return build_groups_from_mapping(self.mapping, self.getNumImages())

    def getNumImages(self):
        return self.images.shape[0]

    def getNumZipCodes(self):
        return len(self.zipCodes)


def process(X, transform):
    if transform:
        X = [transform(img).unsqueeze(0) for img in X]
        X = torch.cat(X)
    else:
        X = np.array([x.flatten() / 255 for x in X])
    return X


def prepareDataset(model, augment, seed=0):

    mapping = Mapping()
    try:
        mapping.loadMapping()
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        mapping.createMapping()
        mapping.saveMapping()

    mapping.createLabels()
    X, y = mapping.getLabeledData()
    groups = mapping.getGroups()
    X_train, X_test, y_train, y_test, _, _ = grouped_train_test_split(
        X,
        y,
        groups,
        test_size=DEFAULT_TEST_SIZE,
        random_state=seed,
    )

    if model == 'cnn':
        if augment:
            transform_train = transforms.Compose([
                            transforms.Lambda(lambda x: x.astype(np.uint8)),
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ])

            transform_test = transforms.Compose([
                        transforms.Lambda(lambda x: x.astype(np.uint8)),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                    ])

        else:
            transform_train = transforms.Compose([
                            transforms.Lambda(lambda x: x.astype(np.uint8)),
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ])

            transform_test = transforms.Compose([
                        transforms.Lambda(lambda x: x.astype(np.uint8)),
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                    ])

        X_train = process(X_train, transform_train)
        y_train = torch.from_numpy(y_train).float()
        X_test = process(X_test, transform_test)
        y_test = torch.from_numpy(y_test).float()

    else:
        transform_train = None
        transform_test = None
        X_train = process(X_train, transform_train)
        X_test = process(X_test, transform_test)

    return X_train, X_test, y_train, y_test
