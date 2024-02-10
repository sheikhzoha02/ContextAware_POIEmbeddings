import math
import os
import pickle as pkl

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import KDTree, Delaunay
from shapely.geometry import Point
from tqdm import tqdm
import sys
from shapely.wkt import loads

class NYCCensusTract:
    def __init__(self):
        self.region_path = '/home/iailab41/sheikhz0/Data/pop_nrw_region_boundary_with_index.csv'
        self.landuse_in_path = '/home/iailab41/sheikhz0/Data/landuse_nrw.csv'
        self.landuse_region = '/home/iailab41/sheikhz0/Data/landuse_region_nrw.csv'
        self.out_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Module/downstream_region_nrw.pkl'
        self.merge_dict = {
            '1': 0,  # One &Two Family Buildings
            '2': 0,  # Multi-Family Walk-Up Buildings
            '3': 0,  # Multi-Family Elevator Buildings
            '4': 1,  # Mixed Residential & Commercial Buildings
            '5': 1,  # Commercial & Office Buildings
            '6': 2,  # Industrial & Manufacturing
            '7': 3,  # Transportation & Utility
            '8': 3,  # Public Facilities & Institutions
            '9': 4,  # Open Space & Outdoor Recreation
            '10': 3,  # Parking Facilities
            '11': 4,  # Vacant Land
        }


    def get(self, force=False):
        if os.path.exists(self.out_path) and not force:
            print('Loading NRW census tract data from disk...')
            with open(self.out_path, 'rb') as f:
                return pkl.load(f)
        # load region shapefile
        region_shapefile = pd.read_csv(self.region_path)
        regions = []
        region_dict = {}
        for index, row in region_shapefile.iterrows():
            name = row['region_id']
            region = {
                'name': name,
                'shape': row['geometry'],
                'land_use': [0.0] * 5,
                'population':row['pop']
            }
            region_dict[name] = region
            regions.append(region)
        # load land use data
        print('Loading land use...')
        landuse_shapefile = pd.read_csv(self.landuse_in_path)
        print('Loading land use region...')
        landuse_region_shapefile = pd.read_csv(self.landuse_region)
        print('Aggregating land use...')
        for index, row in tqdm(landuse_shapefile.iterrows(), total=len(landuse_shapefile)):
            results = landuse_region_shapefile.loc[landuse_region_shapefile['osm_id'] == row['osm_id'], 'region_id'].values
            if len(results) > 0:
                region_id = results[0]
                if region_id not in region_dict:
                    print('Census tract {} not found for landuse {}'.format(region_id, index))
                    continue
                label = str(row['LandUse'])
                if label is None:
                    continue
                geometry_area = loads(row['geometry'])
                region_dict[region_id]['land_use'][self.merge_dict[label]] += geometry_area.area
        # normalize land use
        for idx, region in enumerate(regions):
            region['population'] = region_dict[region['name']]['population']
            total_area = sum(region['land_use'])
            if total_area == 0:
                print('Land use area is 0 for census tract {}'.format(idx))
                continue
            region['land_use'] = [x / total_area for x in region['land_use']]
        # save
        with open(self.out_path, 'wb') as f:
            pkl.dump(regions, f)
        return regions

if __name__ == '__main__':
    os.chdir('../')
    NYCCensusTract().get(force=True)
