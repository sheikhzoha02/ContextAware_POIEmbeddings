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

class NYCCensusTract:
    def __init__(self):
        self.region_path = '/home/iailab41/sheikhz0/Data/region_nyc_with_index.csv'
        self.landuse_in_path = '/home/iailab41/sheikhz0/RegionDCL/data/projected/NYC/landuse/landuse.shp'
        self.population_in_path = '/home/iailab41/sheikhz0/RegionDCL/data/projected/NYC/population/population_cleaned.csv'
        self.out_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Module/downstream_region_nyc.pkl'
        self.merge_dict = {
            '01': 0,  # One &Two Family Buildings
            '02': 0,  # Multi-Family Walk-Up Buildings
            '03': 0,  # Multi-Family Elevator Buildings
            '04': 1,  # Mixed Residential & Commercial Buildings
            '05': 1,  # Commercial & Office Buildings
            '06': 2,  # Industrial & Manufacturing
            '07': 3,  # Transportation & Utility
            '08': 3,  # Public Facilities & Institutions
            '09': 4,  # Open Space & Outdoor Recreation
            '10': 3,  # Parking Facilities
            '11': 4,  # Vacant Land
        }


    def get(self, force=False):
        if os.path.exists(self.out_path) and not force:
            print('Loading NYC census tract data from disk...')
            with open(self.out_path, 'rb') as f:
                return pkl.load(f)
        # load region shapefile
        region_shapefile = pd.read_csv(self.region_path)
        regions = []
        region_dict = {}
        for index, row in region_shapefile.iterrows():
            name = str(row['BoroCT2020'])
            region = {
                'name': name,
                'shape': row['geometry'],
                'land_use': [0.0] * 5,
                'population': 0,
                'region_id':row['region_id']
            }
            region_dict[name] = region
            regions.append(region)
        # load population data
        with open(self.population_in_path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                if line[0] not in region_dict:
                    print('Population for census tract {} not found in the region shapefile'.format(line[0]))
                    continue
                region_dict[line[0]]['population'] = line[1]
        # load land use data
        print('Loading land use...')
        landuse_shapefile = gpd.read_file(self.landuse_in_path)
        print('Aggregating land use...')
        for index, row in tqdm(landuse_shapefile.iterrows(), total=len(landuse_shapefile)):
            BCT2020 = row['BCT2020']
            if BCT2020 not in region_dict:
                print('Census tract {} not found for landuse {}'.format(BCT2020, index))
                continue
            label = row['LandUse']
            if label is None:
                continue
            region_dict[BCT2020]['land_use'][self.merge_dict[label]] += row['geometry'].area
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

    def pack_poi_data(self):
        with open(self.out_path, 'rb') as f:
            regions = pkl.load(f)
        # load poi
        poi_in_path = '/home/iailab41/sheikhz0/RegionDCL/data/projected/NYC/poi/poi.shp'
        pois_shapefile = gpd.read_file(poi_in_path)
        pois = []
        for index, poi_row in tqdm(pois_shapefile.iterrows(), total=pois_shapefile.shape[0]):
            output = {}
            # process point
            output['x'] = poi_row['geometry'].x
            output['y'] = poi_row['geometry'].y
            output['code'] = poi_row['code']
            output['fclass'] = poi_row['fclass']
            pois.append(output)
        # turn code & fclass into numbers
        code_dict = {}
        fclass_dict = {}
        for poi in pois:
            if poi['code'] not in code_dict:
                code_dict[poi['code']] = len(code_dict)
            if poi['fclass'] not in fclass_dict:
                fclass_dict[poi['fclass']] = len(fclass_dict)
        for poi in pois:
            poi['code'] = code_dict[poi['code']]
            poi['fclass'] = fclass_dict[poi['fclass']]
        # aggregate poi data
        print('Poi number:', len(pois))
        poi_loc = [[poi['x'], poi['y']] for poi in pois]
        poi_tree = KDTree(poi_loc)
        print('Aggregating poi...')
        count_no_point = 0
        for idx, region in tqdm(enumerate(regions)):
            print(region)
            region_shape = region['shape']
            print(region_shape)
            region.pop('shape')
            # calculate region diameter
            dx = region_shape.bounds[2] - region_shape.bounds[0]
            dy = region_shape.bounds[3] - region_shape.bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy) / 2
            # find poi in the region
            poi_index = poi_tree.query_ball_point(region_shape.centroid, diameter)
            for index in poi_index:
                if region_shape.contains(Point(poi_loc[index])):
                    pois[index]['region'] = idx
        print('Region without poi:', count_no_point)
        poi_within_region = [poi for poi in pois if 'region' in poi]
        # save to csv
        with open('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Module/poi_for_baselines.csv', 'w') as f:
            f.write(' ,index,X,Y,FirstLevel,SecondLevel,PoiID,ZoneID')
            for idx, poi in enumerate(poi_within_region):
                f.write('\n')
                f.write(','.join([str(idx), str(idx), str(poi['x']), str(poi['y']), str(poi['code']), str(poi['fclass']), str(poi['osm_id']), str(poi['region'])]))


if __name__ == '__main__':
    os.chdir('../')
    NYCCensusTract().get(force=True)
#    NYCCensusTract().pack_poi_data()
