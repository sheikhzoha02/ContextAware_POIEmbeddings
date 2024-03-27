The files 'nyc_category_edge_weights_1.csv', 'nyc_distance_edge_weights.csv' and 'nyc_data.pkl' already added in the shared Data folder in Drive.
Sequence of Commands:
1. python poi_embedding.py
2. python poi_region_embedding.py

In order to run the Downstream tasks
1. python Downstream_tasks/functional_distribution_learning.py
2. python Downstream_tasks/population_density.py

Right now the paths are not relative so you need to change paths manually in all the files and I know right now its too much work but I will try to make paths relative in my future commits.
