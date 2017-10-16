import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import urllib, os
import pandas as pd

data = pd.read_csv('Addis_data.csv')
data = data.fillna(-1)
lats = data['bl_bi24latitude'].values
lons = data['bl_bi24longitude'].values
s = [5]*len(lats)
# draw map with markers for float locations
m = Basemap(projection='cyl',resolution='f',llcrnrlon=35.617809,llcrnrlat=6.911369,urcrnrlon=45.58559,urcrnrlat=12.217216)
# x, y = m(*np.meshgrid(lons,lats))
# m.drawmapboundary()
# m.fillcontinents()
# m.etopo(zorder=0)
m.shadedrelief(zorder=0)
m.scatter(lons, lats, latlon=True, marker='o', color='k', s=s)
plt.show()