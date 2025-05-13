import pygmt
import numpy as np


regions = dict(np.load('../data/regions.npy', allow_pickle=True).tolist())

stations = [] 
for i in range(len(regions['station_longitudes'])):
    station = {
        'longitudes': regions['station_longitudes'][i],
        'latitudes': regions['station_latitudes'][i],
    }
    stations.append(station)

stations_finally = []
for station in stations:
    if station not in stations_finally:
        stations_finally.append(station)

stations_longitudes = []     
stations_latitudes = []   
for station in stations_finally:
    stations_longitudes.append(station['longitudes'])
    stations_latitudes.append(station['latitudes'])

region = [min(int(np.min(regions['source_longitudes'])-1), int(np.min(stations_longitudes)-1)), max(int(np.max(regions['source_longitudes'])+1), int(np.max(stations_longitudes)+1)), \
              min(int(np.min(regions['source_latitudes'])-1), int(np.min(stations_latitudes)-1)), max(int(np.max(regions['source_latitudes'])+1), int(np.max(stations_latitudes)+1))]

print(region)
fig = pygmt.Figure()

fig.coast(region=region,
    # Set projection to Mercator, and the figure size to 15 centimeters
    projection="M12c",
    # Set the color of the land to light gray
    land="papayawhip", 
    water="lightsteelblue",
    borders="1/0.25p,dimgrey",
    # Display the shorelines and set the pen thickness to 0.5p
    shorelines="0.25p,dimgrey",
    # Set the frame to display annotations and gridlines
    frame="a60f15"
    )
fig.plot(x=stations_longitudes, y=stations_latitudes, style="t0.25c", fill="salmon", pen="black", label='Station Location')
fig.plot(x=regions['source_longitudes'], y=regions['source_latitudes'], style="c0.2c", fill="thistle", pen="black", label='Earthquake Location')
fig.legend(transparency=20, position="JTR+jTR+o0.2c", box="+gwhite+p1p")
fig.savefig('RegionA.png', dpi=600)
