"""Read public zipped shapefiles and project their geometries."""
import io
import zipfile

import shapefile
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping


def read_shapes(path, suffix):
    with zipfile.ZipFile(path) as archive:
        stem = next(n[:-4] for n in archive.namelist() if n.endswith(suffix))
        reader = shapefile.Reader(shp=io.BytesIO(archive.read(stem+'.shp')),
                                 shx=io.BytesIO(archive.read(stem+'.shx')),
                                 dbf=io.BytesIO(archive.read(stem+'.dbf')))
        names = [f[0] for f in reader.fields[1:]]
        crs = CRS.from_wkt(archive.read(stem+'.prj').decode())
        rows = [(dict(zip(names,item.record)),shape(item.shape.__geo_interface__))
                for item in reader.iterShapeRecords()]
    return crs, rows


def project(geometry, source_crs, target_crs):
    return shape(transform_geom(source_crs,target_crs,mapping(geometry)))
