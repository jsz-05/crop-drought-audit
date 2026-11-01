"""Extract the bundled corn masks and county geometry from public source files."""
from pathlib import Path
import argparse
import json

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping, shape

from geo import project, read_shapes

ROOT = Path(__file__).resolve().parents[1]


def prepare(county_zip, crop_files, output, verify=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_crs, all_counties = read_shapes(county_zip, '.shp')
    report = []
    for state, fips, year in [('IA','19',2021),('NE','31',2022),('KS','20',2022)]:
        records = sorted([(p,g) for p,g in all_counties if p['STATEFP']==fips],
                         key=lambda item:item[0]['GEOID'])
        with rasterio.open(crop_files[state]) as raster:
            grid, crs, grid_shape = raster.transform, raster.crs, raster.shape
            corn = raster.read(1)==1
        counties = [(p,project(g,source_crs,crs)) for p,g in records]
        with rasterio.Env(GDAL_CACHEMAX=128*1024*1024):
            county_grid = rasterize([(mapping(g),i+1) for i,(_,g) in enumerate(counties)],
                                    out_shape=grid_shape, transform=grid, dtype='uint16')
        totals = np.bincount(county_grid[corn],minlength=len(counties)+1)
        geometry = {'crs':str(crs),'counties':[]}
        for i,(properties,county) in enumerate(counties):
            internal = shape({'type':'Point','coordinates':
                              [float(properties['INTPTLON']),float(properties['INTPTLAT'])]})
            internal = project(internal,source_crs,crs)
            geometry['counties'].append({
                'fips':properties['GEOID'],'county':properties['NAME'],
                'geometry':mapping(county),'centroid':mapping(county.centroid),
                'internal':mapping(internal),'corn_pixels':int(totals[i+1])})
        name = f'corn_{state}_{year}.npz'
        packed = np.packbits(corn)
        np.savez_compressed(output/name, packed=packed, shape=np.asarray(grid_shape),
                            transform=np.asarray(tuple(grid)), crs=str(crs))
        serialized = json.dumps(geometry)
        (output/f'counties_{state}.json').write_text(serialized)
        if verify:
            bundle = np.load(ROOT/'data'/name)
            assert np.array_equal(packed,bundle['packed'])
            assert np.array_equal(np.asarray(grid_shape),bundle['shape'])
            assert np.array_equal(np.asarray(tuple(grid)),bundle['transform'])
            assert str(crs)==str(bundle['crs'])
            assert json.loads(serialized)==json.loads((ROOT/'data'/f'counties_{state}.json').read_text())
        report.append({'state':state,'crop_year':year,'counties':len(counties),
                       'corn_pixels':int(totals[1:].sum()),'unassigned_corn_pixels':int(totals[0]),
                       'bundle_equal':verify})
        print('Prepared',state,'and verified bundled values' if verify else '',flush=True)
        del corn,county_grid
    (output/'preparation_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--county-zip',type=Path,required=True)
    for state in ('ia','ne','ks'):
        parser.add_argument(f'--{state}-cdl',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'tmp'/'rebuilt-data')
    parser.add_argument('--verify-bundled',action='store_true')
    args=parser.parse_args()
    report=prepare(args.county_zip,{'IA':args.ia_cdl,'NE':args.ne_cdl,'KS':args.ks_cdl},
                   args.output,args.verify_bundled)
    print(json.dumps(report,indent=2))
