"""Check the highlighted transition by point-in-polygon, without rasterization."""
from pathlib import Path
import csv
import json

import numpy as np
from affine import Affine
from shapely import intersects_xy, prepare
from shapely.geometry import shape
from shapely.ops import unary_union

from geo import read_shapes, project

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / 'data'
    archive = np.load(data / 'corn_KS_2022.npz')
    grid_shape = tuple(archive['shape'])
    corn = np.unpackbits(archive['packed'], count=int(np.prod(grid_shape))).reshape(grid_shape).astype(bool)
    transform = Affine(*archive['transform'][:6])
    features = json.loads((data / 'counties_KS.json').read_text())
    counties = [(f, shape(f['geometry'])) for f in features['counties']]
    state = unary_union([g for _, g in counties])
    droughts = {}
    for stamp in ('20230829', '20230905'):
        crs, raw = read_shapes(data / f'usdm_{stamp}.zip', 'D1.shp')
        droughts[stamp] = unary_union([p for _, g in raw
                                      if (p := project(g, crs, features['crs'])).intersects(state.envelope)])
    with (ROOT / 'results' / 'operator_panel_counties.csv').open() as stream:
        original = {(r['fips'], r['valid_date']): r for r in csv.DictReader(stream)
                    if r['state'] == 'KS' and r['valid_date'] in droughts}
    checks = []
    for feature, geom in counties:
        xmin, ymin, xmax, ymax = geom.bounds
        c0 = max(0, int(np.floor((xmin-transform.c)/transform.a)))
        c1 = min(grid_shape[1], int(np.ceil((xmax-transform.c)/transform.a)))
        r0 = max(0, int(np.floor((ymax-transform.f)/transform.e)))
        r1 = min(grid_shape[0], int(np.ceil((ymin-transform.f)/transform.e)))
        rr, cc = np.nonzero(corn[r0:r1, c0:c1])
        x = transform.c + (cc+c0+.5)*transform.a
        y = transform.f + (rr+r0+.5)*transform.e
        prepare(geom)
        inside = intersects_xy(geom, x, y)
        x, y = x[inside], y[inside]
        for stamp, drought in droughts.items():
            intersection = geom.intersection(drought)
            prepare(intersection)
            count = int(intersects_xy(intersection, x, y).sum())
            expected = original[(feature['fips'], stamp)]
            checks.append({'county': feature['county'], 'fips': feature['fips'], 'valid_date': stamp,
                           'corn_point_count': len(x), 'corn_raster_count': int(expected['corn_pixels']),
                           'drought_point_count': count, 'drought_raster_count': int(expected['drought_corn_pixels'])})
    totals = {}
    for stamp in droughts:
        rows = [r for r in checks if r['valid_date'] == stamp]
        totals[stamp] = {'point_count': sum(r['drought_point_count'] for r in rows),
                         'raster_count': sum(r['drought_raster_count'] for r in rows),
                         'corn_point_count': sum(r['corn_point_count'] for r in rows),
                         'corn_raster_count': sum(r['corn_raster_count'] for r in rows)}
        totals[stamp]['point_exposure_pct'] = 100*totals[stamp]['point_count']/totals[stamp]['corn_point_count']
        totals[stamp]['raster_exposure_pct'] = 100*totals[stamp]['raster_count']/totals[stamp]['corn_raster_count']
    result = {'method': 'GEOS vector point-in-polygon for every mapped corn cell in all 105 Kansas county windows; no GDAL rasterization in this check.',
              'county_dates_checked': len(checks),
              'all_county_corn_counts_match': all(r['corn_point_count'] == r['corn_raster_count'] for r in checks),
              'all_county_drought_counts_match': all(r['drought_point_count'] == r['drought_raster_count'] for r in checks),
              'state_totals': totals, 'county_results': checks}
    (ROOT / 'results' / 'independent_geometry_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'county_results'}, indent=2), flush=True)
    assert result['all_county_corn_counts_match']
    assert result['all_county_drought_counts_match']


if __name__ == '__main__':
    main()
