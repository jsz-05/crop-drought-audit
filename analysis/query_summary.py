"""Query a county crop-count summary against a bundled weekly drought map."""
from pathlib import Path
import argparse
import json

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

from geo import project, read_shapes

ROOT = Path(__file__).resolve().parents[1]


def query(state='KS', valid_date='20230905', cells=16):
    if cells not in (1, 4, 16):
        raise ValueError('Use 1, 4, or 16 spatial cells per county.')
    state = state.upper()
    features = json.loads((ROOT/'data'/f'counties_{state}.json').read_text())
    summaries = json.loads((ROOT/'data'/f'crop_counts_{state}.json').read_text())
    crop_counts = {f['fips']: f for f in summaries['counties']}
    counties = [(f, shape(f['geometry'])) for f in features['counties']]
    extent = unary_union([g for _, g in counties]).envelope.buffer(10000)
    path = ROOT/'data'/f'usdm_{valid_date}.zip'
    crs, raw = read_shapes(path, 'D1.shp')
    drought = unary_union([p for _, g in raw if
                           (p := project(g, crs, features['crs'])).intersects(extent)])
    rows = []
    for feature, county in counties:
        summary = crop_counts[feature['fips']]
        population = summary['corn_pixels']
        if cells == 1:
            estimate = population * county.intersection(drought).area/county.area
        else:
            n = int(np.sqrt(cells))
            x0, y0, x1, y1 = summary['bounds']
            xs, ys = np.linspace(x0, x1, n+1), np.linspace(y0, y1, n+1)
            estimate = 0.
            for k, count in enumerate(summary[f'counts_{cells}']):
                if not count:
                    continue
                iy, ix = divmod(k, n)
                cell = county.intersection(box(xs[ix], ys[iy], xs[ix+1], ys[iy+1]))
                assert cell.area > 0
                estimate += count * cell.intersection(drought).area/cell.area
        rows.append({'fips':feature['fips'], 'county':feature['county'],
                     'corn_pixels':population, 'fraction':estimate/population})
    total = sum(r['corn_pixels'] for r in rows)
    return {'state':state,'valid_date':valid_date,'cells_per_county':cells,
            'state_exposure_pct':100*sum(r['corn_pixels']*r['fraction'] for r in rows)/total,
            'counties':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', choices=('IA','NE','KS'), default='KS')
    parser.add_argument('--date', default='20230905', help='USDM valid date, YYYYMMDD')
    parser.add_argument('--cells', type=int, choices=(1,4,16), default=16)
    arguments = parser.parse_args()
    result = query(arguments.state, arguments.date, arguments.cells)
    print(json.dumps({k:v for k,v in result.items() if k != 'counties'}, indent=2))
