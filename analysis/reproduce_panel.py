"""Recompute all matched operator counts using only the bundled public data."""
from datetime import date, timedelta
from pathlib import Path
import csv
import json
import time

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from geo import read_shapes, project

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    expected = records(ROOT / 'results' / 'operator_panel_counties.csv')
    expected_states = records(ROOT / 'results' / 'operator_panel_states.csv')
    checks, timings = [], []
    for state in ('IA', 'NE', 'KS'):
        tick = time.perf_counter()
        selected = [r for r in expected if r['state'] == state]
        crop_year = selected[0]['crop_year']
        data = np.load(ROOT / 'data' / f'corn_{state}_{crop_year}.npz')
        grid_shape = tuple(data['shape'])
        corn = np.unpackbits(data['packed'], count=int(np.prod(grid_shape))).reshape(grid_shape).astype(bool)
        transform = Affine(*data['transform'][:6])
        source = json.loads((ROOT / 'data' / f'counties_{state}.json').read_text())
        features = source['counties']
        counties = [shape(f['geometry']) for f in features]
        region = unary_union(counties).envelope.buffer(10000)
        grid = rasterize([(mapping(g), i+1) for i, g in enumerate(counties)],
                         out_shape=grid_shape, transform=transform, dtype='uint16')
        county_ids = grid[corn]
        totals = np.bincount(county_ids, minlength=len(counties)+1)
        assert np.array_equal(totals[1:], [f['corn_pixels'] for f in features])
        for stamp in sorted(set(r['valid_date'] for r in selected)):
            drought_crs, raw = read_shapes(ROOT / 'data' / f'usdm_{stamp}.zip', 'D1.shp')
            drought = unary_union([p for _, g in raw if
                                   (p := project(g, drought_crs, source['crs'])).intersects(region)])
            mask = rasterize([(mapping(drought), 1)], out_shape=grid_shape,
                             transform=transform, dtype='uint8') if not drought.is_empty else np.zeros(grid_shape,dtype='uint8')
            overlap = np.bincount(county_ids[mask[corn].astype(bool)], minlength=len(counties)+1)
            current = {r['fips']: r for r in selected if r['valid_date'] == stamp}
            sums = {m:0. for m in ('pixel','centroid','internal','area')}
            for i, (feature, county) in enumerate(zip(features, counties)):
                ref = current[feature['fips']]
                fractions = {'pixel':float(overlap[i+1]/totals[i+1]),
                             'centroid':float(drought.covers(shape(feature['centroid']))),
                             'internal':float(drought.covers(shape(feature['internal']))),
                             'area':county.intersection(drought).area/county.area}
                assert int(overlap[i+1]) == int(ref['drought_corn_pixels'])
                for method, fraction in fractions.items():
                    difference = abs(fraction-float(ref[method+'_fraction']))
                    assert difference < 1e-9, (state, stamp, feature['fips'], method, difference)
                    sums[method] += int(totals[i+1])*fraction
                checks.append((state, stamp, feature['fips']))
            aggregate = next(r for r in expected_states if r['state']==state and r['valid_date']==stamp)
            for method, total in sums.items():
                value = 100*total/int(totals[1:].sum())
                assert abs(value-float(aggregate[method+'_exposure_pct'])) < 1e-8
        elapsed = time.perf_counter()-tick
        timings.append({'state':state,'seconds':elapsed})
        print(f'{state}: all 13 weeks match ({elapsed:.2f} seconds).', flush=True)
        del corn, grid, county_ids, mask
    result = {'county_week_records_matched':len(checks), 'state_week_records_matched':39,
              'fraction_tolerance':1e-9,'state_percentage_tolerance':1e-8,
              'timings':timings,
              'interpretation':'Deterministic reproduction of stored map comparisons, not uncertainty over future states or dates.'}
    (ROOT / 'results' / 'reproduction_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
