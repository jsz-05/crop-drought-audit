"""Exploratory, same-input comparisons of compact county crop summaries."""
from pathlib import Path
import csv
import json

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box, shape, mapping
from shapely.ops import unary_union

from geo import read_shapes, project

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def polygon_covariance(geom):
    """Exact second central moment of a uniform polygon, in projected units."""
    origin = np.asarray(geom.centroid.coords[0])
    area, second = 0., np.zeros((2, 2))
    polygons = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    for polygon in polygons:
        for ring, sign in [(polygon.exterior, 1)] + [(r, -1) for r in polygon.interiors]:
            xy = np.asarray(ring.coords) - origin
            x, y, xx, yy = xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1]
            cross = x * yy - xx * y
            orientation = np.sign(cross.sum())
            cross *= sign * orientation
            area += cross.sum() / 2
            second[0, 0] += np.sum(cross * (x*x + x*xx + xx*xx)) / 12
            second[1, 1] += np.sum(cross * (y*y + y*yy + yy*yy)) / 12
            second[0, 1] += np.sum(cross * (2*x*y + x*yy + xx*y + 2*xx*yy)) / 24
    second[1, 0] = second[0, 1]
    assert abs(area / geom.area - 1) < 1e-7
    return second / area


def write_csv(name, rows):
    with (ROOT / 'results' / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert np.allclose(polygon_covariance(box(-1, -2, 1, 2)), np.diag([1/3, 4/3]))
    with (ROOT / 'results' / 'operator_panel_counties.csv').open() as stream:
        records = list(csv.DictReader(stream))
    outputs = []
    for state, year in [('IA', 2021), ('NE', 2022), ('KS', 2022)]:
        source = np.load(DATA / f'corn_{state}_{year}.npz')
        shape_grid = tuple(source['shape'])
        corn = np.unpackbits(source['packed'], count=int(np.prod(shape_grid))).reshape(shape_grid).astype(bool)
        transform = Affine(*source['transform'][:6])
        features = json.loads((DATA / f'counties_{state}.json').read_text())
        geometries = [shape(f['geometry']) for f in features['counties']]
        grid = rasterize([(mapping(g), i + 1) for i, g in enumerate(geometries)],
                         out_shape=shape_grid, transform=transform, dtype='uint16')
        totals = np.zeros(len(geometries) + 1)
        xsum, ysum = np.zeros_like(totals), np.zeros_like(totals)
        histograms = {n: np.zeros((len(geometries), n*n)) for n in (2, 4)}
        bounds = np.asarray([g.bounds for g in geometries])
        for start in range(0, shape_grid[0], 256):
            rr, cc = np.nonzero(corn[start:start+256])
            rr += start
            ids = grid[rr, cc]
            keep = ids > 0
            ids, rr, cc = ids[keep], rr[keep], cc[keep]
            x = transform.c + (cc + .5) * transform.a
            y = transform.f + (rr + .5) * transform.e
            totals += np.bincount(ids, minlength=len(totals))
            xsum += np.bincount(ids, weights=x, minlength=len(totals))
            ysum += np.bincount(ids, weights=y, minlength=len(totals))
            b = bounds[ids-1]
            for n in histograms:
                ix = np.clip(((x-b[:, 0])/(b[:, 2]-b[:, 0])*n).astype(int), 0, n-1)
                iy = np.clip(((y-b[:, 1])/(b[:, 3]-b[:, 1])*n).astype(int), 0, n-1)
                cell = (ids-1)*n*n + iy*n + ix
                histograms[n] += np.bincount(cell, minlength=len(geometries)*n*n).reshape(-1, n*n)
        crop_centers = np.column_stack([xsum[1:] / totals[1:], ysum[1:] / totals[1:]])
        assert np.array_equal(totals[1:], [f['corn_pixels'] for f in features['counties']])
        for histogram in histograms.values():
            assert np.array_equal(histogram.sum(axis=1), totals[1:])
        saved_summary = {
            'state': state, 'crop_year': year, 'crs': features['crs'],
            'definition': 'Regular 2x2 or 4x4 grid of each projected county bounding box, clipped to the county. Counts use native corn-cell centers assigned to that county.',
            'counties': [
                {'fips': feature['fips'], 'county': feature['county'],
                 'corn_pixels': int(totals[i+1]), 'bounds': bounds[i].tolist(),
                 'crop_centroid': crop_centers[i].tolist(),
                 'counts_4': histograms[2][i].astype(int).tolist(),
                 'counts_16': histograms[4][i].astype(int).tolist()}
                for i, feature in enumerate(features['counties'])]}
        (DATA / f'crop_counts_{state}.json').write_text(json.dumps(saved_summary, indent=2)+'\n')
        cells = {}
        for n in histograms:
            cells[n] = []
            for geom, b in zip(geometries, bounds):
                xs, ys = np.linspace(b[0], b[2], n+1), np.linspace(b[1], b[3], n+1)
                cells[n].append([geom.intersection(box(xs[ix], ys[iy], xs[ix+1], ys[iy+1]))
                                 for iy in range(n) for ix in range(n)])
        covariances = [polygon_covariance(g) for g in geometries]
        selected = [r for r in records if r['state'] == state]
        drought_cache = {}
        state_envelope = unary_union(geometries).envelope.buffer(10000)
        for stamp in sorted(set(r['valid_date'] for r in selected)):
            crs, polys = read_shapes(DATA / f'usdm_{stamp}.zip', 'D1.shp')
            # Project the full cumulative D1+ layer to keep the feature extraction simple.
            drought_cache[stamp] = unary_union([p for _, g in polys
                                               if (p := project(g, crs, features['crs'])).intersects(state_envelope)])
        by_fips = {f['fips']: i for i, f in enumerate(features['counties'])}
        for row in selected:
            i, drought = by_fips[row['fips']], drought_cache[row['valid_date']]
            geom = geometries[i]
            intersection = geom.intersection(drought)
            area = intersection.area / geom.area
            term = 0.
            if not intersection.is_empty and area > 0:
                land_center = np.asarray(geom.centroid.coords[0])
                wet_center = np.asarray(intersection.centroid.coords[0])
                term = area * (wet_center-land_center) @ np.linalg.solve(
                    covariances[i], crop_centers[i]-land_center)
            out = {k: row[k] for k in ('state', 'valid_date', 'fips', 'county')}
            out.update(corn_pixels=int(row['corn_pixels']), reference=float(row['pixel_fraction']),
                       area=area, moment_term=float(term))
            for n, histogram in histograms.items():
                fraction = 0.
                for cell, count in zip(cells[n][i], histogram[i]):
                    if count and cell.area:
                        fraction += count * cell.intersection(drought).area / cell.area
                out[f'histogram_{n*n}'] = fraction / totals[i+1]
            outputs.append(out)
        print('Finished spatial summaries:', state, flush=True)
        del corn, grid
    development = [r for r in outputs if r['state'] == 'IA']
    numerator = sum(r['corn_pixels']*r['moment_term']*(r['reference']-r['area']) for r in development)
    denominator = sum(r['corn_pixels']*r['moment_term']**2 for r in development)
    coefficient = float(np.clip(numerator/denominator, 0, 2)) if denominator else 0.
    for row in outputs:
        row['moment_fixed'] = float(np.clip(row['area'] + row['moment_term'], 0, 1))
        row['moment_fitted'] = float(np.clip(row['area'] + coefficient*row['moment_term'], 0, 1))
    write_csv('spatial_summary_counties.csv', outputs)
    summary = []
    for state in ('IA', 'NE', 'KS'):
        selected = [r for r in outputs if r['state'] == state]
        for method in ('area', 'moment_fixed', 'moment_fitted', 'histogram_4', 'histogram_16'):
            abs_errors, signed_errors = [], []
            for stamp in sorted(set(r['valid_date'] for r in selected)):
                rows = [r for r in selected if r['valid_date'] == stamp]
                weights = np.asarray([r['corn_pixels'] for r in rows], float)
                errors = np.asarray([r[method]-r['reference'] for r in rows])
                abs_errors.append(float(np.average(np.abs(errors), weights=weights)*100))
                signed_errors.append(float(np.average(errors, weights=weights)*100))
            summary.append({'state': state, 'method': method,
                            'county_weighted_mae_pp': float(np.mean(abs_errors)),
                            'state_mae_pp': float(np.mean(np.abs(signed_errors))),
                            'fitted_lambda_iowa_only': coefficient})
    write_csv('spatial_summary_results.csv', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
