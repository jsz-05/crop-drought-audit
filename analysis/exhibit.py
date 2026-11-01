"""Make a native-data figure explaining one weekly aggregation disagreement."""
from datetime import datetime
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from affine import Affine
import numpy as np
from shapely.geometry import shape, box
from shapely.ops import unary_union
from shapely.plotting import plot_polygon

from geo import read_shapes, project

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'pixel': '#222222', 'centroid': '#c65d20', 'area': '#167798'}
NAMES = {'pixel': 'Pixel centers', 'centroid': 'County center', 'area': 'County area'}


def rows(name):
    with (ROOT / 'results' / name).open() as stream:
        return list(csv.DictReader(stream))


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [p for g in geometry.geoms for p in polygons(g)]


def outlines(ax, geometry, **kwargs):
    for polygon in polygons(geometry):
        for ring in [polygon.exterior] + list(polygon.interiors):
            xy = np.asarray(ring.coords)
            ax.plot(xy[:, 0]/1000, xy[:, 1]/1000, **kwargs)


def contributions(state='KS', before='20230829', after='20230905'):
    if (datetime.strptime(after,'%Y%m%d')-datetime.strptime(before,'%Y%m%d')).days != 7:
        raise ValueError('Choose consecutive weekly maps, seven days apart.')
    selected = [r for r in rows('operator_panel_counties.csv') if r['state'] == state]
    left = {r['fips']: r for r in selected if r['valid_date'] == before}
    right = {r['fips']: r for r in selected if r['valid_date'] == after}
    if not left or left.keys() != right.keys():
        raise ValueError('Select two available dates in the same state.')
    population = sum(int(r['corn_pixels']) for r in left.values())
    output = []
    for fips, l in left.items():
        r = right[fips]
        entry = {'fips': fips, 'county': l['county']}
        for method in NAMES:
            entry[method] = (100*int(l['corn_pixels'])/population *
                             (float(r[method+'_fraction'])-float(l[method+'_fraction'])))
        entry['difference'] = entry['centroid']-entry['pixel']
        output.append(entry)
    return sorted(output, key=lambda r: abs(r['difference']), reverse=True)


def make_exhibit(state='KS', before='20230829', after='20230905', output=None):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    output = Path(output) if output else ROOT / 'figures' / f'{state}_{before}_{after}'
    output.parent.mkdir(parents=True, exist_ok=True)
    state_rows = [r for r in rows('operator_panel_states.csv') if r['state'] == state]
    changes = contributions(state, before, after)
    features = json.loads((ROOT/'data'/f'counties_{state}.json').read_text())
    focus = changes[:2]
    geometries = [shape(f['geometry']) for f in features['counties'] if f['fips'] in {r['fips'] for r in focus}]
    focus_geometry = unary_union(geometries)
    xmin, ymin, xmax, ymax = focus_geometry.bounds
    margin = max(xmax-xmin, ymax-ymin)*.08
    limits = (xmin-margin, ymin-margin, xmax+margin, ymax+margin)
    crop_year = state_rows[0]['crop_year']
    source = np.load(ROOT/'data'/f'corn_{state}_{crop_year}.npz')
    transform = Affine(*source['transform'][:6])
    shape_grid = tuple(source['shape'])
    corn = np.unpackbits(source['packed'], count=int(np.prod(shape_grid))).reshape(shape_grid)
    c0 = max(0, int(np.floor((limits[0]-transform.c)/transform.a)))
    c1 = min(shape_grid[1], int(np.ceil((limits[2]-transform.c)/transform.a)))
    r0 = max(0, int(np.floor((limits[3]-transform.f)/transform.e)))
    r1 = min(shape_grid[0], int(np.ceil((limits[1]-transform.f)/transform.e)))
    display_crop = corn[r0:r1, c0:c1]
    extent = np.asarray([transform.c+c0*transform.a, transform.c+c1*transform.a,
                         transform.f+r1*transform.e, transform.f+r0*transform.e])/1000
    del corn
    droughts = {}
    for stamp in (before, after):
        crs, raw = read_shapes(ROOT/'data'/f'usdm_{stamp}.zip', 'D1.shp')
        droughts[stamp] = unary_union([p for _, g in raw if
                                      (p := project(g, crs, features['crs'])).intersects(box(*limits))]).intersection(box(*limits))
    figure = plt.figure(figsize=(10, 7))
    grid = figure.add_gridspec(2, 2, height_ratios=[.85, 1.15], wspace=.32, hspace=.48)
    ax = figure.add_subplot(grid[0, :])
    dates = [datetime.strptime(r['valid_date'], '%Y%m%d') for r in state_rows]
    for method in NAMES:
        ax.plot(dates, [float(r[method+'_exposure_pct']) for r in state_rows],
                color=COLORS[method], lw=1.8, marker='o', ms=3, label=NAMES[method])
    for stamp in (before, after):
        ax.axvline(datetime.strptime(stamp, '%Y%m%d'), color='.6', lw=.7, ls=':')
    ax.set_title(f'(a) {state}: mapped corn in D1+ drought', loc='left', fontweight='bold')
    ax.set_ylabel('Exposure (%)')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.TU, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc='upper right', ncol=3, frameon=False)
    ax.grid(axis='y', color='.9')
    ax = figure.add_subplot(grid[1, 0])
    ax.imshow(display_crop, extent=extent, origin='upper', interpolation='nearest',
              cmap=ListedColormap(['#ffffff', '#94aa7d']), rasterized=True)
    # Scale the polygon to the plot's kilometre coordinates before shading D1+.
    from shapely.affinity import scale
    for stamp, color in ((before, '#287da0'), (after, '#b65130')):
        plot_polygon(scale(droughts[stamp], xfact=.001, yfact=.001, origin=(0,0)),
                     ax=ax, add_points=False, facecolor=color, alpha=.10, edgecolor='none')
    outlines(ax, droughts[before], color='#287da0', lw=1.8, ls='--')
    outlines(ax, droughts[after], color='#b65130', lw=1.8)
    for feature in features['counties']:
        geom = shape(feature['geometry'])
        if geom.intersects(focus_geometry):
            outlines(ax, geom, color='.2', lw=.7)
            center = shape(feature['centroid'])
            if feature['fips'] in {r['fips'] for r in focus}:
                ax.plot(center.x/1000, center.y/1000, marker='x', color='.1', ms=6, mew=1.4)
                ax.annotate(feature['county'], (center.x/1000, center.y/1000),
                            xytext=(0, -14), textcoords='offset points', ha='center',
                            bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.85, 'pad':1.5})
    ax.set_xlim(limits[0]/1000, limits[2]/1000)
    ax.set_ylim(limits[1]/1000, limits[3]/1000)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(b) Counties driving the disagreement', loc='left', fontweight='bold', pad=8)
    ax.legend(handles=[Line2D([0],[0],color='#287da0',ls='--',label=datetime.strptime(before,'%Y%m%d').strftime('D1+ %b %d, %Y')),
                       Line2D([0],[0],color='#b65130',label=datetime.strptime(after,'%Y%m%d').strftime('D1+ %b %d, %Y')),
                       Line2D([0],[0],color='#94aa7d',lw=5,label=f'{crop_year} corn'),
                       Line2D([0],[0],color='.1',lw=0,marker='x',label='County centroid')],
              loc='lower center', bbox_to_anchor=(.5,-.23), ncol=2, frameon=False, fontsize=8)
    ax = figure.add_subplot(grid[1, 1])
    bars = focus + [{'county':'All other counties', **{m:sum(r[m] for r in changes[2:]) for m in NAMES}}]
    y = np.arange(len(bars))
    for j, method in enumerate(('centroid','area','pixel')):
        ax.barh(y+(j-1)*.23, [r[method] for r in bars], height=.21, label=NAMES[method], color=COLORS[method])
    ax.axvline(0, color='.5', lw=.7)
    ax.set_yticks(y, [r['county'] for r in bars])
    ax.invert_yaxis()
    ax.set_xlabel('Contribution to weekly change (percentage points)')
    ax.set_title('(c) Why the statewide trend changes sign', loc='left', fontweight='bold', pad=8)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.grid(axis='x', color='.9')
    figure.subplots_adjust(left=.08, right=.98, top=.95, bottom=.14)
    for extension in ('pdf', 'png'):
        figure.savefig(str(output)+'.'+extension, dpi=240, bbox_inches='tight')
    plt.close(figure)
    result = {'state':state, 'before':before, 'after':after,
              'weekly_change_pp':{m:sum(r[m] for r in changes) for m in NAMES},
              'county_contributions':changes}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    result = make_exhibit()
    print(json.dumps({k:v for k,v in result.items() if k != 'county_contributions'},indent=2))
