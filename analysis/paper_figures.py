"""Publication figures at the final IEEE two-column width, from the map data."""
from datetime import datetime
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from shapely.affinity import scale
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.plotting import plot_polygon

from exhibit import COLORS, NAMES, contributions, outlines, rows
from geo import project, read_shapes

ROOT = Path(__file__).resolve().parents[1]


def save(figure, name):
    for extension in ('pdf','png'):
        figure.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=300)
    plt.close(figure)


def reversal():
    state, before, after = 'KS', '20230829', '20230905'
    features=json.loads((ROOT/'data/counties_KS.json').read_text())
    focus=[f for f in features['counties'] if f['county'] in ('Finney','Gray')]
    geometries=[shape(f['geometry']) for f in focus]
    extent=unary_union(geometries).bounds
    xmin,ymin,xmax,ymax=extent
    margin=(ymax-ymin)*.06
    limits=(xmin-margin,ymin-margin,xmax+margin,ymax+margin)
    data=np.load(ROOT/'data/corn_KS_2022.npz')
    grid_shape=tuple(data['shape'])
    transform=Affine(*data['transform'][:6])
    corn=np.unpackbits(data['packed'],count=int(np.prod(grid_shape))).reshape(grid_shape)
    c0=max(0,int(np.floor((limits[0]-transform.c)/transform.a)))
    c1=min(grid_shape[1],int(np.ceil((limits[2]-transform.c)/transform.a)))
    r0=max(0,int(np.floor((limits[3]-transform.f)/transform.e)))
    r1=min(grid_shape[0],int(np.ceil((limits[1]-transform.f)/transform.e)))
    window=corn[r0:r1,c0:c1]
    raster_extent=np.asarray([transform.c+c0*transform.a,transform.c+c1*transform.a,
                              transform.f+r1*transform.e,transform.f+r0*transform.e])/1000
    droughts={}
    for stamp in (before,after):
        crs,raw=read_shapes(ROOT/'data'/f'usdm_{stamp}.zip','D1.shp')
        droughts[stamp]=unary_union([p for _,g in raw if
                                    (p:=project(g,crs,features['crs'])).intersects(box(*limits))]).intersection(box(*limits))
    figure=plt.figure(figsize=(7.16,3.02))
    grid=figure.add_gridspec(1,3,width_ratios=[1.08,1.04,1.06],wspace=.64,
                            left=.015,right=.985,bottom=.28,top=.89)
    ax=figure.add_subplot(grid[0,0])
    ax.imshow(window,origin='upper',extent=raster_extent,interpolation='nearest',
              cmap=ListedColormap(['white','#91a87c']),rasterized=True)
    for stamp,color,style in [(before,'#287da0','--'),(after,'#b65130','-')]:
        plot_polygon(scale(droughts[stamp],xfact=.001,yfact=.001,origin=(0,0)),ax=ax,
                     add_points=False,facecolor=color,alpha=.10,edgecolor='none')
        outlines(ax,droughts[stamp],color=color,lw=1.2,ls=style)
    for feature in features['counties']:
        geom=shape(feature['geometry'])
        if geom.intersects(box(*limits)):
            outlines(ax,geom,color='.25',lw=.55)
    for feature in focus:
        center=shape(feature['centroid'])
        ax.plot(center.x/1000,center.y/1000,'x',color='.1',ms=5,mew=1.1)
        ax.annotate(feature['county'],(center.x/1000,center.y/1000),xytext=(0,-12),
                    textcoords='offset points',ha='center',fontsize=8,
                    bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':.7})
    ax.set_xlim(limits[0]/1000,limits[2]/1000)
    ax.set_ylim(limits[1]/1000,limits[3]/1000)
    ax.set_xticks([]); ax.set_yticks([])
    bar_x=(limits[0]+(limits[2]-limits[0])*.07)/1000
    bar_y=(limits[1]+(limits[3]-limits[1])*.06)/1000
    ax.plot([bar_x,bar_x+20],[bar_y,bar_y],color='.15',lw=2)
    ax.text(bar_x+10,bar_y+2,'20 km',ha='center',va='bottom',fontsize=8,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':.5})
    ax.set_title('(a) Crop and drought boundaries',fontsize=8,loc='left',pad=7)
    map_axis=ax
    ax=figure.add_subplot(grid[0,1])
    exposure_axis=ax
    panel=[r for r in rows('operator_panel_states.csv') if r['state']=='KS' and '20230815'<=r['valid_date']<='20230912']
    dates=[datetime.strptime(r['valid_date'],'%Y%m%d') for r in panel]
    for method in ('centroid','area','pixel'):
        ax.plot(dates,[float(r[method+'_exposure_pct']) for r in panel],lw=1.3,
                marker='o',ms=2.8,color=COLORS[method],label=NAMES[method])
    for stamp in (before,after):
        ax.axvline(datetime.strptime(stamp,'%Y%m%d'),color='.55',lw=.6,ls=':')
    ax.set_xticks([dates[i] for i in (0,2,3,4)],['Aug 15','Aug 29','Sep 5','Sep 12'])
    ax.tick_params(axis='x',rotation=40)
    ax.set_ylabel('Mapped corn in D1+ (%)',labelpad=3)
    ax.set_title('(b) Kansas exposure, 2023',fontsize=8,loc='left',pad=7)
    ax.grid(axis='y',color='.9',lw=.5)
    ax=figure.add_subplot(grid[0,2])
    changed=contributions()
    focus_rows=[next(r for r in changed if r['county']==name) for name in ('Finney','Gray')]
    other={'county':'Other',**{m:sum(r[m] for r in changed if r['county'] not in ('Finney','Gray')) for m in NAMES}}
    groups=focus_rows+[other]
    for j,method in enumerate(('centroid','area','pixel')):
        ax.barh(np.arange(3)+(j-1)*.23,[r[method] for r in groups],height=.21,color=COLORS[method])
    ax.set_yticks(range(3),[r['county'] for r in groups])
    ax.invert_yaxis()
    ax.axvline(0,color='.4',lw=.6)
    ax.set_xticks([-2,0,2])
    ax.set_xlabel('Weekly contribution (pp)',labelpad=3)
    ax.grid(axis='x',color='.9',lw=.5)
    ax.set_title('(c) Contributions to the change',fontsize=8,loc='left',pad=7)
    # Give the map dates and the two charts a shared, separate legend row.
    figure.canvas.draw()
    # Use the map's empty lower margin while preserving its geographic aspect.
    map_box=map_axis.get_position()
    map_growth=1.25
    map_axis.set_position([map_box.x0,map_box.y1-map_growth*map_box.height,
                           map_growth*map_box.width,map_growth*map_box.height])
    figure.canvas.draw()
    map_center=(map_axis.get_position().x0+map_axis.get_position().x1)/2
    chart_center=(exposure_axis.get_position().x0+ax.get_position().x1)/2
    figure.legend(handles=[Line2D([0],[0],color='#287da0',ls='--',label='Aug 29'),
                           Line2D([0],[0],color='#b65130',label='Sep 5')],
                  loc='center',bbox_to_anchor=(map_center,.065),ncol=2,fontsize=8,
                  frameon=False,columnspacing=.8,handlelength=1.7,handletextpad=.4)
    figure.legend(handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=NAMES[m]) for m in ('centroid','area','pixel')],
                  loc='center',bbox_to_anchor=(chart_center,.065),ncol=3,frameon=False,
                  fontsize=8,columnspacing=1.1,handlelength=1.6,handletextpad=.4)
    save(figure,'kansas-reversal')


def summaries():
    results=rows('spatial_summary_results.csv')
    figure,axes=plt.subplots(1,2,figsize=(7.16,1.96))
    methods=[('area','County area','#666666'),('histogram_4','4 crop counts','#e5a040'),
             ('histogram_16','16 crop counts','#167798')]
    for ax,metric,title in zip(axes,['county_weighted_mae_pp','state_mae_pp'],
                                ['(a) Error within counties','(b) Error after combining counties']):
        for j,(method,label,color) in enumerate(methods):
            values=[float(next(r for r in results if r['state']==s and r['method']==method)[metric]) for s in ('IA','NE','KS')]
            ax.bar(np.arange(3)+(j-1)*.24,values,width=.22,color=color,label=label)
        ax.set_xticks(range(3),['Iowa','Nebraska','Kansas'])
        ax.set_ylabel('Mean absolute error (pp)',labelpad=3)
        ax.set_title(title,fontsize=8,loc='left',pad=7)
        ax.grid(axis='y',color='.9',lw=.5)
        ax.set_axisbelow(True)
    figure.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,
                  bbox_to_anchor=(.5,-.015),frameon=False,fontsize=8,columnspacing=1.5)
    figure.subplots_adjust(left=.075,right=.985,top=.84,bottom=.29,wspace=.35)
    save(figure,'spatial-summaries')


if __name__=='__main__':
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
    reversal()
    summaries()
    print('Wrote both figures at the final 7.16-inch width, with 8-point labels.')
