"""Check stored aggregates, reporting labels, and summary queries."""
from pathlib import Path
import csv
import json

import numpy as np

from query_summary import query

ROOT=Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT/'results'/name).open() as stream:
        return list(csv.DictReader(stream))


def label(change):
    return 1 if change>=1 else -1 if change<=-1 else 0


def main():
    counties=read('operator_panel_counties.csv')
    states=read('operator_panel_states.csv')
    summaries=read('spatial_summary_counties.csv')
    compact_results=read('spatial_summary_results.csv')
    assert len(counties)==3861 and len(states)==39
    label_rows=[]
    for state,count in [('IA',99),('NE',93),('KS',105)]:
        selected=sorted([r for r in states if r['state']==state],key=lambda r:r['valid_date'])
        assert len(selected)==13
        for row in selected:
            members=[r for r in counties if r['state']==state and r['valid_date']==row['valid_date']]
            assert len(members)==count
            total=sum(int(r['corn_pixels']) for r in members)
            for method in ('pixel','centroid','internal','area'):
                exposure=100*sum(int(r['corn_pixels'])*float(r[method+'_fraction']) for r in members)/total
                assert abs(exposure-float(row[method+'_exposure_pct']))<1e-8
        for previous,current in zip(selected,selected[1:]):
            differences={method:float(current[method+'_exposure_pct'])-float(previous[method+'_exposure_pct'])
                         for method in ('pixel','centroid','internal','area')}
            label_rows.append({'state':state,'before':previous['valid_date'],'after':current['valid_date'],
                               **{method+'_change_pp':value for method,value in differences.items()},
                               **{method+'_class':label(value) for method,value in differences.items()}})
    assert sum(r['centroid_class']!=r['pixel_class'] for r in label_rows)==6
    assert sum(r['area_class']!=r['pixel_class'] for r in label_rows)==0
    reversals=[r for r in label_rows if r['state']!='IA' and
               r['centroid_change_pp']*r['pixel_change_pp']<0 and
               min(abs(r['centroid_change_pp']),abs(r['pixel_change_pp']))>=1]
    assert len(reversals)==1
    queries=0
    for state in ('IA','NE','KS'):
        selected=[r for r in summaries if r['state']==state]
        for stamp in sorted(set(r['valid_date'] for r in selected)):
            current=[r for r in selected if r['valid_date']==stamp]
            by_fips={r['fips']:r for r in current}
            for cells,method in [(1,'area'),(4,'histogram_4'),(16,'histogram_16')]:
                result=query(state,stamp,cells)
                for county in result['counties']:
                    assert abs(county['fraction']-float(by_fips[county['fips']][method]))<1e-9
                queries+=1
        for method in ('area','moment_fixed','moment_fitted','histogram_4','histogram_16'):
            local,state_errors=[],[]
            for stamp in sorted(set(r['valid_date'] for r in selected)):
                current=[r for r in selected if r['valid_date']==stamp]
                weights=np.asarray([int(r['corn_pixels']) for r in current])
                error=np.asarray([float(r[method])-float(r['reference']) for r in current])*100
                local.append(np.average(np.abs(error),weights=weights))
                state_errors.append(abs(np.average(error,weights=weights)))
            expected=next(r for r in compact_results if r['state']==state and r['method']==method)
            assert abs(np.mean(local)-float(expected['county_weighted_mae_pp']))<1e-9
            assert abs(np.mean(state_errors)-float(expected['state_mae_pp']))<1e-9
    with (ROOT/'results/reporting_decisions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(label_rows[0]))
        writer.writeheader(); writer.writerows(label_rows)
    report={'county_weeks':len(counties),'state_weeks':len(states),'summary_queries_matched':queries,
            'transitions':36,'evaluation_transitions':24,'evaluation_opposite_changes_at_least_1pp':len(reversals),
            'center_reporting_disagreements':6,'area_reporting_disagreements':0,
            'summary_aggregation_checked':True}
    (ROOT/'results/consistency_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
