# How Spatial Aggregation Changes Estimates of Crop Exposure to Drought

This notebook compares county-center, county-area, and crop-count summaries of
corn exposure to drought. All methods use the same crop map and acreage weights.

## Run the notebook

Use Python 3.12 and install the dependencies:

```sh
python3 -m pip install -r requirements.txt
```

Open `analysis/reporting_audit.ipynb` in VS Code with the Jupyter extension,
or in an existing Jupyter installation, and run the cells. The requirements
install the Python kernel. To add the Jupyter Notebook interface separately,
run `python3 -m pip install notebook`.
The default is Kansas, August 29 to September 5, 2023. Change the state, dates,
number of crop-count cells, or county in the first code cell to try another case.
Choose consecutive dates, seven days apart.

## Reproduce the results

From this folder, run:

```sh
python3 analysis/reproduce.py
```

This runs `analysis/reproduce_panel.py`, the geometry and summary checks, and
the plotting scripts. All inputs are bundled, so no data download is needed.
The panel step peaked at 2.13 GiB on an M4 Pro. Allow several GB of free RAM
for the full workflow. Tables are in `results/`, and plots are in `figures/`.

## Data

Iowa uses 2021 mapped corn with summer 2022 drought maps. Nebraska and Kansas use
2022 mapped corn with summer 2023 drought maps. The reference counts corn-pixel
centers inside D1+ drought polygons.

- Crop maps: [USDA NASS Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php).
- Weekly drought maps: [U.S. Drought Monitor](https://droughtmonitor.unl.edu/DmData/GISData.aspx), produced by NDMC, USDA, and NOAA.
- County boundaries: [Census 2020 TIGER/Line](https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/).
