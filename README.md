# NHM-Assist
NHM-Assist is a collection of python workflows presented in Jupyter notebooks for evaluating, running and interpreting [National Hydrologic Model](https://www.sciencebase.gov/catalog/item/626c0d67d34e76103cd2ce4a) domains using [pywatershed](https://github.com/EC-USGS/pywatershed)


## Install conda-forge miniforge on your PC
Instructions at this [link](https://github.com/conda-forge/miniforge)


## Build the environment
Open a miniforge prompt.

If the `nhm` environment already exists, then remove it: 
```
conda remove -y --name nhm --all
```

Install a fresh env:
```
conda env create -f environment.yml
```
## 1/8/25 note: need to run 'python pull_domain.py --name=willamette_river' in the command window from the repo to get the example/sample model

## Activate the environment:
```
activate nhm
```


## Launch Jupyter

```
jupyter lab
```

Ready to go! :+1:
