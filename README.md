# NHMassist
Collection of python workflows for evaluating, running and interpreting National Hydrologic Model sub-basin extractions (NHMx)

## Install conda-forge miniforge on your PC
Instructions at this [link](https://github.com/conda-forge/miniforge)

Creating the NHM environment is currently done manually, but will eventually be done using a .yml file. For now...

## Install pyWatershed environment

Instructions at this [link](https://github.com/EC-USGS/pywatershed)

Open a miniforge prompt and enter:
```
git clone https://github.com/EC-USGS/pywatershed.git
cd pywatershed
mamba env create -f environment_w_jupyter.yml
activate pws
pip install -e .
```

## Make a copy of the pws environment
```
conda create --name pws_pyPRMS --clone pws
```
## Activate the copy
```
activate pws_pyPRMS
```

## Install pyPRMS dependencies and other needed libraries:
more information about pyPRMS installation can be found at this [link](https://github.com/paknorton/pyPRMS)
```
mamba install pygeohydro ipyleaflet dataretrieval pyogrio toblar
mamba install matplotlib cartopy numpy netCDF4 xarray (all requested packages installed already, but it wanted to updat matplotlib, but I declined)
mamba install pre-commit rich ipywidgets plotly (check first these may already be there)
pip install hydroeval
pip install hyswap
```
	
## LAST (must be last)! Install pyPRMS
```
pip install git+https://github.com/paknorton/pyPRMS.git@development
```

## Deactivate the pws_pyPRMS environment
```
deactivate
```

## Copy the pws_pyPRMS environment (make a copy):
```
conda create --name NHM --clone pws_pyPRMS
```


## Now, navigate to your NHM-assist repo directory in your miniforge prompt:
```
cd "paste path here"
```

## Activate the environment:
```
activate NHM
```

## and finally launch Jupyter:
```
jupyter lab
```
Ready to go! :+1:
