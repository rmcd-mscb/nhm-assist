# NHMassist
Collection of python workflows for evaluating, running and interpreting National Hydrologic Model sub-basin extractions (NHMx)


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


## Activate the environment:
```
activate nhm
```


## Launch Jupyter

```
jupyter lab
```

Ready to go! :+1:
