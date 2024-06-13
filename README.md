# Elmundo

## Installing
1. navigate to local target directory and clone the repository, then navigate to Elmundo
```
git clone https://github.com/mmoulton1/Elmundo

cd Elmundo
```

2. Create and activate conda environment
```
conda create --name elmundo python=3.9 -y
conda activate elmundo
```
3. Install dependencies
```
conda install geopandas
pip install -r requirements.txt
```
4. Install Elmundo
```
pip install -e .
```