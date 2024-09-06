# Time-Domain Decomposition Adaptive Spatio-temporal Graph Neural Networks for Traffic Forecasting

This is PyTorch implementation of TDAGNN in the following paper:

Time-Domain Decomposition Adaptive Spatio-temporal Graph Neural Networks for Traffic Forecasting.

## Requirements

- scipy
- torch
- tqdm
- h5py
- numpy
- pandas
- PyYAML
- tensorboardX
- torch
- PyEMD

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```
## Data Preparation

Same as [DCRNN](https://github.com/liyaguang/DCRNN), the traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX), and should be put into the `data/` folder. The `*.h5` files store the data in `panads.DataFrame` using the `HDF5` file format. Here is an example:



Using SMD to decompose traffic data
```
python generate_imf_data.py
```
Generate time-domain decomposition matrix
```
python generate_imf_adj.py
```

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Graph Construction

As the currently implementation is based on pre-calculated road network distances between sensors, it currently only supports sensor ids in Los Angeles (see `data/sensor_graph/sensor_info_201206.csv`).

```
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1 --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```

Besides, the locations of sensors in Los Angeles, i.e., METR-LA, are available at [data/sensor_graph/graph_sensor_locations.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv).

## Model Training

We utilize the A  two-stage  training  strategy to further enhance the model  performance. Here are commands for training the model on `METR-LA`. 

```
#first stage
python train_stage1.py --config metr-la-stage1-config --name stage-1
#seconde stage
python train_stage2.py --config metr-la-stage2-config --name stage-2
```

Here are commands for training the model on `PEMS-BAY`. 

```
#first stage
python train_stage1.py --config pems-bay-stage1-config --name stage-1
#seconde stage
python train_stage2.py --config pems-bay-stage2-config --name stage-2
```





