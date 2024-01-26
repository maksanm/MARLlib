# MARLlib
Using ***MARLlib***, a MARL library that utilizes ***Ray*** and one of its toolkits ***RLlib***, to test **Multi-agent Proximal Policy Optimization (PPO)** and **Independent PPO** MARL algorithms in selected multi-agent environments.

## Installation
The installation of the ***MARLlib*** may be quite challenging due to conflicts between the versions of the packages it depends on.
1. Create and activate Python 3.8 (or 3.9) virtual environment (e.g. using ***venv***).
2. Downgrade ***whell*** and install ***MARLlib*** with dependencies. Inside cloned repositorium:
```
cd MARLlib
pip install wheel==0.38.0
pip install -e .
```
3. Install SISL environment (MPE environment installs with ***MARLlib***) and ***gym***.
```
pip install swig
pip install pyglet==1.5.27
pip install pettingzoo[sisl]==1.12.0
pip install gym==0.20.0
```
4. Downgrade ***protobuf*** and apply patches.
```
pip install protobuf==3.20
python marllib/patch/add_patch.py -y
cd ..
```

## Environments and algorithms configuration
The configuration files for the environments can be found in the `/MARLlib/marllib/envs/base_env/config/(mpe|sisl).yaml` files. A list of the available arguments can be found in the [MPE Simple Adversary](https://pettingzoo.farama.org/environments/mpe/simple_adversary/) and [SISL Multiwalker](https://pettingzoo.farama.org/environments/sisl/multiwalker/) sections of the ***PettingZoo*** documentation.

The configurations for the **IPPO** and **MAPPO** algorithms can be found under the `/MARLlib/marllib/marl/algos/hyperparams/finetuned/{environment_name}_{map_name}` directories.

If you want to modify the algorithm used in the scripts, you can do so by replacing `alg_name` in the algorithm initialization statement `marl.algos.{alg_name}(hyperparam_source="source")`.

## Running
To visualize any of the trained models, run the appropriate Python script. For example:
```
python render_ippo_sisl_multiwalker.py
```
All checkpoints for the trained models are located in the `/results` folder. To check the state of a model at a desired iteration, replace `model_path` in the corresponding script.

You can also check the charts illustrating the differences between **IPPO** and **MAPPO** in the `.ipynb` files.