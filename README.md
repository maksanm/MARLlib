# MARLlib
Using ***MARLlib***, a MARL library that utilizes ***Ray*** and one of its toolkits ***RLlib***, to test **Multi-agent Proximal Policy Optimization (PPO)** and **Independent PPO** MARL algorithms in selected multi-agent environments.

## Installation
1. Create and activate Python 3.8 (or 3.9) virtual environment (e.g. using ***venv***).
2. Install ***MARLlib*** with dependencies.
```
cd MARLlib
pip install -e .
```
3. Install environments (MPE environment installs with ***MARLlib***) and ***gym***.
```
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

## Running
