# MARL Load Balancing Environment
- Virtual environment preparation:
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
- Running data generator:
```
python generate_data.py
```
- Running experiments:
```
python main.py --n_steps=500 --n_agents=5 --n_rcv=2 --distr=erlang
```
where:
- \-\-n_steps – number of steps in an iteration
- \-\-n_agents – number of agents / computing devices in the network
- \-\-n_rcv – number of edge computing devices
- \-\-distr – distribution type of given data 