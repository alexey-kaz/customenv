# MARL Load Balancing Environment
- Подготовка окружения:
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
- Запуск генератора данных:
```
python generate_data.py
```
- Запуск экспериментов:
```
python main.py --n_steps=500 --n_agents=5 --n_rcv=2 --distr=erlang
```
где:
- \-\-n_steps – число шагов итерации
- \-\-n_agents – число агентов/вычислительных 	устройств в сети
- \-\-n_rcv – число периферийных агентов
- \-\-distr – вид распределения сгенерированных исходных данных