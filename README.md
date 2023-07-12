## Description
- Fine tuning **DistilBert**
- Site with revies  on **Django**

## Usage
```bash
git clone https://github.com/Fruha/GreenAtomFilms
cd GreenAtomFilms
pip install -r requirements.txt
```
### Server
```bash
python .\manage.py runserver --noreload
```
### Model training
```bash
python nnmodel\train.py
tensorboard --logdir lightning_logs
```