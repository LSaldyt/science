from uuid import uuid4
from pathlib import Path
# import logging
from datetime import datetime

import numbers
import numpy as np

import json, os
from contextlib import contextmanager
from rich import print as rprint
from rich.pretty import pprint
from .settings import Settings

def default_json(obj):
    if isinstance(obj, Settings):
        return obj.params
    try:
        return obj.__name__
    except:
        try:
            try:
                return str(obj)
            except:
                return str(obj, 'utf-8')
        except:
            print('COULD NOT SERIALIZE', obj)
            return '__unserialized__'


class Experiment:
    ''' A custom class for organizing experiments & results '''
    def __init__(self, name, settings):
        self.id = uuid4()
        self.timestamp = datetime.now()
        self.name = name
        self.settings = settings
        self.experiment_dir = Path(f'{settings.parent_dir}/{name}')
        self.instance_dir   = self.experiment_dir.joinpath(self.timestamp.strftime('%b_%d_%Y_%H%M%S'))
        self.phases = ['train', 'test', 'val']
        self.metrics = dict()
        self.aux_metrics = dict()

    def ensure(self, write=True):
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        sub = lambda p : self.instance_dir.joinpath(p)
        for phase in self.phases:
            self.metrics[phase] = sub(f'{self.name}_{phase}.csv')
            self.aux_metrics[phase] = sub(f'{self.name}_{phase}_aux.csv')
        self.metainfo      = sub('meta.json')
        self.figures       = sub('figures')
        self.checkpoints   = sub('checkpoints')
        self.data          = sub('data')

        self.figures.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.headers = set()
        if write:
            self.metainfo.write_text(json.dumps(self.settings.params, indent=4,
                default=default_json))

    def update(self, **kwargs):
        self.settings.update(**kwargs)

    def derive(self, name, **kwargs):
        derived_exp = type(self)(f'{self.name}_{name}', self.settings.derive(**kwargs))
        return derived_exp

    def run(self, *args, **kwargs):
        raise NotImplementedError('The default Experiment class does not implement run(), use a derived class :)')

    def save_json(self, tag, data):
        path = self.data / (tag + '.json')
        print(path)
        path.write_text(json.dumps(data, indent=4))

    def save_tensors(self, tensors):
        ''' tensors is a dictionary {k : v} where k is str, v is castable to ndarray '''
        for k, v in tensors.items():
            path = self.data.joinpath(k + '.npz')
            if not isinstance(v, tuple):
                v = np.array(v)
                np.savez(path, v)
            else:
                for si, sv in enumerate(v):
                    sv = np.array(sv)
                    np.savez(path.with_name(f'{k}_{si}'), sv)

    @contextmanager
    def loggers(self, kind='train'):
        if kind in self.metrics:
            metrics_file = self.metrics[kind]
            aux_file     = self.aux_metrics[kind]
        else:
            raise NotImplementedError(f'Cannot handle file of label {kind} in Experiment {self.name}')
        buffering = 1 if self.settings.flush else -1
        with open(metrics_file, 'a', buffering=buffering) as metrics_open:
            with open(aux_file, 'a', buffering=buffering) as aux_open:
                yield metrics_open, aux_open
                self.flush(aux_open)
            self.flush(metrics_open)

    def log_aux(self, current_file, aux, total_seen, kind='train'):
        ''' Currently only saves numbers '''
        aux = {k : v for k, v in aux.items() if k != 'tensors'}
        self.log(current_file, aux, total_seen, kind=kind, is_aux=True)

    def log(self, current_file, metrics, total_seen, kind='train', is_aux=False,
            out_str='{name:30}: L={loss:2.6f} t={duration:2.3f}'):
        metrics = {k:v for k,v in metrics.items() if k!='aux'} # Do not log aux or write to csv
        order = list(sorted(metrics.keys()))
        if (kind, is_aux) not in self.headers:
            current_file.write(','.join(order) + '\n')
            self.headers.add((kind, is_aux))
        current_file.write(','.join(str(metrics[k]) for k in order) + '\n')
        if total_seen % self.settings.flush_interval == 0:
            self.flush(current_file)
        if total_seen % self.settings.log_interval == 0 and not is_aux:
            rprint(out_str.format(**metrics))

    def flush(self, current_file):
        if self.settings.flush:
            current_file.flush() # Don't let data die in the buffer :)
            os.fsync(current_file)

    def show(self):
        print(f'Settings for experiment: {self.name}')
        self.settings.show()

    def reseed(self, seed):
        raise NotImplementedError('Must define reseeding for derived class')

    def checkpoint(self, model):
        raise NotImplementedError('No default checkpoint() function is defined')


    def __str__(self):
        return f'{self.name}'
