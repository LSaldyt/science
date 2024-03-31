from uuid import uuid4
from pathlib import Path
# import logging
import typing
from datetime import datetime
from dataclasses import dataclass
import csv

from contextlib import contextmanager

import numbers
import numpy as np

import json, os
from collections import namedtuple
from contextlib import contextmanager
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

@dataclass
class Writer:
    file_handle : typing.IO
    dict_writer : csv.DictWriter
    count       : int = 0

class Experiment:
    ''' A custom class for organizing experiments & results '''
    def __init__(self, name, settings):
        self.id = uuid4()
        self.exp_timestamp = datetime.now()
        self.name = name
        self.settings = settings
        self.experiment_dir = Path(f'{settings.parent_dir}/{name}')
        self.instance_dir   = self.experiment_dir.joinpath(self.exp_timestamp.strftime('%b_%d_%Y_%H%M%S'))
        self.phases = ['train', 'test', 'val']
        self.writers = dict()

    def ensure(self, write=True):
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        sub = lambda p : self.instance_dir.joinpath(p)
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

    def cleanup(self):
        for writer in self.writers.values():
            writer.file_handle.close()

    @contextmanager
    def ensured(self):
        self.ensure()
        yield
        self.cleanup()

    def update(self, **kwargs):
        self.settings.update(**kwargs)

    def derive(self, name, **kwargs):
        derived_exp = type(self)(f'{self.name}_{name}', self.settings.derive(**kwargs))
        return derived_exp

    def save_json(self, tag, data, prefix=None):
        location = self.data if prefix is None else self.data / prefix
        path = location / (tag + '.json')
        path.write_text(json.dumps(data, indent=4))

    def save_tensors(self, tensors, prefix=None):
        ''' tensors is a dictionary {k : v} where k is str, v is castable to ndarray '''
        location = self.data if prefix is None else self.data / prefix

        for k, v in tensors.items():
            path = location.joinpath(k + '.npz')
            if not isinstance(v, tuple):
                v = np.array(v)
                np.savez(path, v)
            else:
                for si, sv in enumerate(v):
                    sv = np.array(sv)
                    np.savez(path.with_name(f'{k}_{si}'), sv)

    def timestamp(self, from_dt=None):
        now = datetime.now() if from_dt is None else from_dt
        return now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    def log(self, writer_name, metrics, out_str=None, **additional):
        if self.settings.meta.timestamp:
            metrics['timestamp'] = self.timestamp()
        metrics.update(**additional)
        if writer_name not in self.writers:
            file_handle = open(self.data / f'{writer_name}.csv', 'w')
            self.writers[writer_name] = Writer(file_handle, csv.DictWriter(file_handle, fieldnames=list(metrics.keys())))
        writer = self.writers[writer_name]
        if writer.count % self.settings.meta.flush_interval == 0:
            writer.file_handle.flush()
        if writer.count % self.settings.meta.log_interval == 0:
            if out_str is not None:
                self.settings.meta.log_function(out_str.format(**metrics))
        writer.dict_writer.writerow(metrics)
        writer.count += 1

    def flush(self, writer):
        if self.settings.flush:
            writer.flush() # Don't let data die in the buffer :)
            os.fsync(writer)

    def show(self):
        print(f'Settings for experiment: {self.name}')
        self.settings.show()

    ''' The three unimplemented methods, commonly used in derived classes '''

    def run(self, *args, **kwargs):
        with self.ensured(): # Demonstration of how this should be used
            raise NotImplementedError('The default Experiment class does not implement run(), use a derived class :)')

    def reseed(self, seed):
        raise NotImplementedError('Must define reseeding for derived class')

    def checkpoint(self, model):
        raise NotImplementedError('No default checkpoint() function is defined')
