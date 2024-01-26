from .grid_search import *
from .experiment  import Experiment

from contextlib import contextmanager
from rich import print

def exclusive_add(d, k, v, name='entries'):
    print(f'{name.title()}: [blue] {k}')
    if k not in d:
        d[k] = v
    else:
        raise KeyError(f'Cannot create two {name} with the same key: {k}')

class Registry:
    def __init__(self):
        self.experiments  = dict()
        self.models       = dict()
        self.datasets     = dict()
        self.setups       = dict()
        self.ablations    = dict()
        self.designations = dict()
        self.variations   = dict()
        self.finalized    = False
        self.shared       = None
        self.scope_name   = ''
        self.constructor  = None

    def __contains__(self, key):
        return key in self.experiments

    @contextmanager
    def scope(self, name, settings=None, constructor=None):
        prev = self.scope_name
        prev_constructor = self.constructor
        if constructor is not None:
            self.constructor = constructor
        if settings is None:
            settings = self.shared
        self.scope_name = name
        try:
            yield settings
        finally:
            self.scope_name = prev
            self.constructor = prev_constructor

    def finalize(self, setup=None):
        if self.finalized:
            return # Skip
        if setup is not None:
            self.setups[setup](self, self.shared)
        else:
            for key, fn in self.setups.items():
                print(f'Finalizing {key}')
                fn(self, self.shared)
        self.finalized = True

    def fetch(self, exp):
        if self.finalized:
            return self.experiments[exp]
        else:
            components = exp.split('_')
            for comp in components:
                for key, fn in self.setups.items():
                    if key == comp:
                        fn(self, self.shared) # Required to add exp
                        return self.experiments[exp]
            raise KeyError(exp)

    # Yes, I realize the following is somewhat duplicated, but it's perhaps simple
    #   enough and easy to trace in this form

    def rescope(self, k):
        if self.scope_name == '':
            return k
        elif k == '':
            return self.scope_name
        else:
            return self.scope_name + '_' + k

    def rescope_all(self, d):
        return {self.rescope(k) : v for k, v in d.items()}

    def add_designations(self, designations):
        self.designations.update(self.rescope_all(designations))

    def add_designation(self, des, key=''):
        exclusive_add(self.designations, self.rescope(key), des, name='designations')

    def add_variations(self, variations):
        self.variations.update(self.rescope_all(variations))

    def add_variations(self, des, key=''):
        exclusive_add(self.variations, self.rescope(key), des, name='variations')

    def add_ablations(self, ablations):
        self.ablations.update(self.rescope_all(ablations))

    def add_ablation(self, ablation, key=''):
        exclusive_add(self.ablations, self.rescope(key), albation, name='ablations')

    def add_models(self, models):
        self.models.update(self.rescope_all(models))

    def add_model(self, model, key=''):
        exclusive_add(self.models, self.rescope(key), model, name='models')

    def make_model(self, name, settings, alias=None, apply_wrapper=True):
        raise NotImplementedError('Please define make_model for a specific framework, e.g. jax')

    def add_datasets(self, datasets):
        self.datasets.update(self.rescope_all(datasets))

    def add_dataset(self, dataset, key=''):
        exclusive_add(self.datasets, self.rescope(key), dataset, name='datasets')

    def add_shared(self, shared):
        if self.shared is not None:
            raise ValueError('self.shared has already been set!')
        self.shared = shared

    def add_setup(self, key, setup):
        exclusive_add(self.setups, self.rescope(key), setup, name='setups')
        self.finalized   = False

    def add_experiment(self, experiment):
        experiment.update(registry=self) # At final stage only
        exclusive_add(self.experiments, experiment.name,
                      experiment, name='experiments')

    def run(self, name):
        exp = self.experiments[name]
        exp.update(registry=self) # At final stage only
        for wrapper in exp.settings.wrapped_models:
            wrapper.settings.update(registry=self)
        exp.run()

    def list(self, setup=None):
        self.finalize(setup=setup)
        for k, v in sorted(self.experiments.items(), key=lambda t : t[0]):
            print(f'[blue] {k:40}')

    def list_setups(self):
        for k in sorted(self.setups.keys()):
            print(f'[blue] {k:40}')

    def split_dataset(self, settings, dataset_name):
        raise NotImplementedError('Please define split_dataset for a specific framework, e.g. jax')

    def add(self, settings, name='', exp_constructor=None, dataset='',
            ablations=None, alt_settings=None):
        ''' A helper function to either do a wide comparison or focused ablation '''
        name    = self.rescope(name)
        dataset = self.rescope(dataset)
        if exp_constructor is None:
            exp_constructor = self.constructor
        assert exp_constructor is not None
        assert not (ablations is not None and alt_settings is not None) # XOR
        assert isinstance(settings.models, tuple) or isinstance(settings.models, list)
        if isinstance(ablations, str):
            ablations = self.ablations[ablations] # Ah, yes, clarity..
        new_settings = settings.derive()
        if alt_settings is None and ablations is None:
            new_settings.update(
                wrapped_models=tuple(self.make_model(m, new_settings)
                                     for m in new_settings.models))
        elif ablations is not None:
            wrapped_models = []
            new_models = []
            for model in new_settings.models:
                for ab_name, ablation in ablations:
                    ablated = new_settings.derive(ablation=ablation)
                    alias = f'{model}_{ab_name}'
                    new_models.append(alias)
                    wrapped_models.append(
                        self.make_model(model, ablated, alias=alias))
            new_settings.update(wrapped_models=tuple(wrapped_models),
                                models=tuple(new_models))
        elif alt_settings is not None:
            ''' Generic way to make hyperparameter searches '''
            wrapped_models = []
            new_models = []
            for base_name, grid_name, alt_sets in alt_settings:
                wrapped_models.append(
                    self.make_model(base_name, alt_sets, alias=grid_name))
                new_models.append(grid_name)
            new_settings.update(models=tuple(new_models),
                wrapped_models=tuple(wrapped_models))
        self.split_dataset(new_settings, dataset)
        self.add_experiment(exp_constructor(name, new_settings))

    def add_grid(self, name, base, variations, exp_c=None, dataset=''):
        if isinstance(variations, str):
            variations = self.variations[variations]
        return self.add(base, name=name, dataset=dataset, exp_constructor=exp_c,
                        alt_settings=list(grid_search(base, variations)))



