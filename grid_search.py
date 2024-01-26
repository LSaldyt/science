import numpy as np

def grid_search(base, variations):
    ''' Ex: variations=dict(learning_rate=[1e-3, 1e-4], optimizer=['yogi', 'rmsprop'])
        should produce a 2x2 grid for 4 total experiments'''
    for base_name in base.models:
        settings_list = []
        frozen  = list(variations.items())
        lengths = [len(options) for name, options in frozen]
        total   = np.prod(lengths)
        indices = np.zeros(len(lengths), dtype=np.int32)
        for t in range(total):
            long_name = '_'.join(f'{k}{i}' for k, i in zip(variations.keys(), indices))
            grid_name = f'{base_name}_{long_name}'
            diff = dict()
            for l, i in enumerate(indices):
                name, options = frozen[l]
                option        = options[i]
                diff.update({name : option})
            derived = base.derive(**diff)
            for li, l in reversed(list(enumerate(lengths))):
                if indices[li] < l - 1:
                    indices[li] += 1
                    break
                else:
                    indices[li] = 0
            yield (base_name, grid_name, derived)
