import os
import pickle
import zarr
import h5py

def init_dataset(path, data_size, backend='h5', chunk_size=1024):
    if backend == 'h5':
        f = h5py.File(path, 'w')
        create = lambda name, shape, dtype, maxshape: f.create_dataset(
            name=name, shape=shape, dtype=dtype, maxshape=maxshape
        )
    elif backend == 'zarr':
        store = zarr.DirectoryStore(path)
        f = zarr.group(store=store, overwrite=True)
        create = lambda name, shape, dtype, maxshape: f.create_dataset(
            name=name, shape=shape, dtype=dtype, chunks=(chunk_size, *shape[1:]), maxshape=maxshape, overwrite=True
        )
    else:
        raise ValueError("backend must be 'h5' or 'zarr'")

    create('token', (0, *data_size), 'float32', (None, *data_size))
    create('label_logic', (0, data_size[0], 1), 'int64', (None, data_size[0], 1))
    return f

def append_and_save(path, fileobj, data, label, config=None, backend='h5'):
    # Append token
    token = fileobj['token']
    cur = token.shape[0]
    new = cur + data.shape[0]
    token.resize((new, *token.shape[1:]))
    token[cur:new] = data

    # Append label_logic
    label_ds = fileobj['label_logic']
    cur_l = label_ds.shape[0]
    new_l = cur_l + label.shape[0]
    label_ds.resize((new_l, label.shape[1], 1))
    label_ds[cur_l:new_l] = label

    if backend == 'h5':
        fileobj.close()

    # Save config as pickle (if provided)
    if config is not None:
        basepath = os.path.dirname(os.path.dirname(path))
        savepath = os.path.join(basepath, "config.pkl")
        try:
            with open(savepath, "xb") as f:
                pickle.dump(config, f)
        except FileExistsError:
            pass
