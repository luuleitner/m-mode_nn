import os
import pickle
import zarr
import h5py


def init_dataset(path, data_size, backend='h5', chunk_size=1024, label_shape=None, label_dtype='int64'):
    """
    Initialize a dataset file for storing tokens and labels.

    Args:
        path: File path for the dataset
        data_size: Shape of a single data sample (excluding batch dimension)
                   For transformer: (seq_window, C, H, W)
                   For flat/CNN: (C, H, W)
        backend: Storage backend - 'h5' or 'zarr'
        chunk_size: Chunk size for zarr backend
        label_shape: Shape of labels per sample (excluding batch dimension)
                     For transformer: (seq_window, 1) or (seq_window, num_classes)
                     For flat with hard labels: (1,)
                     For flat with soft labels: (num_classes,)
                     If None, defaults to (data_size[0], 1) for backwards compatibility
        label_dtype: Data type for labels - 'int64' for hard, 'float32' for soft

    Returns:
        File handle (h5py.File or zarr.Group)
    """
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

    # Create token dataset
    create('token', (0, *data_size), 'float32', (None, *data_size))

    # Create label dataset with flexible shape
    if label_shape is None:
        # Backwards compatibility: assume transformer mode with hard labels
        label_shape = (data_size[0], 1)

    create('label', (0, *label_shape), label_dtype, (None, *label_shape))

    return f

def append_and_save(path, fileobj, data, label, config=None, backend='h5'):
    """
    Append data and labels to an existing dataset file.

    Args:
        path: File path (used for config saving)
        fileobj: Open file handle from init_dataset
        data: Data array to append [batch, ...]
        label: Label array to append [batch, ...]
        config: Optional config object to save as pickle
        backend: Storage backend - 'h5' or 'zarr'
    """
    # Append token
    token = fileobj['token']
    cur = token.shape[0]
    new = cur + data.shape[0]
    token.resize((new, *token.shape[1:]))
    token[cur:new] = data

    # Append label (flexible shape support)
    label_ds = fileobj['label']
    cur_l = label_ds.shape[0]
    new_l = cur_l + label.shape[0]

    # Resize to match the actual label shape from input
    new_shape = (new_l, *label.shape[1:])
    label_ds.resize(new_shape)
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
