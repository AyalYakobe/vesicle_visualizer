import neuroglancer
import numpy as np
import h5py


def setup_neuroglancer(ip='localhost', port=8080):
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer()

    # Define the coordinate space
    res = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units=['nm', 'nm', 'nm'],
        scales=[8, 8, 30]
    )

    # Load the data from the HDF5 files
    file_im = 'data/vol0_im.h5'
    file_mask = 'data/vol0_mask.h5'
    file_vesicle = 'data/vol0_vesicle_ins.h5'

    print('Loading image, mask, and vesicle data...')
    with h5py.File(file_im, 'r') as f_im:
        im = np.array(f_im['main'])

    with h5py.File(file_mask, 'r') as f_mask:
        mask = np.array(f_mask['main'])

    with h5py.File(file_vesicle, 'r') as f_vesicle:
        vesicle = np.array(f_vesicle['main'])

    print(im.shape, mask.shape, vesicle.shape)
    print("Vesicle IDs: {}".format(np.unique(vesicle)[1:]))

    # Function to create a Neuroglancer layer
    def ngLayer(data, res, oo=[1, 1, 1], tt='segmentation'):
        return neuroglancer.LocalVolume(data, dimensions=res, volume_type=tt, voxel_offset=oo)

    # Add layers to the Neuroglancer viewer
    with viewer.txn() as s:
        s.layers.append(name='im', layer=ngLayer(im, res, tt='image'))
        s.layers.append(name='mask', layer=ngLayer(mask, res, tt='segmentation'))
        s.layers.append(name='vesicle', layer=ngLayer(vesicle, res, tt='segmentation'))

    print(viewer)

    # Keep the server running
    print("Neuroglancer server is running. Press Ctrl+C to exit.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down Neuroglancer server.")
