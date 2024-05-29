from scripts.neuroglancer_viewer import setup_neuroglancer
from scripts.vesicle_load_object_files import display_and_count_intersections
from scripts.vesicle_load_tensor_files import load_calculate_and_visualize_neuron_and_vesicles


def vizualize_veiscles_dyo():
    file_paths = [
        '/Users/ayalyakobe/Desktop/SHL17/_0019_SHL_17.obj',
        '/Users/ayalyakobe/Desktop/SHL18/_0021_SHL_18.obj',
        '/Users/ayalyakobe/Desktop/SHL18/Segment__0006_LV.obj',
        '/Users/ayalyakobe/Desktop/SHL18/Segment__0010_DV.obj',
        '/Users/ayalyakobe/Desktop/SHL17/Segment__0010_DV.obj',
        '/Users/ayalyakobe/Desktop/SHL17/Segment__0006_LV.obj'
    ]

    display_and_count_intersections(file_paths)

def neuro_viewer():
    setup_neuroglancer()

def tensor_vizualization_and_calculations():
    load_calculate_and_visualize_neuron_and_vesicles('data/vol0_mask.h5', 'data/vol0_vesicle_ins.h5')


if __name__ == "__main__":
    tensor_vizualization_and_calculations()