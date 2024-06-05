from scripts.relevant.convert_tensor_to_object_files import convert_h5_to_obj
from scripts.experimental.neuroglancer_viewer import setup_neuroglancer
from scripts.experimental.vesicle_load_object_files import display_and_count_intersections
from scripts.relevant.vesicle_load_tensor_files import load_calculate_and_visualize_neuron_and_vesicles
from scripts.relevant.vesicle_perimeter_only import load_calculate_and_visualize_perimeter


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


def vizualize_vesicles_neuroglancer():
    setup_neuroglancer()


def vizualization_and_calculations_tensors():
    load_calculate_and_visualize_neuron_and_vesicles('data/vol0_mask.h5', 'data/vol0_vesicle_ins.h5')


def vizualization_and_calculations_tensors_just_perimeter():
    load_calculate_and_visualize_perimeter('data/vol0_mask.h5')


def object_file_converter():
    neuron_file_path = 'data/vol0_mask.h5'
    vesicle_file_path = 'data/vol0_vesicle_ins.h5'
    neuron_obj_path = 'data/vol0_mask.obj'
    vesicle_obj_path = 'data/vol0_vesicle_ins.obj'

    convert_h5_to_obj(neuron_file_path, vesicle_file_path, neuron_obj_path, vesicle_obj_path)


if __name__ == "__main__":
    vizualization_and_calculations_tensors()
