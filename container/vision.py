# Basics
import numpy as np
import sys

# Data loading
import torch
from torch.nn.functional import pad
import h5py

# Ridge regression
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config

# Model
from transformers import BridgeTowerModel, BridgeTowerProcessor

# Specialized functions
from container.utils import remove_nan, generate_leave_one_run_out, \
    calc_correlation, Delayer

from tqdm import tqdm


# Helper functions
def load_hdf5_array(file_name, key=None, slice=slice(0, None)):
    """Function to load data from an hdf file.

    Parameters
    ----------
    file_name: string
        hdf5 file name.
    key: string
        Key name to load. If not provided, all keys will be loaded.
    slice: slice, or tuple of slices
        Load only a slice of the hdf5 array. It will load `array[slice]`.
        Use a tuple of slices to get a slice in multiple dimensions.

    Returns
    -------
    result : array or dictionary
        Array, or dictionary of arrays (if `key` is None).
    """
    with h5py.File(file_name, mode='r') as hf:
        if key is None:
            data = dict()
            for k in hf.keys():
                data[k] = hf[k][slice]
            return data
        else:
            return hf[key][slice]


def setup_model(layer):
    """Function to setup transformers model with layer hooks.

    Parameters
    ----------
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer

    Returns
    -------
    device : cuda or cpu for gpu acceleration if accessible.
    model: BridgeTower model.
    processor: BridgeTower processor.
    features: Dictionary
        A placeholder for batch features, one for each forward
        hook.
    layer_selected: Relevant layer chosen for forward hook.
    """
    # Define Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
    model = model.to(device)

    # Define layers
    model_layers = {
            1: model.cross_modal_text_transform,
            2: model.cross_modal_image_transform,
            3: model.token_type_embeddings,
            4: model.vision_model.visual.ln_post,
            5: model.text_model.encoder.layer[-1].output.LayerNorm,
            6: model.cross_modal_image_layers[-1].output,
            7: model.cross_modal_text_layers[-1].output,
            8: model.cross_modal_image_pooler,
            9: model.cross_modal_text_pooler,
            10: model.cross_modal_text_layernorm,
            11: model.cross_modal_image_layernorm,
            12: model.cross_modal_text_link_tower[-1],
            13: model.cross_modal_image_link_tower[-1],
        }

    # placeholder for batch features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            # detached_outputs = [tensor.detach() for tensor in output]
            last_output = output[-1].detach()
            features[name] = last_output  # detached_outputs
        return hook

    # register forward hooks with layers of choice
    layer_selected = model_layers[layer].register_forward_hook(
        get_features(f'layer_{layer}'))

    processor = BridgeTowerProcessor.from_pretrained(
        "BridgeTower/bridgetower-base")

    return device, model, processor, features, layer_selected


# Main functions
def get_movie_features(movie, subject, layer, n=30):
    """Function to average feature vectors over every n inputs.

    Parameters
    ----------
    movie_data: Array
        An array of shape (n_images, 512, 512). Represents frames from
        a color movie.
    n (optional): int
        Number of frames to average over. Set at 30 to mimick an MRI
        TR = 2 with a 15 fps movie.

    Returns
    -------
    data : Dictionary
        Dictionary where keys are the model layer from which activations are
        extracted. Values are lists representing activations of 768 dimensions
        over the course of n_images / 30.
    """
    try:
        data = np.load(f"results/features/movie/{subject}/layer{layer}" +
                       f"_{movie}.npy")
        print("Loaded movie features")
    except FileNotFoundError:
        data_path = 'data/raw_stimuli/shortclips/stimuli/'

        print("loading HDF array")
        movie_data = load_hdf5_array(f"{data_path}{movie}.hdf", key='stimuli')

        # Define Model
        device, model, processor, features, layer_selected = setup_model(layer)

        # create overall data structure for average feature vectors
        # a dictionary with layer names as keys and a 
        # list of vectors as it values
        data = {}

        # a dictionary to store vectors for n consecutive trials
        avg_data = {}

        print("Running movie through model")
        # loop through all inputs
        for i, image in tqdm(enumerate(movie_data)):

            model_input = processor(image, "", return_tensors="pt")
            # Assuming model_input is a dictionary of tensors
            model_input = {key: value.to(device) for key,
                           value in model_input.items()}

            _ = model(**model_input)

            for name, tensor in features.items():
                if name not in avg_data:
                    avg_data[name] = []
                avg_data[name].append(tensor)

            # check if average should be stored
            if (i + 1) % n == 0:
                for name, tensors in avg_data.items():
                    first_size = tensors[0].size()

                    if all(tensor.size() == first_size for tensor in tensors):
                        avg_feature = torch.mean(torch.stack(tensors), dim=0)
                        avg_feature_numpy = avg_feature.detach().cpu().numpy()
                        # print(len(avg_feature_numpy))
                    else:
                        # Find problem dimension
                        for dim in range(tensors[0].dim()):
                            first_dim = tensors[0].size(dim)

                            if not all(tensor.size(dim) == first_dim
                                       for tensor in tensors):
                                # Specify place to pad
                                p_dim = (tensors[0].dim()*2) - (dim + 2)
                                # print(p_dim)
                                max_size = max(tensor.size(dim)
                                               for tensor in tensors)
                                padded_tensors = []

                                for tensor in tensors:
                                    # Make a list with length of 2*dimensions
                                    # - 1 to insert pad later
                                    pad_list = [0] * ((2*tensor[0].dim()) - 1)
                                    pad_list.insert(
                                        p_dim, max_size - tensor.size(dim))
                                    # print(tuple(pad_list))
                                    padded_tensor = pad(tensor,
                                                        tuple(pad_list))
                                    padded_tensors.append(padded_tensor)

                        avg_feature = torch.mean(torch.stack(padded_tensors),
                                                 dim=0)
                        avg_feature_numpy = avg_feature.detach().cpu().numpy()
                        # print(len(avg_feature_numpy))

                    if name not in data:
                        data[name] = []
                    data[name].append(avg_feature_numpy)

                avg_data = {}

        layer_selected.remove()

        # Save data
        data = np.array(data[f"layer_{layer}"])
        print("Got movie features")

        np.save(f"results/features/movie/{subject}/layer{layer}_{movie}.npy",
                data)

    return data


def vision_model(subject, layer):
    """Function to build the vision encoding model. Creates a
    matrix mapping the linear relationship between BridgeTower features
    and brain voxel activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.

    Returns
    -------
    average_coef: Array
        Array of shape (layer_output_size*4, num_voxels) mapping
        the relationship of delayed feature vectors to each voxel
        in the fmri data.
"""
    print("Extracting features from data")

    # Extract features from raw stimuli
    train00 = get_movie_features('train_00', subject, layer)
    train01 = get_movie_features('train_01', subject, layer)
    train02 = get_movie_features('train_02', subject, layer)
    train03 = get_movie_features('train_03', subject, layer)
    train04 = get_movie_features('train_04', subject, layer)
    train05 = get_movie_features('train_05', subject, layer)
    train06 = get_movie_features('train_06', subject, layer)
    train07 = get_movie_features('train_07', subject, layer)
    train08 = get_movie_features('train_08', subject, layer)
    train09 = get_movie_features('train_09', subject, layer)
    train10 = get_movie_features('train_10', subject, layer)
    train11 = get_movie_features('train_11', subject, layer)

    # Build encoding model
    print("Loading movie fMRI data")
    # Load fMRI data
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")

    # Prep data
    train_fmri = remove_nan(fmri_train)

    # fmri_arrays = train_fmri
    feature_arrays = [train00, train01, train02, train03, train04,
                      train05, train06, train07, train08, train09,
                      train10, train11]

    # Combine data
    Y_train = train_fmri
    X_train = np.vstack(feature_arrays)

    # Define cross-validation
    run_onsets = []
    current_index = 0
    for arr in feature_arrays:
        next_index = current_index + arr.shape[0]
        run_onsets.append(current_index)
        current_index = next_index

    n_samples_train = X_train.shape[0]
    cv = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # cross-validation splitter into a reusable list

    # Define the model
    scaler = StandardScaler(with_mean=True, with_std=False)

    delayer = Delayer(delays=[1, 2, 3, 4])

    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

    X_train = X_train.astype("float32")

    alphas = np.logspace(1, 20, 20)

    print("Running linear model")
    ridge_cv = RidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100))

    pipeline = make_pipeline(
        scaler,
        delayer,
        ridge_cv,
    )

    set_config(display='diagram')  # requires scikit-learn 0.23
    pipeline

    _ = pipeline.fit(X_train, Y_train)

    coef = pipeline[-1].coef_
    coef = backend.to_numpy(coef)
    print("(n_delays * n_features, n_voxels) =", coef.shape)

    # Regularize coefficients
    coef /= np.linalg.norm(coef, axis=0)[None]

    # split the ridge coefficients per delays
    delayer = pipeline.named_steps['delayer']
    coef_per_delay = delayer.reshape_by_delays(coef, axis=0)
    print("(n_delays, n_features, n_voxels) =", coef_per_delay.shape)
    del coef

    # average over delays
    average_coef = np.mean(coef_per_delay, axis=0)
    print("(n_features, n_voxels) =", average_coef.shape)
    del coef_per_delay

    print("Finished vision encoding model")

    print(average_coef.shape)
    return average_coef


def vision_prediction(subject, layer, vision_encoding_matrix):
    """Function to run the vision encoding model. Predicts brain activity
    to story listening and return correlations between predictions and real
    brain activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.
    vision_encoding_matrix: array
        Generated by vision_model() function. A matrix mapping the relationship
        between feature vectors and brain activity

    Returns
    -------
    correlations: Array
        Array of shape (num_voxels) representing the correlation between
        predictions and real brain activity for each voxel.
    """
    # Get test features
    test = get_movie_features('test', subject, layer)

    # Load fmri data
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Prep data
    test_fmri = remove_nan(fmri_test)

    # Make fmri predictions
    test_predictions = np.dot(test, vision_encoding_matrix)

    # Calculate correlations
    test_correlations = calc_correlation(test_predictions, test_fmri)

    print("Max correlation:", np.nanmax(test_correlations))

    return test_correlations


if __name__ == "__main__":
    if len(sys.argv) == 3:
        subject = sys.argv[1]
        layer = int(sys.argv[2])

        print("Building vision model")
        # Build encoding model
        vision_encoding_matrix = vision_model(subject, layer)

        print("Predicting fMRI data and calculating correlations")
        # Predict story fmri with vision model
        correlations = vision_prediction(subject, layer,
                                         vision_encoding_matrix)

        np.save('results/vision_model/' + subject +
                '/layer' + str(layer) + '_correlations.npy', correlations)
    else:
        print("This script requires exactly two arguments: subject \
               and layer. Ex. python vision.py S1 8")
