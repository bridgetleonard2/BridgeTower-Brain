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
from functions import remove_nan, generate_leave_one_run_out, \
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
        movie_data = np.load(f"results/features/movie/{subject}/layer{layer}" +
                             f"_{movie}.npy")
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
    test = get_movie_features('test', subject, layer)

    feature_arrays = np.vstack([train00, train01, train02, train03, train04,
                                train05, train06, train07, train08, train09,
                                train10, train11, test])

    # Build encoding model
    print("Loading movie fMRI data")
    # Load fMRI data
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Split the fmri train data to match features (12 parts)
    fmri_train00 = fmri_train[:300]
    fmri_train01 = fmri_train[300:600]
    fmri_train02 = fmri_train[600:900]
    fmri_train03 = fmri_train[900:1200]
    fmri_train04 = fmri_train[1200:1500]
    fmri_train05 = fmri_train[1500:1800]
    fmri_train06 = fmri_train[1800:2100]
    fmri_train07 = fmri_train[2100:2400]
    fmri_train08 = fmri_train[2400:2700]
    fmri_train09 = fmri_train[2700:3000]
    fmri_train10 = fmri_train[3000:3300]
    fmri_train11 = fmri_train[3300:]

    # Prep data
    train00_fmri = remove_nan(fmri_train00)
    train01_fmri = remove_nan(fmri_train01)
    train02_fmri = remove_nan(fmri_train02)
    train03_fmri = remove_nan(fmri_train03)
    train04_fmri = remove_nan(fmri_train04)
    train05_fmri = remove_nan(fmri_train05)
    train06_fmri = remove_nan(fmri_train06)
    train07_fmri = remove_nan(fmri_train07)
    train08_fmri = remove_nan(fmri_train08)
    train09_fmri = remove_nan(fmri_train09)
    train10_fmri = remove_nan(fmri_train10)
    train11_fmri = remove_nan(fmri_train11)
    test_fmri = remove_nan(fmri_test)

    fmri_arrays = np.vstack([train00_fmri, train01_fmri, train02_fmri,
                            train03_fmri, train04_fmri, train05_fmri,
                            train06_fmri, train07_fmri, train08_fmri,
                            train09_fmri, train10_fmri, train11_fmri,
                            test_fmri])

    correlations = []

    # For each of the 12 x,y pairs, we will train
    # a model on 11 and test using the held out one
    for i in range(len(fmri_arrays)):
        print("leaving out run", i)
        X_train = np.delete(feature_arrays, i, axis=0)
        Y_train = np.delete(fmri_arrays, i, axis=0)

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

        # Test the model
        X_test = feature_arrays[i]
        Y_test = fmri_arrays[i]

        # Predict
        Y_pred = np.dot(X_test, average_coef)

        test_correlations = calc_correlation(Y_pred, Y_test)

        print("Max correlation:", np.nanmax(test_correlations))

        correlations.append(test_correlations)

    print("Finished vision encoding model")

    print(correlations.shape)

    # Take average correlations over all runs
    average_correlations = np.mean(correlations, axis=0)

    return average_correlations


if __name__ == "__main__":
    if len(sys.argv) == 3:
        subject = sys.argv[1]
        layer = int(sys.argv[2])

        print("Building vision model")
        # Build encoding model
        correlations = vision_model(subject, layer)

        np.save('results/vision_mode/' + subject +
                '/layer' + str(layer) + '_correlations.npy', correlations)
    else:
        print("This script requires exactly two arguments: subject, modality, \
               and layer. Ex. python crossmodal.py S1 vision 1")
