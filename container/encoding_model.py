# Basics
import numpy as np
import sys
import re

# Data loading
import torch
from torch.nn.functional import pad
import h5py
from datasets import load_dataset

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
from functions import remove_nan, prep_data, generate_leave_one_run_out, \
    calc_correlation, Delayer


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


def textgrid_to_array(textgrid):
    """Function to load transcript from textgrid into a list.

    Parameters
    ----------
    textgrid: string
        TextGrid file name.

    Returns
    -------
    full_transcript : Array
        Array with each word in the story.
    """
    if textgrid == 'data/raw_stimuli/textgrids/stimuli/legacy.TextGrid':
        with open(textgrid, 'r')as file:
            data = file.readlines()

        full_transcript = []
        # Important info starts at line 5
        for line in data[5:]:
            if line.startswith('2'):
                index = data.index(line)
                word = re.search(r'"([^"]*)"', data[index+1].strip()).group(1)
                full_transcript.append(word)
    elif textgrid == 'data/raw_stimuli/textgrids/stimuli/life.TextGrid':
        with open(textgrid, 'r') as file:
            data = file.readlines()

        full_transcript = []
        for line in data:
            if "word" in line:
                index = data.index(line)
                words = data[index+6:]  # this is where first word starts

        for i, word in enumerate(words):
            if i % 3 == 0:
                word = re.search(r'"([^"]*)"', word.strip()).group(1)
                full_transcript.append(word)
    else:
        with open(textgrid, 'r') as file:
            data = file.readlines()

        # Important info starts at line 8
        for line in data[8:]:
            # We only want item [2] info because those are the words instead
            # of phonemes
            if "item [2]" in line:
                index = data.index(line)

        summary_info = [line.strip() for line in data[index+1:index+6]]
        print(summary_info)

        word_script = data[index+6:]
        full_transcript = []
        for line in word_script:
            if "intervals" in line:
                # keep track of which interval we're on
                ind = word_script.index(line)
                word = re.search(r'"([^"]*)"',
                                 word_script[ind+3].strip()).group(1)
                full_transcript.append(word)

    return np.array(full_transcript)


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
def get_movie_features(movie_data, layer, n=30):
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
    print("loading HDF array")
    movie_data = load_hdf5_array(movie_data, key='stimuli')

    print("Running movie through model")

    # Define Model
    device, model, processor, features, layer_selected = setup_model(layer)

    # create overall data structure for average feature vectors
    # a dictionary with layer names as keys and a list of vectors as it values
    data = {}

    # a dictionary to store vectors for n consecutive trials
    avg_data = {}

    # loop through all inputs
    for i, image in enumerate(movie_data):

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
                    avg_feature_numpy = avg_feature.cpu().numpy()
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
                                # Make a list with length of 2*dimensions - 1
                                # to insert pad later
                                pad_list = [0] * ((2*tensor[0].dim()) - 1)
                                pad_list.insert(
                                    p_dim, max_size - tensor.size(dim))
                                # print(tuple(pad_list))
                                padded_tensor = pad(tensor, tuple(pad_list))
                                padded_tensors.append(padded_tensor)

                    avg_feature = torch.mean(torch.stack(padded_tensors),
                                             dim=0)
                    avg_feature_numpy = avg_feature.cpu().numpy()
                    # print(len(avg_feature_numpy))

                if name not in data:
                    data[name] = []
                data[name].append(avg_feature_numpy)

            avg_data = {}

    layer_selected.remove()

    # Save data
    data = np.array(data[f"layer_{layer}"])
    # print(data.shape)

    return data


def get_story_features(story_data, layer, n=20):
    """Function to extract feature vectors for each word of a story.

    Parameters
    ----------
    story_data: Array
        An array containing each word of the story in order.
    n (optional): int
        Number of words to to pad the target word with for
        context (before and after).

    Returns
    -------
    data : Dictionary
        Dictionary where keys are the model layer from which activations are
        extracted. Values are lists representing activations of 768 dimensions
        over the course of each word in the story.
    """
    print("loading textgrid")
    story_data = textgrid_to_array(story_data)

    print("Running story through model")
    # Define Model
    device, model, processor, features, layer_selected = setup_model(layer)

    # Create a numpy array filled with gray values (128 in this case)
    # THis will act as tthe zero image input***
    gray_value = 128
    image_array = np.full((512, 512, 3), gray_value, dtype=np.uint8)

    # create overall data structure for average feature vectors
    # a dictionary with layer names as keys and a list of vectors as it values
    data = {}

    # loop through all inputs
    for i, word in enumerate(story_data):
        # if one of first 20 words, just pad with all the words before it
        if i < n:
            # collapse list of strings into a single one
            word_with_context = ' '.join(story_data[:(i+n)])
        # if one of last 20 words, just pad with all the words after it
        elif i > (len(story_data) - n):
            # collapse list of strings into a single one
            word_with_context = ' '.join(story_data[(i-n):])
            # collapse list of strings into a single one
        else:
            word_with_context = ' '.join(story_data[(i-n):(i+n)])

        model_input = processor(image_array, word_with_context,
                                return_tensors="pt")
        # Assuming model_input is a dictionary of tensors
        model_input = {key: value.to(device) for key,
                       value in model_input.items()}

        _ = model(**model_input)

        for name, tensor in features.items():
            if name not in data:
                data[name] = []
            numpy_tensor = tensor.cpu().numpy()

            data[name].append(numpy_tensor)

    layer_selected.remove()

    # Save data
    data = np.array(data[f'layer_{layer}'])

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
    data_path = 'data/raw_stimuli/shortclips/stimuli/'
    print("extracting features from data")
    
    # Extract features from raw stimuli
    train00 = get_movie_features(data_path + 'train_00.hdf', layer)
    train01 = get_movie_features(data_path + 'train_01.hdf', layer)
    train02 = get_movie_features(data_path + 'train_02.hdf', layer)
    train03 = get_movie_features(data_path + 'train_03.hdf', layer)
    train04 = get_movie_features(data_path + 'train_04.hdf', layer)
    train05 = get_movie_features(data_path + 'train_05.hdf', layer)
    train06 = get_movie_features(data_path + 'train_06.hdf', layer)
    train07 = get_movie_features(data_path + 'train_07.hdf', layer)
    train08 = get_movie_features(data_path + 'train_08.hdf', layer)
    train09 = get_movie_features(data_path + 'train_09.hdf', layer)
    train10 = get_movie_features(data_path + 'train_10.hdf', layer)
    train11 = get_movie_features(data_path + 'train_11.hdf', layer)
    test = get_movie_features(data_path + 'test.hdf', layer)

    # Build encoding model
    print("Load movie data")
    # Load fMRI data
    # Using all data for cross-modality encoding model
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Prep data
    train_fmri = remove_nan(fmri_train)
    test_fmri = remove_nan(fmri_test)

    fmri_arrays = [train_fmri, test_fmri]
    feature_arrays = [train00, train01, train02, train03, train04,
                      train05, train06, train07, train08, train09,
                      train10, train11, test]

    # Combine data
    Y_train = np.vstack(fmri_arrays)
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

    # # Calculate scores
    # scores = pipeline.score(X_test, Y_test)
    # print("(n_voxels,) =", scores.shape)
    # scores = backend.to_numpy(scores)

    # best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
    coef = pipeline[-1].coef_
    coef = backend.to_numpy(coef)
    print("(n_delays * n_features, n_voxels) =", coef.shape)

    # Regularize coefficients
    coef /= np.linalg.norm(coef, axis=0)[None]
    # coef *= np.sqrt(np.maximum(0, scores))[None]

    # split the ridge coefficients per delays
    delayer = pipeline.named_steps['delayer']
    coef_per_delay = delayer.reshape_by_delays(coef, axis=0)
    print("(n_delays, n_features, n_voxels) =", coef_per_delay.shape)
    del coef

    # average over delays
    average_coef = np.mean(coef_per_delay, axis=0)
    print("(n_features, n_voxels) =", average_coef.shape)
    del coef_per_delay

    return average_coef


def language_model(subject, layer):
    """Function to build the language encoding model. Creates a
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
    data_path = 'data/raw_stimuli/textgrids/stimuli/'
    # Extract features from raw stimuli
    alternateithicatom = get_story_features(data_path +
                                            'alternateithicatom.TextGrid',
                                            layer)
    avatar = get_story_features(data_path + 'avatar.TextGrid', layer)
    howtodraw = get_story_features(data_path + 'howtodraw.TextGrid', layer)
    legacy = get_story_features(data_path + 'legacy.TextGrid', layer)
    life = get_story_features(data_path + 'life.TextGrid', layer)
    yankees = get_story_features(data_path +
                                 'myfirstdaywiththeyankees.TextGrid', layer)
    naked = get_story_features(data_path +
                               'alternateithicatom.TextGrid', layer)
    ode = get_story_features(data_path + 'naked.TextGrid', layer)
    souls = get_story_features(data_path + 'odetostepfather.TextGrid', layer)
    undertheinfluence = get_story_features(data_path +
                                           'undertheinfluence.TextGrid', layer)

    # Build encoding model
    print('Load movie data')
    # Load fmri data
    # Using all data for cross-modality encoding model
    fmri_alternateithicatom = np.load("data/storydata/" + subject +
                                      "/alternateithicatom.npy")
    fmri_avatar = np.load("data/storydata/" + subject + "/avatar.npy")
    fmri_howtodraw = np.load("data/storydata/" + subject + "/howtodraw.npy")
    fmri_legacy = np.load("data/storydata/" + subject + "/legacy.npy")
    fmri_life = np.load("data/storydata/" + subject + "/life.npy")
    fmri_yankees = np.load("data/storydata/" + subject +
                           "/myfirstdaywiththeyankees.npy")
    fmri_naked = np.load("data/storydata/" + subject + "/naked.npy")
    fmri_ode = np.load("data/storydata/" + subject + "/odetostepfather.npy")
    fmri_souls = np.load("data/storydata/" + subject + "/souls.npy")
    fmri_undertheinfluence = np.load("data/storydata/" + subject +
                                     "/undertheinfluence.npy")

    # Prep data
    fmri_alternateithicatom, ai_features = prep_data(fmri_alternateithicatom,
                                                     alternateithicatom)
    fmri_avatar, avatar_features = prep_data(fmri_avatar, avatar)
    fmri_howtodraw, howtodraw_features = prep_data(fmri_howtodraw, howtodraw)
    fmri_legacy, legacy_features = prep_data(fmri_legacy, legacy)
    fmri_life, life_features = prep_data(fmri_life, life)
    fmri_yankees, yankees_features = prep_data(fmri_yankees, yankees)
    fmri_naked, naked_features = prep_data(fmri_naked, naked)
    fmri_ode, odetostepfather_features = prep_data(fmri_ode, ode)
    fmri_souls, souls_features = prep_data(fmri_souls, souls)
    fmri_undertheinfluence, under_features = prep_data(fmri_undertheinfluence,
                                                       undertheinfluence)

    fmri_arrays = [fmri_alternateithicatom, fmri_avatar, fmri_howtodraw,
                   fmri_legacy, fmri_life, fmri_yankees, fmri_naked,
                   fmri_ode, fmri_souls, fmri_undertheinfluence]
    feature_arrays = [ai_features, avatar_features, howtodraw_features,
                      legacy_features, life_features, yankees_features,
                      naked_features, odetostepfather_features,
                      souls_features, under_features]
    # Combine data
    Y_train = np.vstack(fmri_arrays)
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

    ridge_cv = RidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100))

    pipeline = make_pipeline(
        scaler,
        delayer,
        ridge_cv,
    )

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

    return average_coef


if __name__ == "__main__":
    if len(sys.argv) == 4:
        subject = sys.argv[1]
        modality = sys.argv[2]
        layer = int(sys.argv[3])

        if modality == "vision":
            print("Building vision model")
            # Build encoding model
            vision_encoding_matrix = vision_model(subject, layer)

            np.save(vision_encoding_matrix, 'results/vision_model/' + subject +
                    '/layer' + layer + '_vision_model.npy')

        elif modality == "language":
            print("Building language model")
            # Build encoding model
            language_encoding_model = language_model(subject, layer)

            np.save(language_encoding_model, 'results/language_model/' + subject +
                    '/layer' + layer + 'language_model.npy')

    else:
        print("This script requires exactly two arguments: subject, modality, \
               and layer. Ex. python crossmodal.py S1 vision 1")

