# Basics
import numpy as np
import os

# Data loading
from PIL import Image
import torch
from torch.nn.functional import pad
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
import utils

# Progress bar
from tqdm import tqdm


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
        movie_data = utils.load_hdf5_array(f"{data_path}{movie}.hdf",
                                           key='stimuli')

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


def get_story_features(story, subject, layer, n=20):
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
    try:
        data = np.load(f"results/features/story/{subject}/" +
                       f"layer{layer}_{story}.npy")
        print("Loaded story features")
    except FileNotFoundError:
        data_path = 'data/raw_stimuli/textgrids/stimuli/'
        print("loading textgrid")

        story_data = utils.textgrid_to_array(f"{data_path}{story}.TextGrid")

        # Define Model
        device, model, processor, features, layer_selected = setup_model(layer)

        # Create a numpy array filled with gray values (128 in this case)
        # THis will act as tthe zero image input***
        gray_value = 128
        image_array = np.full((512, 512, 3), gray_value, dtype=np.uint8)

        # create overall data structure for average feature vectors
        # a dictionary with layer names as keys and a list of vectors
        # as it values
        data = {}

        print("Running story through model")
        # loop through all inputs
        for i, word in tqdm(enumerate(story_data)):
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
                numpy_tensor = tensor.detach().cpu().numpy()

                data[name].append(numpy_tensor)

        layer_selected.remove()

        # Save data
        data = np.array(data[f'layer_{layer}'])
        print("Got story features")

        np.save(f"results/features/story/{subject}/layer{layer}_{story}.npy",
                data)

    return data


def alignment(layer):
    """Function generate matrices for feature alignment. Capture the
    linear relationship between caption features and image features
    output by a specific layer of the BridgeTower model.

    Parameters
    ----------
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer

    Returns
    -------
    coef_images_to_captions : Array
        Array of shape (layer_output_size, layer_output_size) mapping
        the relationship of image features to caption features.
    coef_captions_to_images: Array
        Array of shape (layer_output_size, layer_output_size) mapping
            the relationship of caption features to image features.
    """
    # Check if alignment is already done
    try:
        coef_images_to_captions = np.load(f'results/alignment/layer_{layer}/'
                                          'coef_images_to_captions.npy')
        coef_captions_to_images = np.load(f'results/alignment/layer_{layer}/'
                                          'coef_captions_to_images.npy')
        print("Alignment already done, retrieving coefficients")
    except FileNotFoundError:
        print("Starting feature alignment")
        # Stream the dataset so it doesn't download to device
        test_dataset = load_dataset("nlphuji/flickr30k", split='test',
                                    streaming=True)

        # Define Model
        device, model, processor, features, layer_selected = setup_model(layer)

        data = []

        print("Running flickr through model")
        # Assuming 'test_dataset' is an IterableDataset from a streaming source
        for item in tqdm(test_dataset):
            # Access data directly from the item, no need for indexing
            image = item['image']
            image_array = np.array(image)
            caption = " ".join(item['caption'])

            # Run image
            image_input = processor(image_array, "", return_tensors="pt")
            image_input = {key: value.to(device)
                           for key, value in image_input.items()}

            _ = model(**image_input)
            image_vector = features[f'layer_{layer}']

            # Run caption
            # Create a numpy array filled with gray values (128 in this case)
            # This will act as the zero image input
            gray_value = 128
            gray_image_array = np.full((512, 512, 3), gray_value,
                                       dtype=np.uint8)

            caption_input = processor(gray_image_array, caption,
                                      return_tensors="pt")
            caption_input = {key: value.to(device)
                             for key, value in caption_input.items()}
            _ = model(**caption_input)

            caption_vector = features[f'layer_{layer}']

            data.append([image_vector.detach().cpu().numpy(),
                        caption_vector.detach().cpu().numpy()])

        # Run encoding model
        backend = set_backend("torch_cuda", on_error="warn")
        print(backend)

        data = np.array(data)
        # Test data
        print(data.shape)
        # Variables
        captions = data[:, 1, :]
        images = data[:, 0, :]

        alphas = np.logspace(1, 20, 20)
        scaler = StandardScaler(with_mean=True, with_std=False)

        ridge_cv = RidgeCV(
            alphas=alphas, cv=5,
            solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                               n_targets_batch_refit=100))

        pipeline = make_pipeline(
            scaler,
            ridge_cv
        )

        _ = pipeline.fit(images, captions)
        coef_images_to_captions = backend.to_numpy(pipeline[-1].coef_)
        coef_images_to_captions /= np.linalg.norm(coef_images_to_captions,
                                                  axis=0)[None]

        _ = pipeline.fit(captions, images)
        coef_captions_to_images = backend.to_numpy(pipeline[-1].coef_)
        coef_captions_to_images /= np.linalg.norm(coef_captions_to_images,
                                                  axis=0)[None]

        print("Finished feature alignment, saving coefficients")
        # Save coefficients
        np.save(f'results/alignment/layer_{layer}/coef_images_to_captions.npy',
                coef_images_to_captions)
        np.save(f'results/alignment/layer_{layer}/coef_captions_to_images.npy',
                coef_captions_to_images)
    return coef_images_to_captions, coef_captions_to_images


def crossmodal_vision_model(subject, layer):
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

    # Build encoding model
    print("Loading movie fMRI data")
    # Load fMRI data
    # Using all data for cross-modality encoding model
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Prep data
    train_fmri = utils.remove_nan(fmri_train)
    test_fmri = utils.remove_nan(fmri_test)

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
    cv = utils.generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # cross-validation splitter into a reusable list

    # Define the model
    scaler = StandardScaler(with_mean=True, with_std=False)

    delayer = utils.Delayer(delays=[1, 2, 3, 4])

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
    np.save(f'results/movie_to_story/{subject}/' +
            f'layer{str(layer)}_correlations.npy', average_coef)

    return average_coef


def crossmodal_language_model(subject, layer):
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
    print("Extracting features from data")

    # Extract features from raw stimuli
    alternateithicatom = get_story_features('alternateithicatom', subject,
                                            layer)
    avatar = get_story_features('avatar', subject, layer)
    howtodraw = get_story_features('howtodraw', subject, layer)
    legacy = get_story_features('legacy', subject, layer)
    life = get_story_features('life', subject, layer)
    yankees = get_story_features('myfirstdaywiththeyankees', subject,
                                 layer)
    naked = get_story_features('naked', subject, layer)
    ode = get_story_features('odetostepfather', subject, layer)
    souls = get_story_features('souls', subject, layer)
    undertheinfluence = get_story_features('undertheinfluence',
                                           subject, layer)

    # Build encoding model
    print('Load story fMRI data')
    # Load fmri data
    # Using all data for cross-modality encoding model
    fmri_alternateithicatom = np.load(f"data/storydata/{subject}/" +
                                      "alternateithicatom.npy")
    fmri_avatar = np.load(f"data/storydata/{subject}/avatar.npy")
    fmri_howtodraw = np.load(f"data/storydata/{subject}/howtodraw.npy")
    fmri_legacy = np.load(f"data/storydata/{subject}/legacy.npy")
    fmri_life = np.load(f"data/storydata/{subject}/life.npy")
    fmri_yankees = np.load(f"data/storydata/{subject}/" +
                           "myfirstdaywiththeyankees.npy")
    fmri_naked = np.load(f"data/storydata/{subject}/naked.npy")
    fmri_ode = np.load(f"data/storydata/{subject}/odetostepfather.npy")
    fmri_souls = np.load(f"data/storydata/{subject}/souls.npy")
    fmri_undertheinfluence = np.load(f"data/storydata/{subject}/" +
                                     "undertheinfluence.npy")

    print(alternateithicatom.shape)
    # Prep data
    fmri_ai, ai_features = utils.prep_data(fmri_alternateithicatom,
                                           alternateithicatom)
    fmri_avatar, avatar_features = utils.prep_data(fmri_avatar, avatar)
    fmri_howtodraw, howtodraw_features = utils.prep_data(fmri_howtodraw,
                                                         howtodraw)
    fmri_legacy, legacy_features = utils.prep_data(fmri_legacy, legacy)
    fmri_life, life_features = utils.prep_data(fmri_life, life)
    fmri_yankees, yankees_features = utils.prep_data(fmri_yankees, yankees)
    fmri_naked, naked_features = utils.prep_data(fmri_naked, naked)
    fmri_ode, odetostepfather_features = utils.prep_data(fmri_ode, ode)
    fmri_souls, souls_features = utils.prep_data(fmri_souls, souls)
    fmri_under, under_features = utils.prep_data(fmri_undertheinfluence,
                                                 undertheinfluence)

    fmri_arrays = [fmri_ai, fmri_avatar, fmri_howtodraw,
                   fmri_legacy, fmri_life, fmri_yankees, fmri_naked,
                   fmri_ode, fmri_souls, fmri_under]
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
    cv = utils.generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # cross-validation splitter into a reusable list

    # Define the model
    scaler = StandardScaler(with_mean=True, with_std=False)

    delayer = utils.Delayer(delays=[1, 2, 3, 4])

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

    print("Finished language encoding model")
    np.save(f'results/story_to_movie/{subject}/' +
            f'layer{str(layer)}_correlations.npy', average_coef)

    return average_coef


def story_prediction(subject, layer, vision_encoding_matrix):
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
    _, coef_captions_to_images = alignment(layer)

    # Get story features
    alternateithicatom = get_story_features('alternateithicatom', subject,
                                            layer)
    avatar = get_story_features('avatar', subject, layer)
    howtodraw = get_story_features('howtodraw', subject, layer)
    legacy = get_story_features('legacy', subject, layer)
    life = get_story_features('life', subject, layer)
    yankees = get_story_features('myfirstdaywiththeyankees', subject,
                                 layer)
    naked = get_story_features('naked', subject, layer)
    ode = get_story_features('odetostepfather', subject, layer)
    souls = get_story_features('souls', subject, layer)
    undertheinfluence = get_story_features('undertheinfluence',
                                           subject, layer)

    # Project features into opposite space
    alternateithicatom_transformed = np.dot(alternateithicatom,
                                            coef_captions_to_images.T)
    avatar_transformed = np.dot(avatar, coef_captions_to_images.T)
    howtodraw_transformed = np.dot(howtodraw, coef_captions_to_images.T)
    legacy_transformed = np.dot(legacy, coef_captions_to_images.T)
    life_transformed = np.dot(life, coef_captions_to_images.T)
    yankees_transformed = np.dot(yankees, coef_captions_to_images.T)
    naked_transformed = np.dot(naked, coef_captions_to_images.T)
    ode_transformed = np.dot(ode, coef_captions_to_images.T)
    souls_transformed = np.dot(souls, coef_captions_to_images.T)
    undertheinfluence_transformed = np.dot(undertheinfluence,
                                           coef_captions_to_images.T)

    # Load fmri data
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
    fmri_ai, ai_features = utils.prep_data(fmri_alternateithicatom,
                                           alternateithicatom_transformed)
    fmri_avatar, avatar_features = utils.prep_data(fmri_avatar,
                                                   avatar_transformed)
    fmri_howtodraw, howtodraw_features = utils.prep_data(fmri_howtodraw,
                                                         howtodraw_transformed)
    fmri_legacy, legacy_features = utils.prep_data(fmri_legacy,
                                                   legacy_transformed)
    fmri_life, life_features = utils.prep_data(fmri_life, life_transformed)
    fmri_yankees, yankees_features = utils.prep_data(fmri_yankees,
                                                     yankees_transformed)
    fmri_naked, naked_features = utils.prep_data(fmri_naked, naked_transformed)
    fmri_ode, odetostepfather_features = utils.prep_data(fmri_ode,
                                                         ode_transformed)
    fmri_souls, souls_features = utils.prep_data(fmri_souls, souls_transformed)
    fmri_under, under_features = utils.prep_data(fmri_undertheinfluence,
                                                 undertheinfluence_transformed)

    # Make fmri predictions
    ai_predictions = np.dot(ai_features, vision_encoding_matrix)
    avatar_predictions = np.dot(avatar_features, vision_encoding_matrix)
    howtodraw_predictions = np.dot(howtodraw_features, vision_encoding_matrix)
    legacy_predictions = np.dot(legacy_features, vision_encoding_matrix)
    life_predictions = np.dot(life_features, vision_encoding_matrix)
    yankees_predictions = np.dot(yankees_features, vision_encoding_matrix)
    naked_predictions = np.dot(naked_features, vision_encoding_matrix)
    odetostepfather_predictions = np.dot(odetostepfather_features,
                                         vision_encoding_matrix)
    souls_predictions = np.dot(souls_features, vision_encoding_matrix)
    under_predictions = np.dot(under_features, vision_encoding_matrix)

    # Calculate correlations
    ai_correlations = utils.calc_correlation(ai_predictions, fmri_ai)
    avatar_correlations = utils.calc_correlation(avatar_predictions,
                                                 fmri_avatar)
    howtodraw_correlations = utils.calc_correlation(howtodraw_predictions,
                                                    fmri_howtodraw)
    legacy_correlations = utils.calc_correlation(legacy_predictions,
                                                 fmri_legacy)
    life_correlations = utils.calc_correlation(life_predictions, fmri_life)
    yankees_correlations = utils.calc_correlation(yankees_predictions,
                                                  fmri_yankees)
    naked_correlations = utils.calc_correlation(naked_predictions, fmri_naked)
    ode_correlations = utils.calc_correlation(odetostepfather_predictions,
                                              fmri_ode)
    souls_correlations = utils.calc_correlation(souls_predictions, fmri_souls)
    under_correlations = utils.calc_correlation(under_predictions, fmri_under)

    # Get mean correlation
    all_correlations = np.stack((ai_correlations, avatar_correlations,
                                 howtodraw_correlations, legacy_correlations,
                                 life_correlations, yankees_correlations,
                                 naked_correlations, ode_correlations,
                                 souls_correlations, under_correlations))

    story_correlations = np.nanmean(all_correlations, axis=0)
    print("Max correlation:", np.nanmax(story_correlations))

    return story_correlations


def movie_predictions(subject, layer, language_encoding_model):
    """Function to run the language encoding model. Predicts brain activity
    to movie watching and return correlations between predictions and real
    brain activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.
    language_encoding_matrix: array
        Generated by language_model() function. A matrix mapping the
        relationship between feature vectors and brain activity

    Returns
    -------
    correlations: Array
        Array of shape (num_voxels) representing the correlation between
        predictions and real brain activity for each voxel.
    """
    coef_images_to_captions, _ = alignment(layer)

    # Get movie features
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

    # Project features into opposite space
    test_transformed = np.dot(test, coef_images_to_captions.T)
    train00_transformed = np.dot(train00, coef_images_to_captions.T)
    train01_transformed = np.dot(train01, coef_images_to_captions.T)
    train02_transformed = np.dot(train02, coef_images_to_captions.T)
    train03_transformed = np.dot(train03, coef_images_to_captions.T)
    train04_transformed = np.dot(train04, coef_images_to_captions.T)
    train05_transformed = np.dot(train05, coef_images_to_captions.T)
    train06_transformed = np.dot(train06, coef_images_to_captions.T)
    train07_transformed = np.dot(train07, coef_images_to_captions.T)
    train08_transformed = np.dot(train08, coef_images_to_captions.T)
    train09_transformed = np.dot(train09, coef_images_to_captions.T)
    train10_transformed = np.dot(train10, coef_images_to_captions.T)
    train11_transformed = np.dot(train11, coef_images_to_captions.T)

    # Load fmri data
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Prep data
    fmri_train = utils.remove_nan(fmri_train)
    fmri_test = utils.remove_nan(fmri_test)

    # Make fmri predictions
    feature_arrays = [train00_transformed, train01_transformed,
                      train02_transformed, train03_transformed,
                      train04_transformed, train05_transformed,
                      train06_transformed, train07_transformed,
                      train08_transformed, train09_transformed,
                      train10_transformed, train11_transformed]

    features_train = np.vstack(feature_arrays)
    features_test = test_transformed

    predictions_train = np.dot(features_train, language_encoding_model)
    predictions_test = np.dot(features_test, language_encoding_model)

    # Calculate correlations
    correlations_train = utils.calc_correlation(predictions_train, fmri_train)
    correlations_test = utils.calc_correlation(predictions_test, fmri_test)

    # Get mean correlation
    all_correlations = np.stack((correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_test))

    correlations = np.nanmean(all_correlations, axis=0)
    print('max correlation', np.nanmax(correlations))

    return correlations


def withinmodal_vision_model(subject, layer):
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

    feature_arrays = [train00, train01, train02, train03, train04,
                      train05, train06, train07, train08, train09,
                      train10, train11, test]

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
    train00_fmri = utils.remove_nan(fmri_train00)
    train01_fmri = utils.remove_nan(fmri_train01)
    train02_fmri = utils.remove_nan(fmri_train02)
    train03_fmri = utils.remove_nan(fmri_train03)
    train04_fmri = utils.remove_nan(fmri_train04)
    train05_fmri = utils.remove_nan(fmri_train05)
    train06_fmri = utils.remove_nan(fmri_train06)
    train07_fmri = utils.remove_nan(fmri_train07)
    train08_fmri = utils.remove_nan(fmri_train08)
    train09_fmri = utils.remove_nan(fmri_train09)
    train10_fmri = utils.remove_nan(fmri_train10)
    train11_fmri = utils.remove_nan(fmri_train11)
    test_fmri = utils.remove_nan(fmri_test)

    fmri_arrays = [train00_fmri, train01_fmri, train02_fmri,
                   train03_fmri, train04_fmri, train05_fmri,
                   train06_fmri, train07_fmri, train08_fmri,
                   train09_fmri, train10_fmri, train11_fmri,
                   test_fmri]

    correlations = []

    # For each of the 12 x,y pairs, we will train
    # a model on 11 and test using the held out one
    for i in range(len(feature_arrays)):
        print("leaving out run", i)
        new_feat_arrays = utils.remove_run(feature_arrays, i)
        X_train = np.vstack(new_feat_arrays)
        Y_train = np.vstack(utils.remove_run(fmri_arrays, i))

        print("X_train shape", X_train.shape)
        # Define cross-validation
        run_onsets = []
        current_index = 0
        for arr in new_feat_arrays:
            next_index = current_index + arr.shape[0]
            run_onsets.append(current_index)
            current_index = next_index

        print(run_onsets)
        n_samples_train = X_train.shape[0]
        cv = utils.generate_leave_one_run_out(n_samples_train, run_onsets)
        cv = check_cv(cv)  # cross-validation splitter into a reusable list

        # Define the model
        scaler = StandardScaler(with_mean=True, with_std=False)
        delayer = utils.Delayer(delays=[1, 2, 3, 4])
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

        test_correlations = utils.calc_correlation(Y_pred, Y_test)

        print("Max correlation:", np.nanmax(test_correlations))

        correlations.append(test_correlations)

    print("Finished vision encoding model")

    # Make correlations np array
    correlations = np.array(correlations)
    print(correlations.shape)

    # Take average correlations over all runs
    average_correlations = np.nanmean(correlations, axis=0)

    np.save('results/vision_model/' + subject +
            '/layer' + str(layer) + '_correlations.npy', average_correlations)

    return average_correlations


def faceLandscape_prediction(subject, modality, layer, vision_encoding_matrix):
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
    if modality == 'face':
        data_path = 'data/face_stimuli'
    elif modality == 'landscape':
        data_path = 'data/landscape_stimuli'
    else:
        print("Invalid modality. Please choose 'face' or 'landscape'.")

    # Define Model
    device, model, processor, features, layer_selected = setup_model(layer)

    # Initiate data dict
    data = {}

    print("number of images:", len(os.listdir(data_path)))
    # Get face features
    for i, image_filename in tqdm(enumerate(os.listdir(data_path))):
        # Load image as PIL
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg',
                                            '.bmp', '.gif')):
            image_path = os.path.join(data_path, image_filename)
            try:
                image = Image.open(image_path).convert('RGB')
                model_input = processor(image, "", return_tensors="pt")
                model_input = {key: value.to(device) for
                               key, value in model_input.items()}
            except Exception as e:
                print(f"Failed to process {image_filename}: {str(e)}")

        _ = model(**model_input)

        for name, tensor in features.items():
            if name not in data:
                data[name] = []
            numpy_tensor = tensor.detach().cpu().numpy()

            data[name].append(numpy_tensor)

        layer_selected.remove()

    # Save data
    data = np.array(data[f"layer_{layer}"])

    # Data should be 2d of shape (n_images/n, num_features)
    # if data is above 2d, average 2nd+ dimensions
    if data.ndim > 2:
        data = np.mean(data, axis=1)

    print("Got face features")

    print('encoding matrix shape:', vision_encoding_matrix.shape)
    print('data shape:', data.shape)
    # Make fmri predictions
    fmri_predictions = np.dot(data, vision_encoding_matrix)
    print('predictions shape:', fmri_predictions.shape)

    average_predictions = np.mean(fmri_predictions, axis=0)

    if modality == 'face':
        np.save('results/faces/' + subject +
                '/layer' + str(layer) + '_predictions.npy',
                average_predictions)
    elif modality == 'landscape':
        np.save('results/landscapes/' + subject +
                '/layer' + str(layer) + '_predictions.npy',
                average_predictions)

    return average_predictions