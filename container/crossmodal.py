from transformers import BridgeTowerModel, BridgeTowerProcessor
import torch
import numpy as np
from torch.nn.functional import pad
import sys
import h5py
import re
from datasets import load_dataset
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from functions import prep_data
from functions import generate_leave_one_run_out
from functions import Delayer
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend

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
        with open("data/raw_stimuli/textgrids/stimuli/legacy.TextGrid", 'r') as file:
            data = file.readlines()

        full_transcript = []
        # Important info starts at line 5
        for line in data[5:]:
            if line.startswith('2'):
                index = data.index(line)
                word = re.search(r'"([^"]*)"', data[index+1].strip()).group(1)
                full_transcript.append(word)
    elif textgrid == 'data/raw_stimuli/textgrids/stimuli/life.TextGrid':
        with open("data/raw_stimuli/textgrids/stimuli/life.TextGrid", 'r') as file:
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
                word = re.search(r'"([^"]*)"', word_script[ind+3].strip()).group(1)
                full_transcript.append(word)

    return np.array(full_transcript)


def setup_model(layer):
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
    
    return model, processor, features, layer_selected


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
    model, processor, features, layer_selected = setup_model(layer)

    # create overall data structure for average feature vectors
    # a dictionary with layer names as keys and a list of vectors as it values
    data = {}

    # a dictionary to store vectors for n consecutive trials
    avg_data = {}

    # loop through all inputs
    for i, image in enumerate(movie_data):

        model_input = processor(image, "", return_tensors="pt")
        # Assuming model_input is a dictionary of tensors
        model_input = {key: value.to(device) for key, value in model_input.items()}

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

                if name not in data:
                    data[name] = []
                data[name].append(avg_feature)

            avg_data = {}

    layer_selected.remove()

    # Save data
    data = data[f'layer_{layer}'].cpu()
    data = data.numpy()

    return data


def alignment(layer):
    dataset = load_dataset("nlphuji/flickr30k")
    test_dataset = dataset['test']

    # Define Model
    model, processor, features, layer_selected = setup_model(layer)
    
    data = []

    for item in range(len(test_dataset)):
        image = test_dataset[item]['image']
        image_array = np.array(image)
        caption = " ".join(test_dataset[item]['caption'])

        # Run image
        image_input = processor(image_array, "", return_tensors="pt")
        image_input = {key: value.to(device) for key, value in image_input.items()}

        _ = model(**image_input)

        image_vector = features['layer_8']

        # Run caption
        # Create a numpy array filled with gray values (128 in this case)
        # THis will act as tthe zero image input***
        gray_value = 128
        gray_image_array = np.full((512, 512, 3), gray_value, dtype=np.uint8)

        caption_input = processor(gray_image_array, caption,
                                    return_tensors="pt")
        caption_input = {key: value.to(device) for key, value in caption_input.items()}
        _ = model(**caption_input)

        caption_vector = features['layer_8']

        data.append([image_vector, caption_vector])

    # Run encoding model
    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

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

    _ = pipeline.fit(captions, images)
    coef_captions_to_images = backend.to_numpy(pipeline[-1].coef_)

    return coef_images_to_captions, coef_captions_to_images


def movie_to_story(subject, layer):

    # Extract features from raw stimuli
    train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf', layer)
    train01 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_01.hdf', layer)
    train02 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_02.hdf', layer)
    train03 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_03.hdf', layer)
    train04 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_04.hdf', layer)
    train05 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_05.hdf', layer)
    train06 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_06.hdf', layer)
    train07 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_07.hdf', layer)
    train08 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_08.hdf', layer)
    train09 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_09.hdf', layer)
    train10 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_10.hdf', layer)
    train11 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_11.hdf', layer)
    test = get_movie_features('data/raw_stimuli/shortclips/stimuli/test.hdf')

    # Project into cross modal space
    ## Build alignment matrix
    coef_images_to_captions, coef_captions_to_images = alignment(layer)

    # Build encoding model
    

if __name__ == "__main__":
    if len(sys.argv) == 4: 
        subject = sys.argv[1]
        modality = sys.argv[2]
        layer = sys.argv[3]

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

        # Extract features from raw stimuli
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')
        train00 = get_movie_features('data/raw_stimuli/shortclips/stimuli/train_00.hdf')

        # Project into cross modal space

        # Build encoding model and predict
    else:
        print("This script requires exactly two arguments: subject, modality, and layer. \
              Ex. python crossmodal.py S1 vision 1")
        