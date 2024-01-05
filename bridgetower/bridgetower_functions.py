from transformers import BridgeTowerModel, BridgeTowerProcessor
import torch
import numpy as np
import re
import h5py
from torch.nn.functional import pad


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


def get_movie_features(movie_data, n=30):
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
    print("running" + "f{movie_data}")
    # set-up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
    model = model.to(device)

    # placeholder for batch features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            # detached_outputs = [tensor.detach() for tensor in output]
            last_output = output[-1].detach()
            features[name] = last_output  # detached_outputs
        return hook

    # register forward hooks with layers of choice
    # First, convolutional layers
    patch_embed = model.cross_modal_image_pooler.register_forward_hook(
        get_features('layer_8'))

    processor = BridgeTowerProcessor.from_pretrained(
        "BridgeTower/bridgetower-base")

    # create overall data structure for average feature vectors
    # a dictionary with layer names as keys and a list of vectors as it values
    data = {}

    # a dictionary to store vectors for n consecutive trials
    avg_data = {}

    # loop through all inputs
    for i, image in enumerate(movie_data):

        model_input = processor(image, "", return_tensors="pt")
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
            print(
                str(round((((i + 1) / n) / (movie_data.shape[0]/30)) * 100, 2))
                + "%" + " complete")
    patch_embed.remove()
    return data


def get_story_features(story_data, n=20):
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
    print("running" + "f{story_data}")
    # set-up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
    model = model.to(device)

    # placeholder for batch features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            # detached_outputs = [tensor.detach() for tensor in output]
            last_output = output[-1].detach()
            features[name] = last_output  # detached_outputs
        return hook

    # register forward hooks with layers of choice
    # First, convolutional layers
    patch_embed = model.cross_modal_image_pooler.register_forward_hook(
        get_features('layer_8'))

    processor = BridgeTowerProcessor.from_pretrained(
        "BridgeTower/bridgetower-base")

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
        _ = model(**model_input)

        for name, tensor in features.items():
            if name not in data:
                data[name] = []
            data[name].append(tensor)

        # Print some checkpoints
        quarter = round(len(story_data) * 0.25)
        half = round(len(story_data) * 0.5)
        threequarter = round(len(story_data) * 0.75)
        if i == quarter:
            print("25% complete")
        elif i == half:
            print("50% complete")
        elif i == threequarter:
            print("75% complete")

    patch_embed.remove()
    return data
