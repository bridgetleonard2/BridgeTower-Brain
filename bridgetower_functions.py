from transformers import BridgeTowerModel, BridgeTowerProcessor
import torch
import numpy as np
import re
import h5py
from scipy.signal import resample
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
    print("Running movie through model")
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

            # Create checkpoints
            ten = round((movie_data.shape[0]/30) * 0.1)
            twen = round((movie_data.shape[0]/30) * 0.2)
            thir = round((movie_data.shape[0]/30) * 0.3)
            four = round((movie_data.shape[0]/30) * 0.4)
            fif = round((movie_data.shape[0]/30) * 0.5)
            six = round((movie_data.shape[0]/30) * 0.6)
            sev = round((movie_data.shape[0]/30) * 0.7)
            eig = round((movie_data.shape[0]/30) * 0.8)
            nine = round((movie_data.shape[0]/30) * 0.9)

            if i == ten:
                print("10% done")
            elif i == twen:
                print("20% done")
            elif i == thir:
                print("30% done")
            elif i == four:
                print("40% done")
            elif i == fif:
                print("50% done")
            elif i == six:
                print("60% done")
            elif i == sev:
                print("70% done")
            elif i == eig:
                print("80% done")
            elif i == nine:
                print("90% done")
    print("Complete!")
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
    print("Running story through model")
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
        # Create checkpoints
        ten = round(len(story_data) * 0.1)
        twen = round(len(story_data) * 0.2)
        thir = round(len(story_data) * 0.3)
        four = round(len(story_data) * 0.4)
        fif = round(len(story_data) * 0.5)
        six = round(len(story_data) * 0.6)
        sev = round(len(story_data) * 0.7)
        eig = round(len(story_data) * 0.8)
        nine = round(len(story_data) * 0.9)

        if i == ten:
            print("10% done")
        elif i == twen:
            print("20% done")
        elif i == thir:
            print("30% done")
        elif i == four:
            print("40% done")
        elif i == fif:
            print("50% done")
        elif i == six:
            print("60% done")
        elif i == sev:
            print("70% done")
        elif i == eig:
            print("80% done")
        elif i == nine:
            print("90% done")
    print("Complete!")

    patch_embed.remove()
    return data


def remove_nan(data):
    mask = ~np.isnan(data)

    # Apply the mask and then flatten
    # This will keep only the non-NaN values
    data_reshaped = data[mask].reshape(data.shape[0], -1)

    print("fMRI shape:", data_reshaped.shape)
    return data_reshaped


def resample_to_acq(feature_data, fmri_data):
    dimensions = fmri_data.shape[0]
    data_transposed = feature_data.T
    data_resampled = np.empty((data_transposed.shape[0], dimensions))

    for i in range(data_transposed.shape[0]):
        data_resampled[i, :] = resample(data_transposed[i, :],
                                        dimensions, window=('kaiser', 14))

    print("Shape after resampling:", data_resampled.T.shape)
    return data_resampled.T


def prep_data(fmri_data, feature_data):
    fmri_reshaped = remove_nan(fmri_data)

    feature_resampled = resample_to_acq(feature_data, fmri_reshaped)

    return fmri_reshaped, feature_resampled


def safe_correlation(x, y):
    """Calculate the Pearson correlation coefficient safely."""
    # Mean centering
    x_mean = x - np.mean(x)
    y_mean = y - np.mean(y)
    
    # Numerator: sum of the product of mean-centered variables
    numerator = np.sum(x_mean * y_mean)
    
    # Denominator: sqrt of the product of the sums of squared mean-centered variables
    denominator = np.sqrt(np.sum(x_mean**2) * np.sum(y_mean**2))
    
    # Safe division
    if denominator == 0:
        # Return NaN or another value to indicate undefined correlation
        return np.nan
    else:
        return numerator / denominator


def calc_correlation(predicted_fMRI, real_fMRI):
    # Calculate correlations for each voxel
    correlation_coefficients = np.array([safe_correlation(predicted_fMRI[:, i], real_fMRI[:, i]) for i in range(predicted_fMRI.shape[1])])

    # Check for NaNs in the result to identify voxels with undefined correlations
    nans_in_correlations = np.isnan(correlation_coefficients).any()
    print(f"NaNs in correlation coefficients: {nans_in_correlations}")

    return correlation_coefficients
