from transformers import BridgeTowerModel, BridgeTowerProcessor
import torch
import numpy as np
from torch.nn.functional import pad
import sys
import h5py
import re


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
    print("loading HDF array")
    movie_data = load_hdf5_array(movie_data, key='stimuli')

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
    layer_1 = model.cross_modal_text_transform.register_forward_hook(
        get_features('layer_1'))
    layer_2 = model.cross_modal_image_transform.register_forward_hook(
        get_features('layer_2'))
    layer_3 = model.token_type_embeddings.register_forward_hook(
        get_features('layer_3'))
    layer_4 = model.vision_model.visual.ln_post.register_forward_hook(
        get_features('layer_4'))
    layer_5 = model.text_model.encoder.layer[-1].output.register_forward_hook(
        get_features('layer_5'))
    layer_6 = model.cross_modal_image_layers[-1].output.register_forward_hook(
        get_features('layer_6'))
    layer_7 = model.cross_modal_text_layers[-1].output.register_forward_hook(
        get_features('layer_7'))
    layer_8 = model.cross_modal_image_pooler.register_forward_hook(
        get_features('layer_8'))
    layer_9 = model.cross_modal_text_pooler.register_forward_hook(
        get_features('layer_9'))
    layer_10 = model.cross_modal_text_layernorm.register_forward_hook(
        get_features('layer_10'))
    layer_11 = model.cross_modal_image_layernorm.register_forward_hook(
        get_features('layer_11'))
    layer_12 = model.cross_modal_text_link_tower[-1].register_forward_hook(
        get_features('layer_12'))
    layer_13 = model.cross_modal_image_link_tower[-1].register_forward_hook(
        get_features('layer_13'))

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

    layer_1.remove()
    layer_2.remove()
    layer_3.remove()
    layer_4.remove()
    layer_5.remove()
    layer_6.remove()
    layer_7.remove()
    layer_8.remove()
    layer_9.remove()
    layer_10.remove()
    layer_11.remove()
    layer_12.remove()
    layer_13.remove()

    # Save data
    torch.save(data['layer_1'], "results/feature_vectors/movie/layer1/" + raw_stim + "_data.pt")
    torch.save(data['layer_2'], "results/feature_vectors/movie/layer2/" + raw_stim + "_data.pt")
    torch.save(data['layer_3'], "results/feature_vectors/movie/layer3/" + raw_stim + "_data.pt")
    torch.save(data['layer_4'], "results/feature_vectors/movie/layer4/" + raw_stim + "_data.pt")
    torch.save(data['layer_5'], "results/feature_vectors/movie/layer5/" + raw_stim + "_data.pt")
    torch.save(data['layer_6'], "results/feature_vectors/movie/layer6/" + raw_stim + "_data.pt")
    torch.save(data['layer_7'], "results/feature_vectors/movie/layer7/" + raw_stim + "_data.pt")
    torch.save(data['layer_8'], "results/feature_vectors/movie/layer8/" + raw_stim + "_data.pt")
    torch.save(data['layer_9'], "results/feature_vectors/movie/layer9/" + raw_stim + "_data.pt")
    torch.save(data['layer_10'], "results/feature_vectors/movie/layer10/" + raw_stim + "_data.pt")
    torch.save(data['layer_11'], "results/feature_vectors/movie/layer11/" + raw_stim + "_data.pt")
    torch.save(data['layer_12'], "results/feature_vectors/movie/layer12/" + raw_stim + "_data.pt")
    torch.save(data['layer_13'], "results/feature_vectors/movie/layer13/" + raw_stim + "_data.pt")

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
    print("loading textgrid")
    story_data = textgrid_to_array(story_data)

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
    layer_1 = model.cross_modal_text_transform.register_forward_hook(
        get_features('layer_1'))
    layer_2 = model.cross_modal_image_transform.register_forward_hook(
        get_features('layer_2'))
    layer_3 = model.token_type_embeddings.register_forward_hook(
        get_features('layer_3'))
    layer_4 = model.vision_model.visual.ln_post.register_forward_hook(
        get_features('layer_4'))
    layer_5 = model.text_model.encoder.layer[-1].output.register_forward_hook(
        get_features('layer_5'))
    layer_6 = model.cross_modal_image_layers[-1].output.register_forward_hook(
        get_features('layer_6'))
    layer_7 = model.cross_modal_text_layers[-1].output.register_forward_hook(
        get_features('layer_7'))
    layer_8 = model.cross_modal_image_pooler.register_forward_hook(
        get_features('layer_8'))
    layer_9 = model.cross_modal_text_pooler.register_forward_hook(
        get_features('layer_9'))
    layer_10 = model.cross_modal_text_layernorm.register_forward_hook(
        get_features('layer_10'))
    layer_11 = model.cross_modal_image_layernorm.register_forward_hook(
        get_features('layer_11'))
    layer_12 = model.cross_modal_text_link_tower[-1].register_forward_hook(
        get_features('layer_12'))
    layer_13 = model.cross_modal_image_link_tower[-1].register_forward_hook(
        get_features('layer_13'))

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
        model_input = {key: value.to(device) for key, value in model_input.items()}

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

    layer_1.remove()
    layer_2.remove()
    layer_3.remove()
    layer_4.remove()
    layer_5.remove()
    layer_6.remove()
    layer_7.remove()
    layer_8.remove()
    layer_9.remove()
    layer_10.remove()
    layer_11.remove()
    layer_12.remove()
    layer_13.remove()

    # Save data
    torch.save(data['layer_1'], "results/feature_vectors/story/layer1/" + raw_stim + "_data.pt")
    torch.save(data['layer_2'], "results/feature_vectors/story/layer2/" + raw_stim + "_data.pt")
    torch.save(data['layer_3'], "results/feature_vectors/story/layer3/" + raw_stim + "_data.pt")
    torch.save(data['layer_4'], "results/feature_vectors/story/layer4/" + raw_stim + "_data.pt")
    torch.save(data['layer_5'], "results/feature_vectors/story/layer5/" + raw_stim + "_data.pt")
    torch.save(data['layer_6'], "results/feature_vectors/story/layer6/" + raw_stim + "_data.pt")
    torch.save(data['layer_7'], "results/feature_vectors/story/layer7/" + raw_stim + "_data.pt")
    torch.save(data['layer_8'], "results/feature_vectors/story/layer8/" + raw_stim + "_data.pt")
    torch.save(data['layer_9'], "results/feature_vectors/story/layer9/" + raw_stim + "_data.pt")
    torch.save(data['layer_10'], "results/feature_vectors/story/layer10/" + raw_stim + "_data.pt")
    torch.save(data['layer_11'], "results/feature_vectors/story/layer11/" + raw_stim + "_data.pt")
    torch.save(data['layer_12'], "results/feature_vectors/story/layer12/" + raw_stim + "_data.pt")
    torch.save(data['layer_13'], "results/feature_vectors/story/layer13/" + raw_stim + "_data.pt")

    return data


if __name__ == "__main__":
    if len(sys.argv) == 3: 
        data = sys.argv[1]
        raw_stim = sys.argv[2]
        if data == 'movie':
            features = get_movie_features("data/raw_stimuli/shortclips/stimuli/" + raw_stim + ".hdf")
        elif data == 'story':
            features = get_story_features("data/raw_stimuli/textgrids/stimuli/" + raw_stim + ".TextGrid")
    else:
        print("This script requires exactly two arguments: data type and stimulus. Ex. python feature_extraction.py movie train_00")