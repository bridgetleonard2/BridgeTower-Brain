from transformers import BridgeTowerModel, BridgeTowerProcessor
import torch
from torch.nn.functional import pad

# set-up
# model
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


def get_feature_vecs(stim_data, n=30):
    # create overall data structure for average feature vectors
    # a dictionary with layer names as keys and a list of vectors as it values
    data = {}

    # a dictionary to store vectors for n consecutive trials
    avg_data = {}

    # loop through all inputs
    for i, image in enumerate(stim_data):

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
                str(round((((i + 1) / n) / (stim_data.shape[0]/30)) * 100, 2))
                + "%" + " complete")
    return data
