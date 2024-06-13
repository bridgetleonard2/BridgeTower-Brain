import fun
import numpy as np
import sys


class CrossModal:
    def __init__(self, subject, modality, layer):
        self.subject = subject
        self.modality = modality
        self.layer = layer
        self.encoding_model = None
        self.correlations = None

    def build_encoding_model(self):
        if self.modality == "vision":
            print("Building vision model")
            # Build encoding model
            self.encoding_model = fun.vision_model(self.subject, self.layer)
        elif self.modality == "language":
            print("Building language model")
            # Build encoding model
            self.encoding_model = fun.language_model(self.subject, self.layer)

    def predict_fmri_data(self):
        if self.modality == "vision":
            print("Predicting fMRI data and calculating correlations")
            # Predict story fmri with vision model
            self.correlations = fun.story_prediction(self.subject, self.layer,
                                                     self.encoding_model)
            np.save(f'results/movie_to_story/{self.subject}/' +
                    f'layer{str(self.layer)}_correlations.npy',
                    self.correlations)
        elif self.modality == "language":
            print("Predicting fMRI data and calculating correlations")
            # Predict story fmri with language model
            self.correlations = fun.movie_predictions(self.subject, self.layer,
                                                      self.encoding_model)
            np.save(f'results/story_to_movie/{self.subject}/' +
                    f'layer{str(self.layer)}_correlations.npy',
                    self.correlations)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        subject = sys.argv[1]
        modality = sys.argv[2]
        layer = int(sys.argv[3])

        crossmodal = CrossModal(subject, modality, layer)
        crossmodal.build_encoding_model()
        crossmodal.predict_fmri_data()

    else:
        print("This script requires exactly two arguments: subject, modality, \
               and layer. Ex. python crossmodal.py S1 vision 1")
