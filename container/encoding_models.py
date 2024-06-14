import fun
import numpy as np


class EncodingModel:
    def __init__(self, subject, layer):
        self.subject = subject
        self.layer = layer
        self.encoding_model = None
        self.correlations = None

    def crossmodal_vision_model(self):
        print("Building vision model")
        self.encoding_model = fun.crossmodal_vision_model(self.subject,
                                                          self.layer)
        print("Predicting fMRI data and calculating correlations")
        self.correlations = fun.story_prediction(self.subject, self.layer,
                                                 self.encoding_model)
        np.save(f'results/movie_to_story/{self.subject}/' +
                f'layer{str(self.layer)}_correlations.npy',
                self.correlations)

    def crossmodal_language_model(self):
        print("Building language model")
        self.encoding_model = fun.crossmodal_language_model(self.subject,
                                                            self.layer)
        print("Predicting fMRI data and calculating correlations")
        self.correlations = fun.movie_prediction(self.subject, self.layer,
                                                 self.encoding_model)

    def withinmodal_vision_model(self):
        self.correlations = fun.withinmodal_vision_model(self.subject,
                                                         self.layer)

    def faceLand_vision_model(self, modality):
        self.encoding_model = fun.crossmodal_vision_model(self.subject,
                                                          self.layer)
        self.correlations = fun.faceLandscape_prediction(self.subject,
                                                         self.layer,
                                                         modality,
                                                         self.encoding_model)


if __name__ == "__main__":
    crossmodalVision = EncodingModel("S1", 8).crossmodal_vision_model()
    crossmodalLanguage = EncodingModel("S1", 8).crossmodal_language_model()
    withinmodalVision = EncodingModel("S1", 8).withinmodal_vision_model()
    faceVision = EncodingModel("S1", 8).faceLand_vision_model("face")
    landscapeVision = EncodingModel("S1", 8).faceLand_vision_model("landscape")
    print("Encoding models completed")
