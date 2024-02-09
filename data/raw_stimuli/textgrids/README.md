## An fMRI dataset during a passive natural language listening task

This dataset now has a dataset descriptor currently available in Scientific Data, that describes all of the data and code available for working with this dataset. It can be found:

LeBel, A., Wagner, L., Jain, S. et al. A natural language fMRI dataset for voxelwise encoding models. Sci Data 10, 555 (2023). https://doi.org/10.1038/s41597-023-02437-z

A(n incomplete) list of papers using this dataset from our group are listed below:

Tang, J., LeBel, A., Jain, S. et al. Semantic reconstruction of continuous language from non-invasive brain recordings. Nat Neurosci (2023). https://doi.org/10.1038/s41593-023-01304-9

LeBel, A., Jain, S. & Huth, A. G. Voxelwise Encoding Models Show That Cerebellar Language Representations Are Highly Conceptual. J. Neurosci. 41, 10341â€“10355 (2021)

Tang, J., LeBel, A. & Huth, A. G. Cortical Representations of Concrete and Abstract Concepts in Language Combine Visual and Linguistic Representations. bioRxiv 2021.05.19.444701 (2021) doi:10.1101/2021.05.19.444701

Jain, S. et al. Interpretable multi-timescale models for predicting fMRI responses to continuous natural speech. Advances in Neural Information Processing Systems 34, (2020)

## Dataset Derivatives

1. preprocessed data: fully preprocessed data as described in previous works.

2. textgrids: aligned transcripts of the stimulus with start and end point for each word and phoneme. 

3. pycortex-db: hand-corrected surfaces for each subject to be used in visualization. This is best used with the pycortex software. 

4. subject_xfms.json: a dictionary with the correct transformation for each subject to align the data to the surface.

5. respdict.json: a dictionary with the number of TRs for each story in the stimulus set.
