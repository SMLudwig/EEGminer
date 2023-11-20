# EEGminer: Discovering Interpretable Features of Brain Activity with Learnable Filters

Siegfried Ludwig, Stylianos Bakas, Dimitrios A. Adamos, Nikolaos Laskaris, Yannis Panagakis, Stefanos Zafeiriou

Objective: The patterns of brain activity associated with different brain processes can be used to identify different brain states and make behavioral predictions. However, the relevant features are not readily apparent and accessible. Our aim is to design a system for learning informative latent representations from multichannel recordings of ongoing EEG activity.
Approach: We propose a novel differentiable decoding pipeline consisting of learnable filters and a pre-determined feature extraction module. Specifically, we introduce filters parameterized by generalized Gaussian functions that offer a smooth derivative for stable end-to-end model training and allow for learning interpretable features. For the feature module, we use signal magnitude and functional connectivity estimates.
Main Results: We demonstrate the utility of our model on a new EEG dataset of unprecedented size (i.e., 761 subjects), where we identify consistent trends of music perception and related individual differences. Furthermore, we train and apply our model in two additional datasets, specifically for emotion recognition on SEED and workload classification on STEW. The discovered features align well with previous neuroscience studies and offer new insights, such as marked differences in the functional connectivity profile between left and right temporal areas during music listening. This agrees with the specialisation of the temporal lobes regarding music perception proposed in the literature.
Significance: The proposed method offers strong interpretability of learned features while reaching similar levels of accuracy achieved by black box deep learning models. This improved trustworthiness may promote the use of deep learning models in real world applications.

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
