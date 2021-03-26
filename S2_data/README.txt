Behavioural data collected for the study "A Bayesian Joint Prior model explains
Binding and Fusion effects in Audiovisual Speech Perception"
(Lindborg & Andersen, submitted to PLOS Computational Biology 2020).
Further information about the experimental paradigm can be found in this study.
The data are open access - however, please make sure to cite the study in any
publications building on this experiment.

The dataset is saved as a Matlab® struct object ('data'), readable with Matlab®
or Octave (open source).
The 'data' object has a field corresponding to each stimulus
(eg. AG for auditory G, AVFus for audiovisual McGurk fusion stimuli)
Each of these fields contains the field 'counts' storing response counts in a
3d array (subject, snr, response) and a corresponding 'props' array representing
the same counts as response proportions.
The ordering of the SNR conditions is stored in the field 'snr' for each stimulus.
