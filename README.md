# Bayesian-Models-of-Audiovisual-Integration-in-Speech-Perception
special course project - DTU 

supervised by Tobias Andersen & Agata Wlaszczyk



based on Alma's work: https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0246986



worklog: https://docs.google.com/spreadsheets/d/12fZfqtgxMTg6rTjENT-Q2yRCTC252KaY4XWu0jumSAk/edit#gid=0



## Target problem

From Alma's work, we see both JPM and BCI model could fit the response data quite well, indicating both of them have the ability to estimate the internal logic of audio-visual perception. However, this conclusion might be limited due to the insufficient trails for each condition as well as the rough-classified responses. To explore whether there's a significant difference between these two models over audio-visual perception, we suggest to build a pilot experiment using stimulated data from cognitive models before conduct the complicated cognitive experiment, and address the degree of confidence about the models' consistency. We also hope the correctness of model would be explored through this experiment. 

There are mainly x phases for building this experiment. 

### Phase 1: Re-build the models



### Phase 2: Get the estimated parameters using testers' responses



### Phase 3: Visualize the difference of models with their distributions



### Phase 4: Generate samples from re-built models

Steps for this Phase:

1. We could observe from the visualization that when *snr* equals *"VHigh + ALow"* and with *asynchrony* condition. And for tester No.5,6,7,12, the differences of model distribution are much more obvious. Therefore, we would restrict the condition to asychnocy with VhighAlow. 
2.  Use the best parameters from the fitting process, generate 25\*3 samples for single stimuli (only take the high quality for visual stimuli and low quality for audio stimuli), and 25\*5\*2 samples for fusion stimuli for only Vhigh-Alow snr. The sample size here are set to match the original total size of the responses. 
3. Assign the samples into 9 categories, the boundaries are set by evenly dividing from $\mu_{ag}$ to $\mu__{ab}$. 
4. Store the counts and probabilities as the similar format as the original data responses.  

### Phase 5: Fit the stimulated sample to JPM and BCI respectively and compare the result 





## Things need to get done (updated on 6.11)

1. Two ways for examination. A) as what we did this week. B) test the roc (roc also could be used for testing other distributions) .   
2. Try with only neg log for fusion. 
3. Increase or decrease the sample size unit and see what is the threshold for distinguish the two models
