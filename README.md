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
3. Assign the samples into 9 categories, the boundaries are set by evenly dividing from $\mu_{ag}$ to $\mu_{ab}$. 
4. Store the counts and probabilities as the similar format as the original data responses.  

### Phase 5: Fit the stimulated sample to JPM and BCI respectively and compare the result 

#### 5.1 Model re-fitting

##### 5.1.1 re-fitting procedure

##### 5.1.2 re-fitting results



Fitted models with tester No.12; samples are generated from BCI model.

![](./results/plots/fitted_vis_12_bci_1.png)

![](./results/plots/fitted_vis_12_bci_2.png)



#### 5.2 Result Analysis

##### 5.2.1 The definition of $s_M$  (the score of a model)

The score for evaluating the model is the value of negative log likelihood. Here we compute two types of negative log likelihood, 1) only for fusion condition, 2) the sum of all conditions. 
$$
s_{M_{fusion}} = -log\mathcal{L_{fusion}} \\
s_M = -\sum_{c\in{C}}log\mathcal{L_c}
$$
where $c$ represents the conditions, $C$ represents the collection of all conditions. 



According to the definition of likelihood, the larger the likelihood is, the better the model is. In our case, with the negative log likelihood, the smaller the $s_M$ is, the better the model is. 



##### 5.2.2 Comparison over the difference of scores from 2 models

The first method we use to comparison the results over 2 models is computing the difference between the score from 2 models. 

We define the new comparison factor $diff$ as below:
$$
diff = \frac{s_{JPM}-s_{BCI}}{s_{BCI}}
$$
Considering there's two kinds of $s_M$ and they have different numerical range, we divide the difference by the score of BCI model to normalize all value into the range of $[-1,1]$.  Based on the definition of $s_M$, we'll know that when $diff<0$, meaning JPM fits the sample better; otherwise, BCI model fits the sample better. 

The plot below shows the distribution of $diff$ for both BCI and JPM samples for tester No.12 with the score type of $s_{M_{fusion}}$. As clearly showed in the plot, for most of JPM samples, JPM fits the sample better; and the same story happen to BCI samples as well. Still, there's a small area of overlapping showing in the plots, indicating that for some samples generated from model A, it turned out that model B got the better fitting result. We'll analyze the certainty of one model 'wins' the other in the following section 5.2.4. 

![diff_index0_12](./results/plots/diff_index0_12.png)

The following 2 plots shows the $diff$ results for all 16 testers, using two types of score ($s_M$, $s_{M_{fusion}}$). 

Some key observations:

1. $s_M$ separates the 2 parts of distributions better than $s_{M_{fusion}} , which could be observed from section **5.2.4** as well
2. In general, for BCI samples,  BCI model fitting outperforms JPM fitting; for JPM samples, JPM model fitting outperformed BCI model fitting. What worth mentioning is that , the degree of outperforming is much more significant for BCI sampling, which might indicates the BCI model might be closer to the human audio-visual perception. 
3. There are some testers (8,13,15), the $diff$ values are very similar for both models. What need to notice is that for these testers we use different x axis scale(1e-5). 

![](./results/plots/diff_index1_all_rangead.png)



![](./results/plots/diff_index0_all_rangead.png)

##### 5.2.3 ROC curve analysis

Another way of analyzing the refitting result is simply checking the ROC curve. From signal detection theory, the ROC curve is used for measure the sensitivity between two signals (or one signal and one noise). Just as the plot shows below, when curve is more close to the diagonal, two signals are more similar to each other. 

![roc_illustration](./img/roc_illustration.JPG)

The plot below shows the ROC curve for 16 tester.

![](./results/plots/roc_curve.png)

We could observe from this plot that for most of the tester, the 2 fitted model are quite similar with each other, just like the plot we show before {**plots needed(the fitted distribution for every tester)**}. What worth mentioning is the tester no.5,6,7 and 12 are quite different from the others. They show a 'peak' with low $P_{FA}$ area while being close to the diagonal with large $P_{FA}$ area.

ROC curve in Gaussian coordinates could not only help us to measure the sensitivity (d') at one glimpse (with even covariance, d' is the intercept of the fitted line), but also reveal whether the signals are both gaussian distributed. If both signals are gaussian distributed, then the nodes should be linear distributed. 

Below the plot shows the ROC in Gaussian coordinates for 16 testers. It could be seen that tester No.2,3,5,6,7,10,12 are clearly having non-Gaussian distribution for BCI model. 



![](./results/plots/roc_curve_gau.png) 

##### 5.2.4 *p*-value and significance test

The last method we used for analyzing the difference between 2 fitted model is significance test. The idea is  to exam how confident we are with the statement about Model A 'wins' Model B. The null hypothesis are: 

1. for JPM samples, JPM model works less better than BCI model 
2. (also the other way around) for BCI samples, BCI model works less better than JPM model.

Notice that the null hypothesis is the statement that we'd like to prove against. 

We are going to generate one set of samples for each tester from both fitted distributions, then using both model to fit the stimulated data again. After this the $s_M$ for both model would be calculated, then we will use $diff$ for testing is it significantly larger/less than 0. 

The above experiment would be done for 100 times. By collecting all the p-value from 100 experiments, the hypothesis could be well tested.

> $H_{a_{0}}$: For samples generated from JPM, $s_{JPM} > s_{BCI}$ , which equals to $s_{JPM} - s_{BCI} > 0$
>
> $H_{b_0}$: For samples generated from BCI, $s_{JPM} < s_{BCI}$ , which equals to $s_{JPM} - s_{BCI}<0$

Notice that both $H_0$ is one-tailed. 

as for each experiment, sample_size < 30, so considering using t-test.



* $s_M$, alpha = 0.01
  * H_a_0, p-value less then 0.01:92
  * H_b_0, p-value less then 0.01:100

* $s_M$,alpha = 0.05
  * H_a_0, p-value less then 0.05:95
  * H_b_0, p-value less then 0.05:100

* $s_{M_{fusion}}$,alpha=0.01
  * H_a_0, p-value less then 0.01:0
  * H_b_0, p-value less then 0.01:98

* $s_{M_{fusion}}$,alpha=0.05
  * H_a_0, p-value less then 0.05:12
  * H_b_0, p-value less then 0.05:100

* $s_{M_{fusion}}$,alpha=0.1
  * H_a_0, p-value less then 0.05:31
  * H_b_0, p-value less then 0.05:100







## Things need to get done (updated on 6.11)

1. Two ways for examination. A) as what we did this week. B) test the roc (roc also could be used for testing other distributions) .   
   1. the examination on fitting performance
      * exam on 16(?) tester {at least on 5,6,7,12, maybe also 0,1,2,3,10}
      * do it for 100 experiment
      * 
   2. ROC
      * not like the A method, we don't need to fit the model again (from what I understand). Instead, we only sampled from the distribution, and then compute the proportions and take the inversed value for Gaussian coordinate 
      * This is for examining whether the distribution is Gaussian or not
      * question: what is the noise
      * *updated: error bar
      * *updated: 100 trails with error bar with all 16 testers
2. Try with only neg log for fusion. (done)
3. Increase or decrease the sample size unit and see what is the threshold for distinguish the two models 

(updated on 6.15)

4. statistics on 16 testers 

   for each tester, run one experiment from BCI and JPM samples, and compute the p-value for the hypothesis that one model wins.

   do this for 100 trails.

{plan for 6.22

* to-do-list 4. statistics, about how to define hypothesis and how to get the p-value 

* ROC 100 trails + error bar

* write the report Phase 1& 2, overleaf}

   

### Statistics 
