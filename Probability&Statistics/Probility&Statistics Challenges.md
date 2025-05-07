## Descriptive Statistics  
- **Measures of Central Tendency**  
  Write a function that takes any numeric list (or Pandas Series), returns mean, median, and mode, then plot a bar chart showing those three values on a synthetic dataset.  
- **Measures of Variability**  
  Generate a noisy sine wave, compute its rolling variance and rolling standard deviation (window = 20), and plot them alongside the original signal.  
- **Correlation & Covariance**  
  Download a small real-world dataset (e.g. Iris), compute the covariance and correlation matrices, then visualize both as heatmaps and interpret one strong positive and one weak/negative correlation.  

## Probability Theory  
- **Basic Probability Concepts**  
  Simulate flipping a biased coin (p=0.3 for heads) 10,000 times, estimate the empirical probability of heads and compare to the theoretical value.  
- **Conditional Probability & Bayes’ Theorem**  
  Given a dataset of medical test results (you can simulate with known sensitivities/specificities), write a script that computes P(disease|positive) via Bayes’ theorem and via Monte Carlo simulation to verify.  
- **Random Variables & Expectation**  
  Simulate rolling two dice 100,000 times, estimate the empirical PMF of the sum (2–12), compute its expected value, and compare to the known theoretical expectation (7).  

## Probability Distributions  
- **Bernoulli Distribution**  
  Write your own Bernoulli sampler function (no libraries), generate 1,000 samples, and plot the histogram to verify P(1)=p.  
- **Binomial Distribution**  
  Without using `scipy.stats`, implement the binomial PMF and CDF functions, then compare your results to `scipy.stats.binom` for n=10, p=0.4.  
- **Poisson Distribution**  
  Simulate a Poisson process by interarrival-times (exponential) to generate counts over fixed intervals; verify that the distribution of counts matches the Poisson PMF.  
- **Uniform Distribution**  
  Create a function that draws N samples from a continuous [a,b] uniform distribution, plot the empirical PDF, and overlay the theoretical line.  
- **Normal (Gaussian) Distribution**  
  Use the Box–Muller transform to generate standard normal samples, then scale/shift them to arbitrary μ,σ; verify by plotting a histogram with the true Gaussian curve.  
- **Exponential Distribution**  
  Given a stream of timestamped events, write code to estimate the rate λ via MLE (i.e., λ̂ = 1/mean interarrival time) and overlay the fitted exponential PDF on the empirical histogram.  

## Inferential Statistics  
- **Central Limit Theorem**  
  Sample means from different distributions (e.g. uniform, exponential) in batches of size n=5, 30, 100; for each n plot the sampling distribution histogram and show how it “Gaussian-izes” as n grows.  
- **Confidence Intervals**  
  Given a real or simulated dataset, write a function that computes a 95% confidence interval for the mean (both with known σ and unknown σ using the t-distribution), and empirically check coverage by repeating the experiment many times.  
- **Hypothesis Testing & p-Values**  
  Simulate two groups (e.g. heights of men vs. women with slightly different means), perform a two-sample t-test, report the p-value, and interpret whether you “reject” at α=0.05.  

## Bayesian Methods  
- **Bayesian Inference & Bayes’ Theorem**  
  Implement a Beta-Binomial model: choose a Beta(a,b) prior, observe k successes in n trials, compute the posterior Beta(a+k, b+n−k), and plot prior vs. posterior densities.  
- **Prior vs. Posterior Distributions**  
  Take real A/B-test click-through data (or simulate), pick two different priors (e.g. Beta(1,1) vs. Beta(5,5)), compute the posteriors, and visualize how choice of prior affects your belief.  
- **Markov Chain Monte Carlo (MCMC)**  
  Write a simple Metropolis sampler to draw samples from a 2D Gaussian mixture, visualize the chain trajectory and approximate density contour.  

## Statistical Modeling  
- **Linear Regression**  
  From scratch, implement OLS for multivariate data (solve normal equations), fit on Boston Housing (or synthetic) dataset, and compare your coefficients to scikit-learn’s.  
- **Logistic Regression**  
  Code a binary logistic regression trained via gradient descent, apply it to a binary classification dataset (e.g. breast cancer), plot the decision boundary, and report accuracy/ROC-AUC.  
- **Maximum Likelihood Estimation (MLE)**  
  Given data from an exponential distribution, write an MLE routine to estimate λ by optimizing the log-likelihood, and compare to the analytical estimate 1/mean.  
- **Model Evaluation & Cross-Validation**  
  Using scikit-learn’s cross_val_score or your own splitter, compare performance of k-NN, logistic regression, and decision tree on a dataset, and plot their cross-validated scores.  
- **Regularization (L1 & L2)**  
  Train ridge and lasso regression on a high-dimensional (p > n) synthetic dataset, plot how coefficients shrink as you vary the regularization strength, and observe the sparsity for L1.  

## Information Theory  
- **Entropy**  
  Write a function that computes the entropy of a discrete distribution, then compute and compare the entropy of English letter frequencies vs. a uniform distribution over 26 letters.  
- **Cross-Entropy**  
  Given two probability vectors (true vs. predicted), implement cross-entropy loss, and apply it across a mini neural-net forward pass (e.g. softmax+loss) using NumPy.  
- **KL Divergence**  
  Compute the KL divergence between two fitted Gaussians (estimate means/vars from data), and verify its non-symmetry by computing DKL(p||q) vs. DKL(q||p).  
- **Mutual Information**  
  Using a small categorical dataset (e.g. weather vs. play decisions), compute empirical mutual information and compare to `sklearn.feature_selection.mutual_info_classif`.  

## Markov Chains  
- **Markov Chains Basics**  
  Create a simple weather simulator with states {Sunny, Rainy}, define a transition matrix, simulate 1,000 days, and estimate the stationary distribution by counting frequencies.  
- **Hidden Markov Models (HMMs)**  
  Implement the Viterbi algorithm for a toy HMM (e.g. coin toss with a hidden bias state), feed it an observed sequence, and decode the most likely hidden state path.  

## Statistical Learning Theory  
- **Bias–Variance Tradeoff**  
  On a synthetic polynomial regression problem, fit models of increasing degree (1 to 10), compute train vs. test MSE, and plot the classic U-shaped bias-variance curve.  
- **VC Dimension & Capacity**  
  Write a function that tests for “shattering” for small point sets and a hypothesis class (e.g. 1D intervals), estimate the VC dimension experimentally, and compare to the theoretical value.  
- **PAC Learning**  
  Simulate a simple PAC scenario: choose a target concept (e.g. a threshold function), draw random labeled samples, run a learning algorithm, and empirically show that error drops below ε with high probability as sample size increases.  
- **Generalization & Overfitting**  
  On a neural network (e.g. Keras toy model), experiment with different dropout rates and training set sizes, plot train/test accuracy curves, and observe how dropout helps generalization.  

---

Happy coding! Tackle these one by one (and commit often) to turn theory into real-world skills—plus you’ll have a killer GitHub showcase by the end.