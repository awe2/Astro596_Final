# Astro596_Final

<h1>Spectral Auto-Encoder Photo-Z<h1>

<h4><p>This project will compare a new kind of Photometric Redshfits code: 'a spectral auto-encoder', to baseline algorithems: Random Forest, PCA+KNN, and Multi-Layer Perceptron Networks.</p>

<p>Photometric Redshifts are useful for creating large catalgoues of redshift information quickly without the need to measure spectra, which is more cost and time consuming. With the advent of next generation experiments with wide-deep views (Such as LSST), we will need to perform data analysis on large datasets at a quick pace. The fact is that we will soon be obserserving too many sources per night to measure spectra for each! However, photometric quantities such as images and their derived quantities are relatively cheaper to compute. The challenge is how to re-construct the information in spectra from just a few photometric quantities. What is more, providing accurate, low biased, and precise redshifts with uncertainty are key to future missions like Euclid.</p>

<p>There are generally two methods for photometric redshifts. The first is template fitting approaches, but in the high volume limit artificial neural networks, and more broadly, Machine Learning methods typically outperform. This project will implement a new kind of machine learning artificial neural network, called spectral auto-encoder and compare to baseline ML and DL methods to see if the new method gains upon a few key metrics: Dispersion, measured in Median Absolute Deviation; Bias, the average of residuals; Outlier Fraction, the fractional number of residuals greater than 3 times our MAD. When appropriate and in cases where we recover and estimate of the posterior, we implement PDF metrics Continous Ranked Probability Score and Probability Integral Transform Visual Metrics, to evaluate whether our Posteriors are 'well-behaved.' We will also Analyze our Individual metrics as they change with Redshift.</p>

<p>The New Method, a 'Spectral auto-encoder' is first trained using available spectra from our training set to go from spectra to magnitudes, this is the 'encoder' part. Then from these derived magnitudes, we will go to a single parameter output parameter, redshift. We will evaluate our networks ability to learn correlations from spectra and encode them onto magnitudes, to see if the network can transfer this learned information about spectral correlations in magnitudes onto redshifts. To do this, we will extract features from spectra, pass them through a dense layer of 5 neurons which will represent the magnitudes, then push the magnitudes through a Dense Network Tail to a single parameter Redshift and its uncertainty. Loss will be calculated on both the magnitudes and the redshift for training. Then, in validation, the magnitude to redshift portion of the network is fed magnitudes and Loss is calculated on the redshift distribution. In this scheme, we can evaluate the effectivenss of the model when in a real situation of not having spectra.</p>

<p>We will then compare this Spectral auto-encoder to baseline networks to evaluate whether providing direct spectral information to the network in training improved overall performance.</p></h4>

<p>Andrew Engel  - Spectral Encoder, Tutorial</p>
<p>Tsung-Han Yeh - Artificial Neural Network</p>
<p>Patrick Aleo  - Random Forest</p>
<p>Lina Florez   - Clustering</p>

<p>All</p>       - Presentation

