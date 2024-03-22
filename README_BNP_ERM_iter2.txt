DESCRIPTION OF CODE IN "BNP_RO_ICML24" SCRIPT


The code is divided into four clearly-labeled sections.

The first section ("Functions") defines the functions necessary to perform the stick-breaking/multinomial-dirichlet approximation procedures, sampling atoms from the DP posterior predictive, evaluating the loss function and its gradient, perform the SGD algorithm, and compute out-of-sample performance.

The second section ("Linear Regression Simulation Experiment") reports the code necessary to reproduce the results of the high-dimensional linear regression study as plotted in Figure 2 in the main text of the article.

The third section ("Gaussian Mean Estimation Simulation Experiment") reports the code necessary to reproduce the results of the robust Gaussian location parameter estimation study as plotted in Figure 3 in Appendix C of the Supplement to the article.

The fourth section ("Logistic Regression Simulation Experiment") reports the code necessary to reproduce the results of the high-dimensional logistic regression study as plotted in Figure 4 in Appendix C of the Supplement to the article.

We recommend first running the code in the header (to import the necessary utilities) and in the first section. Then, the code chunks in the subsequent sections can be run independently of each other.
