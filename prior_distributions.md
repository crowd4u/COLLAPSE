# Prior Distributions for Bayesian models

Aggregation algorithms based on approximate Bayesian methods require specifying hyperparameters for their prior distributions. This file provides supplementary information on the prior distributions used in this paper and the effect of changes in these priors on the experimental results.

In particular, this file explains the prior distributions for BDS and HS-DS.
Note that for CBCC, we used the prior distributions as specified in the original implementation from the prior study.

## Prior Distributions for BDS and HS-DS

In this file, we adopt the notation from [(Liu & Wang 2012)](https://dl.acm.org/doi/10.5555/3042573.3042579), and let $n$ to the number of the classes.

In BDS (HybridConfusion), we have to set two hyperparametor (prior distiributions) $\alpha$ and $\Lambda$.

### Class prior
$\alpha$ is the prior distribution for the class distribution $\rho$, and is a vector of length $n$. 
Liu \& Wang set this parameter as follows:

$$\alpha = \[1,1,...,1\]$$

This implies the assumption of a uniform class distribution. 
Since the true class distribution is generally unknown in label aggregation problems, this setting is considered a universal and standard approach. 
We adopt this value for both BDS and HS-DS in this paper.

### Confusion matrix prior
On the other hand, $\Lambda$ represents the prior distribution for the confusion matrices. 
Lin & Wang point out that the optimal setting is dataset-dependent and use several parameters in their experiments. 
We introduce a variable $\tau$ and set $\Lambda$ as follows:

$$
\Lambda = \left[
    \begin{matrix}
        \frac{\tau (n-1)}{1-\tau} & 1 & \dots & 1 \\
        1 & \frac{\tau (n-1)}{1-\tau} & \dots & 1\\
        \vdots & \vdots & \ddots & \vdots\\
        1 & 1 & \dots & \frac{\tau (n-1)}{1-\tau}
    \end{matrix}
\right]_{n\times n}
$$

This implies the assumption of a worker with an average accuracy of $\tau$.

Liu \& Wang define $\Lambda$ using a parameter $\lambda$; in our formulation, this is equivalent to $\lambda=\frac{n\tau -1}{1-\tau}$.

In our paper, we used $\tau=0.75$. This corresponds to $\lambda=3n-4$ (note that inference can be run with different values of $\tau$ by changing the argument `init_worker_accuracy` in the implementation available in the repository).

### The effect of changes in $\tau$.
While it is expected that changes in the value of $\tau$ will have some influence on the experimental results, it was not feasible to run numerous patterns, considering the time required for the experiments (our complete set of experiments took over two weeks to finish). 

Therefore, we provide a case study of the effect of varying $\tau$ in a specific case as supplementary information. 

The following shows the results for BDS and HS-DS on the Dog dataset, with $r=5$ and $a_{AI}=\mu$, for different values of $\tau$.


<img width="743" height="321" alt="homo_acc" src="https://github.com/user-attachments/assets/d6d615bf-3284-467f-adab-d02f0c6a4e98" />
<img width="743" height="321" alt="homo_recall" src="https://github.com/user-attachments/assets/71745963-05d8-4690-a859-b306abfd3863" />
<img width="743" height="321" alt="hetero_acc" src="https://github.com/user-attachments/assets/a000eff3-68ae-40fd-83ac-621f266365f8" />
<img width="743" height="321" alt="hetero_recall" src="https://github.com/user-attachments/assets/b741b25c-9a11-4088-897c-ed625d7f8e05" />

The results show that the experimental outcomes follow a similar trend regardless of the value of $\tau$ (although an extremely high value, such as $\tau=0.95$, causes a decrease in performance in the homo scenario).

This result can be reproduced by changing `init_worker_accuracy` and running the part of the main experiment.
