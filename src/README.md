



- 图像滤波器加速
  -  GaussianKernel
  -  MedianKernel
  -  LaplacianKernel
- 神经网络加速
  - 线性回归
  - 梯度下降
  - 多层感知机
  - 卷积神经网络
  - 循环神经网络
  - 自动编码器
  - 生成对抗网络
- 流体模拟加速：实现基于CUDA的流体模拟算法，比如基于格子的流体模拟（如Lattice Boltzmann方法），并利用GPU加速模拟过程。
- 并行排序算法：实现基于CUDA的并行排序算法，比如快速排序、归并排序的CUDA版本，并比较其与传统CPU排序算法的性能。
- 加密算法加速：实现常见的加密算法（如AES、DES等）的CUDA版本，利用GPU加速加密和解密过程。
- 科学计算应用：选择某个科学计算领域（如分子动力学、地球物理学、量子化学等），实现相应的算法并利用CUDA加速计算过程。


## Modern Data Mining Algorithms in C++ and CUDA C

- Hidden Markov models are chosen and optimized according
to their multivariate correlation with a target. The idea is that
observed variables are used to deduce the current state of a
hidden Markov model, and then this state information is used to
estimate the value of an unobservable target variable. This use of
memory in a time series discourages whipsawing of decisions and
enhances information usage.
- Forward Selection Component Analysis uses forward and
optional backward refinement of maximum-variance-capture
components from a subset of a large group of variables.
This hybrid combination of principal components analysis with
stepwise selection lets us whittle down enormous feature sets,
retaining only those variables that are most important.
- Local Feature Selection identifies predictors that are optimal
in localized areas of the feature space but may not be globally
optimal. Such predictors can be effectively used by nonlinear
models but are neglected by many other feature selection
algorithms that require global predictive power. Thus, this
algorithm can detect vital features that are missed by other feature
selection algorithms.
- Stepwise selection of predictive features is enhanced in three
important ways. First, instead of keeping a single optimal subset
of candidates at each step, this algorithm keeps a large collection
of high-quality subsets and performs a more exhaustive search of
combinations of predictors that have joint but not individual power.
Second, cross-validation is used to select features, rather than using
the traditional in-sample performance. This provides an excellent
means of complexity control, resulting in greatly improved outof-sample performance. Third, a Monte-Carlo permutation test is
applied at each addition step, assessing the probability that a goodlooking feature set may not be good at all, but rather just lucky in its
attainment of a lofty performance criterion.
- Nominal-to-ordinal conversion lets us take a potentially
valuable nominal variable (a category or class membership) that
is unsuitable for input to a prediction model and assign to each
category a sensible numeric value that can be used as a model
input.