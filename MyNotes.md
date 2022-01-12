# Introduction

## Gradient descent and the backpropagation algorithm

- Parameterize model: the input is $x$ and the parameter is $w$
  $$
  \bar{y} = G(x, w)
  $$

  - Exemple: Linear regression
    $$
    \bar{y} = \sum_iw_ix_i \; C(y,\bar{y}) = ||y-\bar{y}||^2
    $$
    $C$ is the cost function and is used the compute the discrepancy or the divergence or the distance between the computed output $\bar{y}$ and the desired output which is always known in supervised learning 

  - Exemple: Nearest neighbor
    $$
    \bar{y} = argmin_k ||x-w_k||^2
    $$

  - <span style='color:red'>The $G$ function could be a more complicated function than linear regression or nearest neighbor</span>.

- Grandient Descent (GD)
  - The full batch gradient descent computes the average gradient of the entire dataset before updating the parameters
  - The stochastic gradient descent (SGD) computes the gradient of a random sample and update the weights. It is less smooth than the full batch GD because of *noise* (or randomness) but it is generally **faster**. This is because only one gradient is computed at a time ant there is generally a lot of redundency in the dataset, making the *same* gradient to be computed multiple times in the full batch gradient descent. SDG is also less prone to **local minima problem**, thanks to randomness
  - In practice, the mini-batch gradient descent is used for parallelization
  - <span style='color:red'>the cost function must be continuous, and differentiable almost everywhere</span>

- Chain rule:
  $$
  g(h(s))' = g'(h(s))\times h'(s)\\
  \frac{dc}{ds} = \frac{dc}{dz}\times\frac{dz}{ds}\\
  \frac{dc}{ds}=\frac{dc}{dz}\times h'(s)
  $$



- Using chain rule for vector functions

$$
z_g: [g_g \times 1] \; Z_f: [d_f \times 1] \\
\frac{\partial c}{\partial z_f} = \frac{\partial c}{\partial z_g}*\frac{\partial z_g}{\partial z_f} \\
[1 \times d_f] = [1 \times d_g]*[d_g \times d_f]
$$

​		$*$: stands for vector-matrix or matrix-matrix multiplication

- Jocobian matrix

  - Partial derivative of i-th output w.r.t j-th input

  $$
  (\frac{\partial z_g}{\partial z_f})_{ij}=\frac{(\partial z_g)_i}{(\partial z_f)_j}
  $$

  

## Neural nets inference

- Linear transformations are: rotation, scaling, reflection, translation, shearing
- Data are generally more difficult to seperate in high dimensional space  because their statistics are very similar. 
- A neural network does two thing:
  1. Rotation by applying a linear transformation using a matrix (i.e the weights).
  2. Squashing by applying the non-linear transformation (e.g ReLu, tanh, etc).

## Modules and architectures

### Activation function

- ReLu - nn.ReLU(): 
  - $ReLU(x)=(x)^+=max(0,x)$
  - Also called *positive part* by mathematicians. 
  - Does not care about the ampliture of the input.
  - Variants: 
    - nn.RReLU: $RReLU(x)=x \;if \; x \ge 0 \; otherwise \;ax$, where $a$ is randomly sample from a uniform distribution
    - nn.PReLU: $PReLU(x) = max(0,x)+a\times min(0,x)$,  same as RReLU, but $a$ is a learnable parameter
    - nn.Softplus: is a smooth approximation of the ReLU: 
      - $Softplus(x)=\frac{1}{\beta}\log(1 + \exp(\beta x))$
      - the higher is $\beta$, the more close is the function to ReLU
    - Etc
- Sigmoid - nn.Sigmoid()
  - $Sigmoid(x)=\sigma (x) = \frac{1}{1+expo(-x)}$
  - a particular case of $softmax()$ where there are two values only and one is always $0$
- Tanh - nn.Tanh()
  - $Tanh(x)=tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}$
  - very similar to Sigmoid, with the adventage to be symetric 
  - a limitation is the gradient for values that are in the flat part will be zero
- Softmax - nn.Softmax()

## Neural nets training

### ANN - supervised learning: classification	
- LogSoftmax - nn.LogSoftmax()
  - applies log to softmax
  - most people use this for classification
  - less prone to vanishing gradient compared to softmax
  - a particular case of cross-entropy
  - more stable to numerical issues than softmax

### Cost functions

- MSE - nn.MSELoss()

- NLL - nn.NLLLoss(): 

  - Negative log likelihood loss

- CE-nn.CrossEntropyLoss(): 

  - combines nn.LogSoftmax() and nn.NLLLoss() in a sisngle class

  - $$
    loss(x,class)=-log(\frac{\exp(x[class])}{\sum_j \exp(x[j])})=-x[class]+log(\sum_j \exp(x[j]))
    $$

    <span style='color:red'>The objective is to maximize the output for the correct class (i.e $x[class]$) and to minimize every output of the incorrect classes **(including the correct class actually)**. Therefore $-x[class]$ is minimized and $log(\sum_j \exp(x[j]))$ is also minimized.</span>

- nn.AdaptiveLogSoftmaxWithLoss(): 

  - efficient softmax approximation for large ouput spaces

- Ranking - nn.MarginRankingLoss(): 

  - try to ensure that the output of the correct class is greater than the output of the second higher output

  - $$
    loss(x_1,x_2,y)=\max(0,-y\times(x_1-x_2)+margin)
    $$

    $x_1$ is the output of the correct class, if $y=1$ then $x_1$ should be ranked first and if $y=-1$ then $x_2$ should be ranked first.

## Homework 1

### Regression task

a) the $5$ to train a model with PyTorch: forward, loss computation, zero grad, compute grad, make step

b) <span style='color:red'>not sure, to be verified</span>

| Layer    | Input | Output                     |
| -------- | ----- | -------------------------- |
| Linear 1 | $x$   | $z_1 = W^{(1)}x+b^{(1)}$   |
| f        | $z_1$ | $z_2 = ReLu(z_1)$          |
| Linear 2 | $z_2$ | $z_3=W^{(2)}z_2 + b^{(2)}$ |
| g        | $z_3$ | $\hat{y}=z_3$              |



c) <span style='color:red'>not sure, to be verified</span>

| Parameter | Gradient                          |
| --------- | --------------------------------- |
| $W^{(1)}$ | $W^{(2)}x$ if $z_1\ge 0$ else $0$ |
| $b^{(1)}$ | $W^{(2)}$ if $z_1\ge 0$ else $0$  |
| $W^{(2)}$ | $z_2$                             |
| $b^{(2)}$ | $1$                               |

