# Theme 1: Introduction

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

### [Reading] Section 6.5 of the [Deep Learning book](https://www.deeplearningbook.org/contents/mlp.html)

- **Backpropagation** is a méthod for computing gradient. It can be applied anytime gradients are needed, not only in DL. 

- In DL/ML, **gradient descent** is an algorithm that uses backprop in order to learn model parameters

- Backprop can compute the **derivatives** of any function as soon the derivatives are defined.

- <u>**Computational graphs**</u>

  - A **node** represents a **variable** which could be a scalar, a vector, a matrix, a tensor or any other type
  - An **operation** is a *simple* **function** of one or more variables. Only a set of allowable functions are available. An operation return a single output **variable**.
  - A **direct link** from node $x$ to node $y$ ( $x \rightarrow y $ ) means that the  **destination variable** obtained by applying an operation on the **source variable**

- <u>**Chain rule of calculus**</u>

  - Not to be confused with the **chain rule for probability** (e.g $P(A \cap B )=P(A|B)\times P(B)$)

  - It a method used to **compute derivatives** of functions **composed** of other functions whose derivatives are known

  - **general form of chain rule**:

    - Le $x \in \R^m$ an $y \in R^n$
    - Let be $y = g(x)$ and $z=f(y)$, we can use the chain rule to compute the gradient of $z$ w.r.t $x$.

    $$
    \frac{\partial z}{\partial x_i}=\sum_j \frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i} \\
    
    \gradient _x z = (\frac{\partial y}{\partial x})^{\intercal}\gradient _y z
    $$

    - To resume backprob is the multiplication of a Jacobian matrix by a gradient

- **<u>Recursively applying the chain rule to obtain Backprop</u>**

  - **Algebraically**, computing a gradient a complex function is easy using the chain rule, but it is **computionally complex**. There is generally a lot of computation to perform and some a are repeated. It should be decided if repeated expression should be **computed many times** (in order to save memory) or **computed once** but store the result (to speed up computation).

# Theme 2: Parameters sharing

- **Hyper-network**:  a network in which a the weights of some subnetwork are predicted by some other subnetworks.
- **Parameters sharing** is used in in **recurrent** and **convolutional** networks

## Recurrent nets 

- Essentially allow a network to have a memory
- A recurrent net is trained by **unfolding** the network then use backprop . For very long input the unfolded neural network is very long and cause **gradient vanishing** problems which when the gradient shrink layer after layer during backprob. Similarly there is the **exploding gradient** problem.
- Non linearity functions such as **sigmoid** limit the exploding gradient, but not the vanishing gradient. 
- **<u>RNN tricks</u>**:
  - Clipping gradients: **normalize** gradients so that they never exceed a certain **value**
  - Weights nitialization: starting in a good point avoids exploding/vanishing gradient.
  - LSTM self-loops: avoid vanishing gradient
- **<u>Gated Reccurent Unit</u>**
  - Can be seen as a simplification of LSTM
- **<u>Long Short-Term Memory</u>**
  - Reduce the problem of vanishing gradient
  - Hard to train in practice because it requires a **lot of data and computing power**. Today LSTM are mainly  replaced by **attention models**. Attention models are able to learn dependencies while being much easier to train.

## Convolutional nets

- Good for sequences, audio, speech, images, video and other natural signal
- CNN are good when **nearby** values in the signal are correlated 
- Based on **discrete convolution**
- Shift invariance
- **<u>Convolution</u>**
  - Definition (**Convolution**): $y_i=\sum_j w_jx_{i-j}$
  - In practice (Cross-correlation): $y_i=\sum_j w_jx_{i+j}$
  - In 2D: $y_{ij}=\sum_{kl} w_{kl}x_{i+k,j+l}$
- A **dense** convolution means a **stride** of $1$. 
- A **skip** or **dilated** convolution means some weights in the **kernel** is $0$
- **<u>Pooling</u>**
  - an **aggregation** operation
  - it is permutation **invariant**
    - Max Pooling
    - Average Pooling
    - Lp Pooling
    - Log Sum Exp Pooling: behave like max/min pooling or average pooling
- **Equivariant** to a transformation $T$: $f(T(x))=T(f(x))$, $T$ is another transformation
- **Invariant** to a transformation $T$: $f(T(x))=f(x)$
- Convolution is equivariant to **translation**
- Pooling is locally invariant to **translation**. This claim is not formally true, but translating the input before applying the pooling or applying the pooling then translate the output give similar results.

## Natural signal properties and the convolution

- **Stationarity** : the same patterns appear many time in natural signals. This is the reason why **parameters sharing** is used. It allows **faster convergence**, **better generalisation** and is not **constrained to input size**. It also reduces the amount of computation
- **Hierachical** composition: natural signals are made by composing simple patterns
- **Locality**: closed values are similar
- **Receptive field**: how many values influence the output of a neuron
- 1D data use 3D kernels collections: **(nb of kernels, number of channel of the input, length of each kernel)**
- **zero padding**: padding the input with $0$. This allows to reduce loss of information at the edges of the input

## Rucurrent neural networks

- **vec2sec**: take a vector as input and produces a sequence as output. This is used for example in **image captioning** where given an image, the network gives a textual description of the image.
- **sec2vec**: take a sequence as input and produces a vector. This can be used to map a textual input to as action. *e.g: Ok Google*

- **sec2vec2seq**: this is used in **machine translation**. The textual input is encoded to a vector (i.e embedding) which is then decoded by a decoder to produce the translation .
- **sec2sec**: used in **next word prediction**

# Theme 3: Enegy based models, foundations

## Energy based models

Feed-forward nets use a finite number of steps to produce a single output. But what if:

- the problem requires a complex computation to produce its output?
- there are multiple possible outputs for a single input ?
- what if there are some constraints that need to be satisfied by the output?



An **energy function $F(x,y)$** is a mesure of incompatibility between its inputs. It takes *low values* when the inputs are compatible and *high values* when they are not.

**Inference**: find the the value of $y$ that minimize $F(x,y)$. There may be multiple solutions $\hat{y}=argmin_yF(x,y)$

A feed-forward model is an **explicit function** that compute $y$ from $x$ while an EBM (Energy Based Model) is an **implicit function** that capture the dependancy between $x$ and $y$. 

If $y$  is continous, $F$ should be smooth and differentiable, so we can use gradient-based inference algorithms.

- **Conditional EBM: $F(x,y)$**.
- **Unconditional EBM: $F(y)$** . Measure the compatibility between the components of $y$. e.g: k-means, PCA, basically any unsupervised learning can be viewed as Unconditional EBM.

#### EBM Architectures

There are many architectures. Some popular ones are joint embedding and latent variables

##### Joint embedding architectures

<img src='ebm-joint-embedding.png' width=200/> 

- $Pred(x)$ and $Pred(y)$ are two neural networks. If they are the same, we have a **siamese** model. 
- The two networks produce an embedding which is considered the representation of the input and the energy function $C(h,h')$ computes the dissimilarity betwen the embeddings . 
-  Since there are multiple $y$ for the same input $x$, the network $Pred(y)$ becomes invariant to transformations in the $y$ domain.

##### Latent variable models



<img src='ebm-latent-variablee.png' width=200 />simultaneous minimization: $\hat{y},\hat{z}=argmin_{y,z}E(x,y,z)$

- the latent variable is $z$. 
- a **latent variable** is a variable that the value is unknown, but knowing the values will make the problem easier. 

Redefinition of $F(x,y)$:
$$
F_\infty(x,y)=\min_z E(x,y,z) \\
F_\beta (x,y) = -\frac{1}{\beta}\log \int_z \exp (-\beta E(x,y,z)) \\
\hat{y}=argmin_y F(x,y)
$$


