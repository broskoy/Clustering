## Algorithm Description

**Initial Approximation and Clustering**  
The algorithm begins by running a constant factor approximation for k-means clustering to find a preliminary set of k centers, denoted as C={c1​,…,ck​}. Once these initial centers are established, the entire point set is partitioned into k clusters, C1​,…,Ck​, by assigning every point in the dataset to its nearest geometric center in C.


**Radius Calculation and Spatial Partitioning**  
For each established cluster Ci​, the algorithm calculates a specific boundary radius ri​ to geometrically separate the points into two groups. This radius is defined by the formula ri​=ϵ⋅∣Ci​∣cost(Ci​,ci​)​​, where cost(Ci​,ci​) is the sum of squared distances of all points in the cluster to the center, ∣Ci​∣ is the total number of points in the cluster, and ϵ is the error parameter. The cluster is then divided: Ciin​ contains all points located within this radius ri​ from ci​, while Ciout​ contains the remaining points located strictly outside of this boundary.

**Uniform Sampling for Inner Points**  
From the inner subset Ciin​, the algorithm draws a predetermined number of samples, siin​, independently and uniformly at random. Because these points are sampled uniformly, each selected point is assigned an identical weight calculated as siin​∣Ciin​∣​. This weighting ensures that the small sample accurately represents the mass of the unsampled points in its immediate vicinity. This resulting weighted sample set is denoted as Siin​.

**Non-Uniform Sensitivity Sampling for Outer Points**  
The algorithm treats the outer points in Ciout​ differently by drawing a sample set Siout​ of size siout​ using a non-uniform probability distribution. The probability Pq​ of selecting any specific point q from Ciout​ is strictly proportional to its squared distance to the center, defined mathematically as Pq​=cost(Ciout​,ci​)Δ(q,ci​)​. The assigned weight for each sampled point is inversely related to this probability, calculated as wq​=siout​⋅Δ(q,ci​)cost(Ciout​,ci​)​. This mechanism guarantees that distant points are sampled with higher probability, while their corresponding weights are scaled down to maintain a mathematically unbiased estimator of the total clustering cost.

**Final Coreset Construction**  
The overall weak coreset, denoted as S, is constructed by aggregating all the inner and outer sampled sets across all k clusters. This is expressed mathematically as S=⋃i=1k​(Siin​∪Siout​). The final set S contains only a small fraction of the original dataset but retains the necessary weighted geometric structure. The computationally demanding standard k-means algorithm can then be executed exclusively on this small, weighted coreset to determine the final cluster centers.

<br><br>

## Parameter Optimization

**Parameter Configuration and Optimization Strategy**  
Translating the theoretical bounds of the weak coreset algorithm into practical Python code requires a specific strategy for handling the input parameters. The theoretical formulas are designed to guarantee success in the absolute worst-case scenarios, meaning strict parameter choices can lead to impractically large sample sizes.

**Determining the Compression Level (k)**  
The parameter k represents the target number of clusters, which directly corresponds to the number of unique colors in the final compressed image. This parameter dictates the compression ratio. The experimental plan involves running the pipeline across varying color depths, specifically testing k=8,16,32, and 64. The resulting visual quality (PSNR) and execution time will be plotted against these k values to evaluate the algorithm's scalability.

**Managing the Error Bound (ϵ) and Sample Size**  
The error parameter ϵ defines the acceptable deviation between the coreset's clustering cost and the mathematically optimal clustering cost. The paper dictates that the required sample sizes for the inner and outer sets must satisfy the following bound:

$$s \ge c \cdot \frac{k \ln(k/\delta)}{\epsilon^5} \ln(k/\epsilon \cdot \ln(1/\delta))$$

Because ϵ is raised to the 5th power in the denominator, choosing a highly strict theoretical error bound (e.g., ϵ=0.1) causes the required sample size to explode, potentially exceeding the total number of pixels in the image. To optimize the implementation, the experimental strategy will not strictly enforce this formula. Instead, the sample sizes will be manually constrained to small, fixed amounts (e.g., 1,000 to 10,000 points). The empirical error will then be measured against the standard k-means++ baseline to demonstrate that practical accuracy significantly outperforms the conservative theoretical bounds.

**Probability Amplification via the Confidence Parameter (δ)**  
The confidence parameter δ defines the acceptable probability that the randomized non-uniform sampling fails to produce a valid coreset. A lower δ (higher required confidence) mathematically forces a larger sample size.

To keep the coreset small without sacrificing reliability, the implementation will utilize probability amplification. The parameter δ will be set to a relatively high value (e.g., δ=0.5, which represents a 50% theoretical failure rate). Because clustering on the resulting tiny coreset is computationally inexpensive, the algorithm will generate multiple independent coresets and run k-means on each. If 5 iterations are run, the probability that all 5 fail drops to 0.55, yielding an actual success probability of 96.875%. The iteration that produces the lowest final clustering cost will be selected as the final compressed output.

<br><br>

## Python Implementation

Clustering/  
│  
├── originals/              # original png images  
├── input/                  # smaller png images  
├── output/                 # compressed images  
├── data/                   # the test data saved in csv
├── plots/                  # different plots for the essay  
│  
├── src/                    # source code  
│   ├── deconstruct.py      # image to array  
│   ├── reconstruct.py      # array to image  
│   ├── coreset.py          # calculates the coreset  
│   ├── compress.py         # compresses one image  
│   └── run.py              # run all of the tests  
│  
├── documentation.md        # detailed explanations here  
├── requirements.txt        # list of dependencies  
└── README.md               # project overview 


<br><br>

## Discussion points

**The Image as a Diagnostic Medium**  
While this implementation processes pixels as data points, the primary objective is not to compete with dedicated image codecs. Instead, the image serves as a multi-dimensional dataset to validate the coreset extraction algorithm visually. By mapping the abstract mathematical output of a clustering algorithm back into a 2D spatial grid, the image provides an observable proof of the algorithm's success in preserving the underlying data. This approach directly translates to modern big data problems, where coresets are deployed to process massive datasets in machine learning, sensor networks, and streaming analysis where the size would exceed available memory.

**Limitations of K-Means in RGB Space**  
The application of k-means clustering to color quantization reveals a fundamental geometric limitation. The k-means objective function strictly minimizes squared Euclidean distances, operating under the assumption that the underlying data forms distinct, spherical clusters. Natural images frequently contain smooth color gradients and continuous transitions, which map as elongated streaks or irregular shapes in a 3D RGB coordinate space. Because the algorithm forces spherical boundaries onto non-spherical data, it occasionally splits continuous gradients arbitrarily, leading to visual color banding.

**NP-Hardness and Algorithmic Trade-offs**  
Finding the mathematically optimal k-means clustering is an NP-hard problem. This algorithm bypasses that limitation by computing a constant-factor approximation to establish initial centers, followed by a probability-based data reduction. While mathematically rigorous, this approach is computationally demanding due to the required distance calculations across all data points. Deterministic spatial partitioning algorithms, such as Octree or Median Cut, perform the same color reduction in a fraction of the time by dividing the color space geometrically. The trade-off is clear: deterministic spatial splits prioritize execution speed and practical hardware efficiency, whereas the coreset methodology prioritizes strict, mathematically proven error bounds on the sum of squared distances.

**Theoretical Bounds versus Empirical Reality**  
A critical finding of this project is the severe divergence between theoretical algorithmic bounds and empirical performance. The coreset algorithm provides a strict mathematical guarantee: the final clustering cost will fall within a $1 \pm \epsilon$ factor of the optimal solution. However, because this theorem must hold true for the absolute worst-case data distribution across any dimension, the resulting formula demands massive sample sizes for small error bounds. In practice, natural image data is highly structured. The empirical results demonstrate that executing the algorithm with drastically smaller sample sizes still achieves high accuracy, highlighting the conservative nature of worst-case theoretical computer science.



## Structure of thesis

**Introduction**
> explain what we are doing here

**Algorithm Used**
> explain k means++
> explain the paper implementation
> talk about the process steps
> focus o the coresets a bit

**Visual Medium**
> image compression is a useful visualization but not the goal
> go into a bit of detail on converting the images
> downsides of having spherical clusters in images

**Problem Complexity**
> np-hardness
> talk about the numbers and intensiveness
> do some formulas
> despite this clustering must be done somehow (find relevat usecases)

**Present Running Times**
> start comparing the coreset method with simple k-means
> present relevant plots

**Improvements to our method**

