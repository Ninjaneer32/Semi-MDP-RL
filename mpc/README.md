# Model Predictive Control Approach to Lifter Control
This is a documentation which comprehensively provides the details of the baseline MPC controller. In short, the MPC controller solves an optimization problem in every decision epoch to obtain an action, where the optimization problem aims to find a sequence of actions that obtains the best result for the finite length of the future. The problem described below is a **mixed integer linear programming (MILP)** problem where each variable represents the state of the environment or encodes an action to be executed.

---
## 0. Notation
To describe our MPC problem formally, we first introduce the following notations:

- $\mathcal{P}$ denotes the set of all possible positions taken by the lifter,
- $\mathfrak{F}$ denotes the set of floors in the fab, which is mathematically a partition of $\mathcal{P}$. In other words, each $f \in \mathfrak{F}$ corresponds to a subset of $\mathcal{P}$ such that every position $p \in \mathcal{P}$ belongs to the unique floor instance $f \in \mathfrak{F}$.

We also recall that $\mathcal{C}$ is used to denote the set of all conveyors, and $\mathcal{A}$ is the action space of our semi-MDP problem. Furthermore, we use the letter $e$ to represent the vector with all entries equal to 1. Finally, For any vectors $X, Y \in \mathbb{R}^n$, $X \circ Y$ denotes the elementwise multiplication of $X$ and $Y$.

---
## 1. Problem Formulation

Our MPC problem is formally defined as follows:
```math
\begin{aligned}
\max&\quad \sum_{k = 0}^{K-1} \textbf{r}^\top A_k \\
\text{s.t.}&\quad 
e^\top A_k = 1, \qquad k = 0, \ldots, K - 1, \\
&\quad e^\top P_k = 1, \qquad k = 0, \ldots, K, \\
&\quad P_{k+1} = \mathscr{P} A_k, \qquad k = 0, \cdots, K - 1, \\
&\quad  [Q_k]_{a, p} \leq [A_k]_a, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
&\quad [Q_k]_{a, p} \leq [P_k]_p, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
&\quad [Q_k]_{a, p} \geq [A_k]_a + [P_k]_p - 1, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
&\quad N_{k+1} \preceq N_k +  \text{tr}(\mathscr{T}^\top Q_k) \hat{\lambda} -  \textbf{L} A_k, \qquad k = 0, \ldots, K - 1, \\
&\quad N_{k+1} \preceq \textbf{N}, \qquad k = 0, \ldots, K - 1, \\
&\quad N_{k+1} \succeq S_k \circ \textbf{N}, \qquad k = 0, \ldots, K - 1, \\
&\quad N_{k+1} \succeq N_k + \text{tr}(\mathscr{T}^\top Q_k) \hat{\lambda} -  \textbf{L} A_k - \tau_{\max} \bar{\lambda} S_k, \qquad k = 0, \ldots, K - 1, \\
&\quad N_k - \textbf{L} A_k \succeq 0, \qquad k = 0, \ldots, K - 1, \\
&\quad D^{\ell}_{k+1} = D^{\ell}_k + \left(\mathscr{D}_k \textbf{L}^{\ell} - \textbf{U}^{\ell} \right) A_k, \qquad k = 0, \ldots, K -1, \\
&\quad D^{u}_{k+1} = D^{u}_k + \left(\mathscr{D}_k \textbf{L}^{u} - \textbf{U}^{u} \right) A_k, \qquad k = 0, \ldots, K -1, \\
&\quad e^\top D^{\ell}_k \leq 1, \qquad k = 0, \ldots, K, \\
&\quad e^\top D^{u}_k \leq 1, \qquad k = 0, \ldots, K,
\end{aligned}
```

where the maximization is taken over the variables $(A, P, Q, N, S, D^\ell, D^u)$. Our MPC aims to maximize the number of carried lots during $K$ steps of operations, where the dynamics of the lot arrival are modeled as a Poisson point process whose parameter is simply inferred from data via maximum likelihood estimation (MLE). 

---
### 1.1. Variables

Each variable represents:
- $A_k \in \{0, 1 \}^{|\mathcal{A}|}$: binary vector that represents the action chosen in the $k$-th step,

- $P_k \in \{0, 1\}^{|\mathcal{P}|}$: binary vector that represents the position of the lifter in the $k$-th step,

- $Q_k \in \mathbb{R}^{|\mathcal{A}| \times |\mathcal{P}|}$: auxillary matrix whose $(a, p)$-entry represents $[A_k]_a [P_k]_p$,

- $N_k \in \mathbb{R}^{|\mathcal{C}|}$: real vector that represents the expected number of waiting lots in the conveyors in the $k$-th step,

- $S_k \in \{0, 1\}^{|\mathcal{C}|}$: auxillary binary vector that are used linearize the constraints related to the capacities of the conveyors,

- $D_k^\ell, D^u_k \in \{0, 1 \}^{|\mathfrak{F}|}$: binary vector that represents the destination of a lot in lower/upper fork of the lifter, which take $\textbf{0}$ when lower/upper fork is empty.

---
### 1.2. Constraints

Then, the feasible choices of these variables are determined by the following linear constraints:


- The first $K$ constraints
$$e^\top A_k = 1, \qquad k = 0, \ldots, K-1,$$
encodes the fact that exactly one action is taken in each time step.

- Similarly, the following $K$ constraints
$$e^\top P_k = 1, \qquad k = 0, \ldots, K,$$
encodes the fact that the lifter occupies exact one position at a time.

- The below constraints illustrate how the position of the lifter changes when an action is applied:
$$P_{k+1} = \mathscr{P} A_k, \qquad k = 0, \cdots, K - 1,$$
where $\mathscr{P} \in \{0, 1 \}^{|\mathcal{P}| \times  |\mathcal{A}|}$ is defined as
```math
\mathscr{P}_{p, a} = \begin{cases} 1 & \text{if $a$ brings the lifter to $p$,} \\ 0 & \text{otherwise,}\end{cases} \qquad p \in \mathcal{P}, \quad a \in \mathcal{A}.
```
Note that the constraints are linear since the current position of the lifter does not affect the next position of the lifter. What depends on the current position is the execution time of the action, which will impose a difficulty later.

- The following group of constraints describes the changes in the number of remaining lots in the conveyor belts:
```math
\begin{equation}
\begin{split}
& [Q_k]_{a, p} \leq [A_k]_a, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& [Q_k]_{a, p} \leq [P_k]_p, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& [Q_k]_{a, p} \geq [A_k]_a + [P_k]_p - 1, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& N_{k+1} \preceq N_k +  \text{tr}(\mathscr{T}^\top Q_k) \hat{\lambda} -  \textbf{L} A_k, \qquad k = 0, \ldots, K - 1, \\
& N_{k+1} \preceq \textbf{N}, \qquad k = 0, \ldots, K - 1, \\
& N_{k+1} \succeq S_k \circ \textbf{N}, \qquad k = 0, \ldots, K - 1, \\
& N_{k+1} \succeq N_k + \text{tr}(\mathscr{T}^\top Q_k) \hat{\lambda} -  \textbf{L} A_k - \tau_{\max} \bar{\lambda} S_k, \qquad k = 0, \ldots, K - 1.
\end{split}
\tag{1}
\end{equation}
```
To explain the meaning of each constraint, we first note that  $N_k$ changes according to the following equation:
```math
\begin{equation}
\tag{2}
N_{k+1} = \min\left(N_k +  \left(\mathscr{T}_k^\top A_k\right) \hat{\lambda} -  \textbf{L} A_k, \; \textbf{N}  \right) , \qquad k = 0, \ldots, K - 1.
\end{equation}
```
where
- $\hat{\lambda} \in \mathbb{R}^{|\mathcal{C}|}$ is the parameter vector characterizing the frequencies of lot arrivals, which is estimated from data online via MLE,
- $\mathscr{T}_k \in \mathbb{R}^{|\mathcal{A}|}$ is an execution time vector determined in $k$-th step (dependent of the lifter position in $k$-th step),
- $\textbf{N} \in \mathbb{Z}^{|\mathcal{C}|}$ is a capacity vector defined as
```math
[\textbf{N}]_c = (\text{capacity of conveyor}\; c), \quad c\in \mathcal{C},
```
and $\textbf{L} \in \{0, 1 \}^{|\mathcal{C}| \times |\mathcal{A}|}$ represents a bit mask corresponding to loading operations:
```math
\begin{gather*}
\textbf{L} = \textbf{L}^u + \textbf{L}^{\ell}, \\
\text{where}\;[\textbf{L}^{u/\ell}]_{c, a} = \begin{cases}1 & \text{if $a$ loads a lot from $c$ at upper/lower fork,} \\ 0 & \text{otherwise.} \end{cases}
\end{gather*}
```

Here the term $\mathscr{T}_k^\top A_k$ indicates the execution time of the action taken in $k$-th step, thus the term $(\mathscr{T}_k^\top A_k)\hat{\lambda}$ corresponds to the expected number of lots newly arrived in the conveyors until $(k+1)$-th step. This is because the lot arrival is assumed to follow the Poisson point process.
Furthermore, note that $\mathscr{T}_k$ can be decomposed into $\mathscr{T}_k = \mathscr{T} P_k$, where $\mathscr{T} \in \mathbb{R}^{|\mathcal{A}| \times  |\mathcal{P}|}$ is an execution time matrix defined as
```math
[\mathscr{T}]_{a, p} = (\text{execution time of $a$ when the lifter is at $p$}), \quad a \in \mathcal{A}, \quad p \in \mathcal{P}.
```
Therefore, the equations (2) become
```math
\begin{equation}
N_{k+1} = \min\left( N_k + \left(P_k^\top \mathscr{T}^\top A_k \right)\hat{\lambda} - \textbf{L} A_k,\; \textbf{N} \right) , \qquad k = 0, \ldots, K - 1.
\tag{3}
\end{equation}
```
It is notable that the non-linearity of (3) comes from the following sources: *(i)* the execution time depends both on the current position of the lifter and the action to be taken, which results in the bilinear term $P_k^\top \mathscr{T}^\top A_k$, and *(ii)* the number of lots in the conveyors has to be truncated due to the capacity constraints, which is expressed as the minimum operation in the equation. 
We now show that these issues can be removed at the cost of increasing number of variables. First of all, we can write
```math
P_k^\top \mathscr{T}^\top A_k = \text{tr}\left( \mathscr{T}^\top Q_k \right)
```
if we define $Q_k = A_k P_k^\top$. However, note that when $x, y$, and $z$ are binary variables, then the identity $z = xy$ can alternatively represented as the following linear inequalities:
```math
\begin{aligned}
z &\leq x, \\
z &\leq y, \\
z &\geq x + y - 1.
\end{aligned}
```
Applying this trick to our case, we get the following set of linear inequalities:
```math
\begin{aligned}
&[Q_k]_{a, p} \leq [A_k]_a, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& [Q_k]_{a, p} \leq [P_k]_p, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& [Q_k]_{a, p} \geq [A_k]_a + [P_k]_p - 1, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}.
\end{aligned}
```
This results in
```math
\begin{aligned}
&[Q_k]_{a, p} \leq [A_k]_a, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& [Q_k]_{a, p} \leq [P_k]_p, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& [Q_k]_{a, p} \geq [A_k]_a + [P_k]_p - 1, \qquad a \in \mathcal{A}, \quad p \in \mathcal{P}, \\
& N_{k+1} = \min\left(N_k +  \text{tr}(\mathscr{T}^\top Q_k) \hat{\lambda} -  \textbf{L} A_k, \textbf{N}\right), \qquad k = 0, \ldots, K - 1,
\end{aligned}
```
Next, to replace the minimum in (2) by its linear variant, we note that the term $N_k + \left(P_k^\top \mathscr{T}^\top A_k \right)\hat{\lambda} - \textbf{L} A_k$ is bounded as follows:
```math
0 \preceq N_k + \left(P_k^\top \mathscr{T}^\top A_k \right)\hat{\lambda} - \textbf{L} A_k \preceq \textbf{N} + \tau_{\max}\bar{\lambda},
```
where $\tau_{\max} = \max_{a, p} [\mathscr{T}]_{a, p}$ is the maximum execution time, and $\bar{\lambda} \in \mathbb{R}$ is an upper bound of $[\hat{\lambda}]_c$'s, which can be taken to be sufficiently large. Then, we introduce the set of binary slack variables $S_k$ and apply a standard trick in MILP to rewrite (3):
```math
\begin{aligned}
& N_{k+1} \preceq N_k +  \text{tr}(\mathscr{T}^\top Q_k) \hat{\lambda} -  \textbf{L} A_k, \qquad k = 0, \ldots, K - 1, \\
& N_{k+1} \preceq \textbf{N}, \qquad k = 0, \ldots, K - 1, \\
& N_{k+1} \succeq S_k \circ \textbf{N}, \qquad k = 0, \ldots, K - 1, \\
& N_{k+1} \succeq N_k + \text{tr}(\mathscr{T}^\top Q_k) \hat{\lambda} -  \textbf{L} A_k - \tau_{\max} \bar{\lambda} S_k, \qquad k = 0, \ldots, K - 1.
\end{aligned}
```
Intuitively, each $[ S_k ]\_c$ serves as the indicator which takes $1$ when the capacity constraint is met in $(k+1)$-th step and takes $0$ when not. Indeed, it is easily checkable that $\[S_k\]\_c = 1$ forces $N_{k+1}$ to satisfy $\[N_{k+1}\]\_c = \[\textbf{N}\]\_c$, and vice versa.
Combining these constraints, we ultimately get (1).

- The following constraints imply that the lifter can not load a lot from the empty queue:
```math
N_k - \textbf{L} A_k \succeq 0, \qquad k = 0, \ldots, K - 1.
```
This is because $\textbf{L}A_k$ represents the number of lots transferred from the conveyors to the lifter in $k$-th step.

- One of the challenges of the problem is that the action constraint depends on the state that evolves in a stochastic manner. To detour this problem, our model assumes that the destinations of the incoming lots are randomly sampled and are given as the parameters of the optimization problem. 

Specifically, we define the destination parameters $\mathscr{D}_k \in \{0, 1 \}^{|\mathfrak{F}| \times |\mathcal{C}|}$ as
```math
\left[\mathscr{D}_k\right]_{f, c} = \begin{cases} 1 & \text{if 
the destination of a foremost lot at $c$ is $f$,} \\ 0 & \text{otherwise.}  \end{cases}
```
For $k = 1, \ldots, K$, $\mathscr{D}_k$ are generated uniformly at random, while the exact value of $\mathscr{D}_0$ is extracted from the current state of the environment.

Now, recall that $D^{u/\ell}_k \in \{0, 1\}^{|\mathfrak{F}|}$ denotes the destination of the lot loaded at upper/lower fork. Then, we have
```math
D^{u/\ell}_{k+1} = D^{u/\ell}_k + \left(\mathscr{D}_k \textbf{L}^{u/\ell} - \textbf{U}^{u/\ell} \right) A_k, \qquad k = 0, \ldots, K -1,
```
where $\textbf{U}^{u/\ell} \in \{0, 1 \}^{|\mathfrak{F}| \times  |\mathcal{A}|}$ is defined similarly to $\textbf{L}^{u/\ell}$, but for unloading operations.
- Finally, the capacity constraints for the lifter are expressed as
```math
e^\top D^{u/\ell}_k \leq 1, \qquad k = 0, \ldots, K,
```
implying that there are at most one lot in each fork, embracing the case when the forks are empty. 
Note that the constraints $D_k^{u/\ell} \succeq 0$ prevent unloading a lot from the empty fork, or misdirected unload operation: if $A_k$ represents such an action, then a component of $D_{k+1}^u$ or $D_{k+1}^\ell$ becomes negative, which violates $D_k^{u/\ell} \succeq 0$.

---

## 1.3. Objective Function

The objective function represents the number of carried lots during $K$ steps, where the vector $\textbf{r} \in \mathbb{R}^{|\mathcal{A}|}$ is defined as
```math
[\textbf{r}]_a = \text{(number of unloaded lots when $a$ is executed)}, \qquad a \in \mathcal{A}.
```
