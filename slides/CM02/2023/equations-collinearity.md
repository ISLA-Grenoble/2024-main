$$
\dfrac{1}{N}\mathbf{X}^T\mathbf{X} = \left[
\begin{array}{ccc}
1 & \bar{x}_1 & \bar{x}_2 \\
\bar{x}_1 & \overline{{x}^2_1} & \overline{{x}_1 x_{2}} \\
\bar{x}_2 & \overline{{x}_1 x_{2}} & \overline{{x}^2_2}
\end{array}
\right] \to \left[
\begin{array}{ccc}
1 & \mathbb{E}[X_1] & \mathbb{E}[X_2] \\
\mathbb{E}[X_1] & \Big(\mathbb{E}[X_1]\Big)^2+ \text{Var}(X_1) & \mathbb{E}[X_1]\mathbb{E}[X_2]+ \text{Cov}(X_1, X_2) \\
\mathbb{E}[X_2] & \mathbb{E}[X_1]\mathbb{E}[X_2]+ \text{Cov}(X_1, X_2) & \Big(\mathbb{E}[X_2]\Big)^2+ \text{Var}(X_2)
\end{array}
\right]
$$

$$
\dfrac{1}{N}\mathbf{X}^T\mathbf{X} \to \left[
\begin{array}{ccc}
1 & 0 & 0 \\
0 & \text{Var}(X_1) & \text{Cov}(X_1, X_2) \\
0 & \text{Cov}(X_1, X_2) & \text{Var}(X_2) \\
\end{array}
\right] = \left[
\begin{array}{ccc}
1 & 0 & 0 \\
0 & \sigma_1^2 & \rho \sigma_1\sigma_2  \\
0 & \rho \sigma_1\sigma_2 & \sigma_2^2 \\
\end{array}
\right]
$$

$$
\left(\dfrac{1}{N}\mathbf{X}^T\mathbf{X}\right)^{-1} = \dfrac{1}{(1-\rho^2)~\sigma_1^2\sigma_2^2}\left[
\begin{array}{ccc}
1 & 0 & 0 \\
0 & \sigma_2^2 & -\rho \sigma_1 \sigma_2 \\
0 & -\rho \sigma_1 \sigma_2 & \sigma_1^2 \\
\end{array}
\right]
$$

$$
\text{Var}(\hat{\beta}_1) = \dfrac{\sigma^2}{N} \times \dfrac{1}{(1-\rho^2)\sigma_1^2}
$$

and
$$
\text{Var}(\hat{\beta}_2) = \dfrac{\sigma^2}{N} \times \dfrac{1}{(1-\rho^2)\sigma_2^2}
$$
We have that
$$
\Sigma = \left[
\begin{array}{cc}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{array}
\right]
$$

$$
p(\lambda) = (\lambda-\sigma_1^2)(\lambda-\sigma_2^2) - \rho^2 \sigma_1^2 \sigma_2^2 = \lambda^2 - (\sigma_1^2 + \sigma_2^2)\lambda + (1-\rho^2)\sigma_1^2 \sigma_2^2
$$

$$
\Delta = \sigma_1^4 + 2(2\rho^2 - 1)\sigma_1^2 \sigma_2^2 + \sigma_2^4
$$

$$
\lambda = \dfrac{(\sigma_1^2 + \sigma_2^2)}{2} \pm \dfrac{1}{2}\sqrt{\sigma_1^4 + 2(2\rho^2-1)\sigma_1^2 \sigma_2^2 + \sigma_2^4}
$$



