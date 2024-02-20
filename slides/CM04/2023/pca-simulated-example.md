Suppose the predictors are generated as per
$$
\mathbf{X} \sim \mathcal{N}\left(\boldsymbol{\mu}, \mathbf{\Sigma}\right) \quad \text{with}\quad \boldsymbol{\mu} = \left[\begin{array}{c}0 \\ 0\end{array}\right] \quad \text{and}\quad \mathbf{\Sigma} = \left[\begin{array}{cc}\cos(\theta) & -\sin(\theta) \\  \sin(\theta) & \cos(\theta)\end{array}\right]\left[\begin{array}{cc}\lambda_1 & 0 \\  0 & \lambda_2\end{array}\right]\left[\begin{array}{cc}\cos(\theta) & -\sin(\theta) \\  \sin(\theta) & \cos(\theta)\end{array}\right]
$$
Expanding the terms
$$
\begin{array}{rcl}
\mathbf{\Sigma} &=& \left[\begin{array}{cc}\cos(\theta) & -\sin(\theta) \\  \sin(\theta) & \cos(\theta)\end{array}\right]\left[\begin{array}{cc}\lambda_1 & 0 \\  0 & \lambda_2\end{array}\right]\left[\begin{array}{cc}\cos(\theta) & -\sin(\theta) \\  \sin(\theta) & \cos(\theta)\end{array}\right] \\[1em]
&=& \left[\begin{array}{cc}\cos(\theta) & -\sin(\theta) \\  \sin(\theta) & \cos(\theta)\end{array}\right]\left[\begin{array}{cc}\lambda_1\cos(\theta) & -\lambda_1\sin(\theta) \\  \lambda_2\sin(\theta) & \lambda_2\cos(\theta)\end{array}\right] \\[1em]
&=& \left[\begin{array}{cc} \lambda_1 - (\lambda_1 + \lambda_2)\sin^2(\theta) & -\frac{1}{2}(\lambda_1 + \lambda_2)\sin(2\theta)  \\ -\frac{1}{2}(\lambda_1 + \lambda_2)\sin(2\theta) & \lambda_2 - (\lambda_1 + \lambda_2)\sin^2(\theta) \end{array}\right] \\[1em]
&=&\left[\begin{array}{cc} \lambda_1  & 0  \\  0 & \lambda_2 \end{array}\right] - \dfrac{1}{2}{(\lambda_1 + \lambda_2)} \left[\begin{array}{cc} 2\sin^2(\theta) & \sin(2\theta)  \\ \sin(2\theta) &  2\sin^2(\theta) \end{array}\right]  \\[1em]
\end{array}
$$
The correlation coefficient is, therefore



