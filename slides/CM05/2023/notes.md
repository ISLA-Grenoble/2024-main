The posterior probability for each class is
$$
\text{Prob}(Y = 1 \mid \mathbf{X} \in \mathcal{A}) = \dfrac{1}{Z(\mathcal{A})}\text{Prob}(Y = 1)\int_\mathcal{A}p(\mathbf{x} \mid Y = 1)\mathrm{d}\mathbf{x}
$$

$$
\text{Prob}(Y = 0 \mid \mathbf{X} \in \mathcal{A}) = \dfrac{1}{Z(\mathcal{A})}\text{Prob}(Y = 0)\int_\mathcal{A}p(\mathbf{x} \mid Y = 0)\mathrm{d}\mathbf{x}
$$

To study the Bayes classifier, we can define the quantity
$$
t = \text{Prob}(Y = 1 \mid \mathbf{X} \in \mathcal{A}) - \text{Prob}(Y = 0 \mid \mathbf{X} \in \mathcal{A})
$$
and check when it is positive, negative, or zero. Note that:
$$
t \geq 0 \iff \int_{\mathcal{A}}p(\mathbf{x}\mid Y = 1)\mathrm{d}\mathbf{x} \geq \dfrac{\text{Prob}(Y = 0)}{\text{Prob}(Y = 1)}\int_{\mathcal{A}}p(\mathbf{x}\mid Y = 0)\mathrm{d}\mathbf{x}
$$
