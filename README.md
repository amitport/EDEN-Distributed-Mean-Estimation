# EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning

This repository is the official implementation of 'EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning' presented at [ICML 2022](https://proceedings.mlr.press/v162/vargaftik22a.html).

> **Update (May 5, 2026):** In *"A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work"* ([arXiv:2604.18555](https://arxiv.org/abs/2604.18555)), we clarify the relationship between the recent TurboQuant work and the earlier DRIVE (NeurIPS 2021) and EDEN (ICML 2022) schemes. An overview is published in [Towards Data Science](https://towardsdatascience.com/how-a-2021-quantization-algorithm-quietly-outperforms-its-2026-successor/).
>
> One practical point worth highlighting here: this code was written for *distributed mean estimation* (DME) and ships with EDEN's unbiased scale $S_\text{unb} = ‖x‖^2 / \langle y, q\rangle$. If you are quantizing a single vector instead (e.g., for KV-cache or weight quantization), a different choice of $S$ may be preferable:
>
> - The **MSE-minimizing biased scale**, $S_\text{bias} = \langle y, q\rangle / ‖q‖^2$, originally developed in [DRIVE (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html).
> - The **matched-norm scale**, $S_\text{mn} = ‖x‖ / ‖q‖$, used in the current vLLM and TurboQuant+ implementations. It is the geometric mean of the unbiased and biased scales ($\sqrt{S_\text{unb} \cdot S_\text{bias}} = ‖x‖ / ‖q‖$) and preserves $‖\hat{x}‖_2 = ‖x‖_2$.

## Context

*EDEN* is a lossy unbiased compression technique for distributed mean estimation that handles heterogeneous communication budgets and packet losses naturally and simply.

## Folder structure 

The `torch` and `tf` folders contain EDEN's implementation in PyTorch and TensorFlow, respectively.  

## Citation

If you find this useful, please cite us:

```bibtex
@InProceedings{pmlr-v162-vargaftik22a,
  title = 	 {{EDEN}: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning},
  author =       {Vargaftik, Shay and Basat, Ran Ben and Portnoy, Amit and Mendelson, Gal and Itzhak, Yaniv Ben and Mitzenmacher, Michael},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {21984--22014},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/vargaftik22a/vargaftik22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/vargaftik22a.html},
  abstract = 	 {Distributed Mean Estimation (DME) is a central building block in federated learning, where clients send local gradients to a parameter server for averaging and updating the model. Due to communication constraints, clients often use lossy compression techniques to compress the gradients, resulting in estimation inaccuracies. DME is more challenging when clients have diverse network conditions, such as constrained communication budgets and packet losses. In such settings, DME techniques often incur a significant increase in the estimation error leading to degraded learning performance. In this work, we propose a robust DME technique named EDEN that naturally handles heterogeneous communication budgets and packet losses. We derive appealing theoretical guarantees for EDEN and evaluate it empirically. Our results demonstrate that EDEN consistently improves over state-of-the-art DME techniques.}
}
```
