

# EDEN: Estimación de la Media Distribuida Eficiente en Comunicación y Robusta para Aprendizaje Federado

Este repositorio contiene la implementación oficial de 'EDEN: Estimación de la Media Distribuida Eficiente en Comunicación y Robusta para Aprendizaje Federado' presentado en [ICML 2022](https://proceedings.mlr.press/v162/vargaftik22a.html).

> **Actualización (5 de mayo de 2026):** En *"A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work"* ([arXiv:2604.18555](https://arxiv.org/abs/2604.18555)), aclaramos la relación entre el trabajo reciente de TurboQuant y los esquemas anteriores de DRIVE (NeurIPS 2021) y EDEN (ICML 2022). Una visión general se publica en [Towards Data Science](https://towardsdatascience.com/how-a-2021-quantization-algorithm-quietly-outperforms-its-2026-successor/).
>
> Un punto práctico que vale la pena destacar aquí: este código fue escrito para *estimación de la media distribuida* (DME) y se incluye con la escala no sesgada de EDEN $S_\text{unb} = ‖x‖^2 / \langle y, q\rangle$. Si en su lugar estás cuantizando un solo vector (por ejemplo, para KV-cache o cuantización de pesos), puede ser preferible una elección diferente de $S$:
>
> - La **escala sesgada que minimiza el error cuadrático medio**, $S_\text{bias} = \langle y, q\rangle / ‖q‖^2$, desarrollada originalmente en [DRIVE (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html).
> - La **escala de norma coincidente**, $S_\text{mn} = ‖x‖ / ‖q‖$, utilizada en las implementaciones actuales de vLLM y TurboQuant+. Es la media geométrica de las escalas no sesgada y sesgada ($\sqrt{S_\text{unb} \cdot S_\text{bias}} = ‖x‖ / ‖q‖$) y preserva $‖\hat{x}‖_2 = ‖x‖_2$.

## Contexto

*EDEN* es una técnica de compresión con pérdida y no sesgada para la estimación de la media distribuida que maneja de forma natural y sencilla los presupuestos de comunicación heterogéneos y las pérdidas de paquetes.

## Estructura de carpetas 

Las carpetas `torch` y `tf` contienen la implementación de EDEN en PyTorch y TensorFlow, respectivamente.  

## Citación

Si encuentras este trabajo útil, por favor cítanos:

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
