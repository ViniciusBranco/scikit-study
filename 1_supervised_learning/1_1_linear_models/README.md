# 1.1 Modelos lineares

> **Referência:** [scikit-learn User Guide — 1.1 Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
> **Seção-mãe:** [1. Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
> **Teoria:** ISLR (*An Introduction to Statistical Learning*, 2ª ed.), cap. 3 (regressão linear), 4 (classificação), 6 (regularização) e 7 (além da linearidade).

---

## Visão geral

Modelos lineares são uma família de métodos em que o valor previsto $\hat{y}$ é uma **combinação linear dos atributos de entrada**:

$$\hat{y}(w, x) = w_0 + w_1 x_1 + \cdots + w_p x_p$$

$w = (w_1, \ldots, w_p)$ são os pesos aprendidos (atributo `coef_`) e $w_0$ é o intercepto (atributo `intercept_`). Apesar da simplicidade, são muito poderosos na prática: interpretáveis, computacionalmente eficientes, e a base de muitas técnicas de regularização, esparsidade, robustez e modelagem probabilística. As variantes abaixo diferem sobretudo na **função de perda** otimizada e/ou na **penalização** aplicada aos coeficientes.

Para classificação com modelos lineares, veja as seções **1.1.11 Regressão logística** e **1.1.12 Modelos lineares generalizados**.

---

## Índice das subseções

| Seção | Modelo | Pasta |
|---|---|---|
| 1.1.1  | Mínimos quadrados ordinários (OLS) | `1_1_1_ordinary_least_squares/` |
| 1.1.2  | Regressão e classificação Ridge | `1_1_2_ridge_regression_and_classification/` |
| 1.1.3  | Lasso | `1_1_3_lasso/` |
| 1.1.4  | Lasso multitarefa | `1_1_4_multi_task_lasso/` |
| 1.1.5  | Elastic Net | `1_1_5_elastic_net/` |
| 1.1.6  | Elastic Net multitarefa | `1_1_6_multi_task_elastic_net/` |
| 1.1.7  | Least Angle Regression (LARS) | `1_1_7_least_angle_regression/` |
| 1.1.8  | LARS Lasso | `1_1_8_lars_lasso/` |
| 1.1.9  | Orthogonal Matching Pursuit (OMP) | `1_1_9_orthogonal_matching_pursuit/` |
| 1.1.10 | Regressão bayesiana | `1_1_10_bayesian_regression/` |
| 1.1.11 | Regressão logística | `1_1_11_logistic_regression/` |
| 1.1.12 | Modelos lineares generalizados (GLM) | `1_1_12_generalized_linear_models/` |
| 1.1.13 | Gradiente descendente estocástico (SGD) | `1_1_13_stochastic_gradient_descent/` |
| 1.1.14 | Regressão robusta | `1_1_14_robustness_regression/` |
| 1.1.15 | Regressão quantílica | `1_1_15_quantile_regression/` |
| 1.1.16 | Regressão polinomial | `1_1_16_polynomial_regression/` |

---

## Resumo dos modelos

### 1.1.1 Mínimos quadrados ordinários (OLS, *Ordinary Least Squares*)

[`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) ajusta um modelo linear minimizando a **soma dos quadrados dos resíduos** (RSS, *Residual Sum of Squares*) entre os valores observados e os previstos: $\min_w \|Xw - y\|_2^2$. É o estimador clássico, sem penalização: simples, não viesado e eficiente, resolvido por SVD (*Singular Value Decomposition*, decomposição em valores singulares) em $O(n_\text{amostras} \cdot n_\text{atributos}^2)$. A principal limitação é a sensibilidade à **multicolinearidade**: quando os atributos são muito correlacionados, os coeficientes ficam instáveis e com alta variância. A opção `positive=True` força coeficientes não negativos (NNLS, *Non-Negative Least Squares*), útil para grandezas fisicamente restritas.

---

### 1.1.2 Regressão e classificação Ridge

[`sklearn.linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) estende o OLS com uma penalização **L2** sobre o tamanho dos coeficientes: $\min_w \|Xw - y\|_2^2 + \alpha\|w\|_2^2$. O hiperparâmetro $\alpha \geq 0$ controla o encolhimento: $\alpha$ maior empurra os coeficientes para zero, o que traz estabilidade numérica sob multicolinearidade ao custo de um pequeno viés. O Ridge também faz classificação com `RidgeClassifier`, que codifica as classes e aplica a regressão Ridge à função de decisão. A escolha automática de $\alpha$ por validação cruzada está em `RidgeCV`.

---

### 1.1.3 Lasso

[`sklearn.linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) (*Least Absolute Shrinkage and Selection Operator*) usa uma penalização **L1**: $\min_w \frac{1}{2n}\|Xw - y\|_2^2 + \alpha\|w\|_1$. A propriedade-chave da L1 é promover **esparsidade**: muitos coeficientes vão exatamente a zero, o que faz uma seleção automática de atributos. Isso é valioso em problemas de alta dimensão em que só alguns preditores devem importar. Quando há atributos correlacionados, o Lasso escolhe um arbitrariamente e descarta os outros; nesse caso, prefira o Elastic Net. O ajuste é feito por descida coordenada.

---

### 1.1.4 Lasso multitarefa

[`sklearn.linear_model.MultiTaskLasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html) estende o Lasso para **várias saídas**: vários problemas de regressão resolvidos juntos, supondo que compartilham os mesmos atributos relevantes. Aplica uma norma mista $\ell_1 / \ell_2$ (norma L21) à matriz de coeficientes, o que impõe **esparsidade conjunta**: o mesmo subconjunto de atributos é selecionado para todas as tarefas. É adequado quando as saídas são relacionadas — por exemplo, prever ao mesmo tempo vários nutrientes do solo (N, P, K) a partir do mesmo conjunto de atributos espectrais.

---

### 1.1.5 Elastic Net

[`sklearn.linear_model.ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) combina as penalizações L1 e L2 num só termo: $\min_w \frac{1}{2n}\|Xw-y\|_2^2 + \alpha\rho\|w\|_1 + \frac{\alpha(1-\rho)}{2}\|w\|_2^2$. O parâmetro de mistura $\rho$ (`l1_ratio`) vai do Lasso puro ($\rho = 1$) ao Ridge ($\rho = 0$). O Elastic Net é preferível ao Lasso quando os atributos são muito correlacionados: a parte L2 cria um efeito de agrupamento (atributos correlacionados recebem coeficientes parecidos e não nulos, em vez de serem zerados arbitrariamente), enquanto a parte L1 ainda promove esparsidade.

---

### 1.1.6 Elastic Net multitarefa

[`sklearn.linear_model.MultiTaskElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html) junta a esparsidade conjunta do Lasso multitarefa com a robustez a correlações do Elastic Net. É a escolha natural quando há várias tarefas de regressão relacionadas, muitos atributos e preditores correlacionados. Como no Lasso multitarefa, os mesmos atributos são selecionados para todas as saídas; a diferença é que o termo L2 estabiliza a solução quando há atributos correlacionados, produzindo coeficientes mais confiáveis.

---

### 1.1.7 Least Angle Regression (LARS)

[`sklearn.linear_model.Lars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html) é um algoritmo eficiente para regressão linear em alta dimensão que constrói a solução **aos poucos**: a cada passo, identifica o atributo mais correlacionado com o resíduo atual e move o coeficiente nessa direção até que outro atributo fique igualmente correlacionado; daí em diante avança com os dois juntos. Equivale à regressão *stepwise* para a frente, mas de forma mais suave e linear por partes. É especialmente eficiente quando o número de atributos $p$ é muito maior que o de amostras $n$, e calcula naturalmente o **caminho de regularização** completo pelo custo de um único ajuste de OLS.

---

### 1.1.8 LARS Lasso

[`sklearn.linear_model.LassoLars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html) resolve o **Lasso pelo algoritmo LARS** (método de homotopia). Chega à mesma solução do Lasso por descida coordenada, mas usa a abordagem geométrica do LARS, o que permite calcular o **caminho completo do Lasso** (todos os valores de $\alpha$) numa única passada. O caminho é linear por partes, e cada "dobra" corresponde a um atributo entrando ou saindo do conjunto ativo. É útil quando o caminho completo é necessário para seleção ou análise de modelos.

---

### 1.1.9 Orthogonal Matching Pursuit (OMP)

[`sklearn.linear_model.OrthogonalMatchingPursuit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html) é um algoritmo **guloso de recuperação esparsa**. Partindo de um conjunto vazio, escolhe a cada passo o atributo mais correlacionado com o resíduo atual, projeta o resíduo no complemento ortogonal dos atributos já escolhidos e repete até atingir um número-alvo de coeficientes não nulos ou um limite para a norma do resíduo. Ao contrário do Lasso, o OMP controla a esparsidade diretamente por uma contagem, e não por uma força de regularização — o que é natural quando o conhecimento do domínio limita o número de preditores ativos.

---

### 1.1.10 Regressão bayesiana

[`sklearn.linear_model.BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html) e [`ARDRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html) colocam ***priors* gaussianas** sobre os coeficientes e estimam os hiperparâmetros (a precisão da *prior* e do ruído) a partir dos dados, por **maximização da evidência** (máxima verossimilhança tipo II, ou Bayes empírico). O `BayesianRidge` equivale ao Ridge, mas com regularização automática definida pelos dados, sem precisar de validação cruzada para $\alpha$. O ARD (*Automatic Relevance Determination*, determinação automática de relevância) vai além e dá uma ***prior* própria a cada coeficiente**, permitindo esparsidade extrema ao eliminar atributos irrelevantes, cuja precisão *a posteriori* diverge.

---

### 1.1.11 Regressão logística

[`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) é o modelo linear padrão para **classificação binária e multiclasse**. Apesar do nome, é um classificador: modela o logaritmo da razão de chances (*log-odds*) de pertencer à classe como função linear dos atributos e aplica a função logística (sigmoide), ou a *softmax* no caso multiclasse. Aceita regularização L1, L2 e Elastic Net, controlada pelo inverso da força de regularização, `C`. Os otimizadores incluem `lbfgs`, `liblinear`, `saga` e `newton-cg`; o `saga` é necessário para L1 e Elastic Net em problemas grandes. Para escolher `C` por validação cruzada, use `LogisticRegressionCV`. Teoria: ISLR cap. 4.

---

### 1.1.12 Modelos lineares generalizados (GLM, *Generalized Linear Models*)

O scikit-learn implementa vários **modelos lineares generalizados** além da hipótese gaussiana: [`TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html), [`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) e [`GammaRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html). Um GLM liga o preditor linear à distribuição do alvo por uma **função de ligação** e otimiza uma log-verossimilhança da família exponencial. Por isso servem para alvos não gaussianos:
- Poisson para contagens (ex.: incidência de pragas por talhão);
- Gamma para valores contínuos positivos e assimétricos (ex.: biomassa da cultura);
- Tweedie para distribuições compostas (ex.: perdas de seguro, volumes de chuva).

---

### 1.1.13 Gradiente descendente estocástico (SGD, *Stochastic Gradient Descent*)

[`sklearn.linear_model.SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) e [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) ajustam modelos lineares por **gradiente descendente estocástico**, atualizando os coeficientes uma amostra (ou um minilote) por vez. Isso os torna muito escaláveis para bases que não cabem na memória, com **aprendizado fora da memória** via `partial_fit`. Podem aproximar vários modelos lineares conforme o parâmetro `loss`: `squared_error` (OLS), `huber` (regressão robusta), `hinge` (SVM linear), `log_loss` (regressão logística), entre outros. Padronizar a escala dos atributos e ajustar o ritmo de aprendizado são essenciais para a convergência.

---

### 1.1.14 Regressão robusta

O scikit-learn oferece estimadores **resistentes a valores atípicos** no alvo: [`HuberRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html), [`RANSACRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html) e [`TheilSenRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html).
- `HuberRegressor` usa uma perda quadrática para resíduos pequenos e linear para os grandes, reduzindo o peso dos atípicos de forma suave.
- `RANSACRegressor` sorteia subconjuntos dos dados para achar um conjunto consensual de pontos "bons" e ajusta o estimador só neles.
- `TheilSenRegressor` usa a mediana das inclinações entre pares de pontos: alta robustez estatística, ao custo de complexidade $O(n^2)$.

---

### 1.1.15 Regressão quantílica

[`sklearn.linear_model.QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html) estima **quantis condicionais** da distribuição do alvo, em vez da média condicional, minimizando a **perda *pinball*** (perda absoluta inclinada): $L_q(y, \hat{y}) = q\max(y-\hat{y}, 0) + (1-q)\max(\hat{y}-y, 0)$. Com `quantile=0.5` obtém-se a mediana (mínimos desvios absolutos); com `quantile=0.9`, o percentil 90. A regressão quantílica é naturalmente **robusta a atípicos** e muito útil para construir **intervalos de previsão** — por exemplo, limites otimista e pessimista para a produtividade esperada de uma lavoura sob clima incerto.

---

### 1.1.16 Regressão polinomial: ampliando modelos lineares com funções de base

O scikit-learn não tem um estimador `PolynomialRegression` próprio. A **regressão polinomial** é montada combinando [`sklearn.preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) com qualquer modelo linear dentro de um [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). `PolynomialFeatures` gera todos os termos polinomiais de grau $d$ (e, se desejado, os termos de interação) a partir dos atributos originais. Assim, um modelo linear no espaço ampliado captura relações não lineares no espaço original. De forma mais geral, qualquer expansão de base não linear (*splines*, funções de base radial) segue o mesmo padrão — transformador de atributos + modelo linear — sem sair do arcabouço linear. Teoria: ISLR cap. 7.

---

> **Nota:** cada subpasta terá seu próprio `README.md`, com mais detalhes, formulação matemática, referência da API do scikit-learn e exemplos agronômicos.
