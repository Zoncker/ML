# Искусственные нейросети
## Метод обратного распространения ошибки

Метод вычисления градиента, который используется при обновлении весов многослойного перцептрона. 

Это итеративный градиентный алгоритм, который используется с целью минимизации ошибки работы многослойного перцептрона (MLP) и получения желаемого выхода.

Основная идея метода состоит в распространении сигналов ошибки от выходов сети к её входам, в направлении, обратном прямому распространению сигналов ошибки от выходов сети к её входам, в направлении, обратном прямому распространению сигналов в обычном режиме работы.

Для возможности применения метода обратного распространения ошибки активационная функция должна быть дифференцируема. Метод является модификацией метода градиентного спуска.

Сам базовый ***BackPropagation*** ![](https://latex.codecogs.com/svg.latex?%5Cinline%20%28%5Ceta%2C%20%5Calpha%2C%20%5C%7Bx%5Ed_i%2Ct%5Ed%5C%7D%5E%7Bn%2Cm%7D_%7Bi%3D1%2Cd%3D1%7D%2C%5Ctextup%7Bsteps%7D%29) :

1. Инициализировать ![](https://latex.codecogs.com/svg.latex?%5C%7Bw_%7Bij%7D%5C%7D_%7Bi%2Cj%7D) маленькими случайными значениями, ![](https://latex.codecogs.com/svg.latex?%5C%7B%5CDelta%20w_%7Bij%7D%5C%7D_%7Bi%2Cj%7D%3D0)
2. Повторить **n_steps** раз:
    -  Для всех **d** от *1* до *m* :
        - Подать ![](https://latex.codecogs.com/svg.latex?%5C%7Bx_i%5Ed%5C%7D)  на вход сети и подсчитать выходы ![](https://latex.codecogs.com/svg.latex?o_i) каждого узла.
        - Для всех ![](https://latex.codecogs.com/svg.latex?k%20%5Cin%20Outputs) **:** ![](https://latex.codecogs.com/svg.latex?%5Cdelta_k%20%3D%20o_k%281-o_k%29%28t_k-o_k%29)
        - Для каждого уровня **l**, начиная с предпоследнего: 
            - Для каждого узла **j** уровня **l** вычислить ![](https://latex.codecogs.com/svg.latex?%5Cdelta_j%20%3D%20o_j%281-o_j%29%20%5Csum%20_%7Bk%20%5Cin%20Child%28j%29%7D%5Cdelta_k%20w_%7Bj%2Ck%7D)
        - Для каждого ребра сети **{i, j}** : 
![](https://latex.codecogs.com/svg.latex?%5CDelta%20w_%7Bi%2Cj%7D%28n%29%20%3D%20%5Calpha%20%5CDelta%20w_%7Bi%2Cj%7D%28n-1%29%20&plus;%20%281-%5Calpha%29%5Ceta%5Cdelta_j%20o_i) ![](https://latex.codecogs.com/svg.latex?w_%7Bi%2Cj%7D%28n%29%3Dw_%7Bi%2Cj%7D%28n-1%29&plus;%5CDelta%20w_%7Bi%2Cj%7D%28n%29)
3. Выдать значения ![](https://latex.codecogs.com/svg.latex?w_%7Bi%2Cj%7D)


где ![](https://latex.codecogs.com/svg.latex?%5Calpha)  — коэффициент инерциальности для сглаживания резких скачков при перемещении по поверхности целевой функции

Несмотря на то, что функция уменьшается в направлении антиградиента, это не гарантирует самое быстрое схождение. Для этих целей были разработаны модификации исходного метода - алгоритмы сопряжённого градиента, где поиск происходит вдоль направлений сопряжения, что, в общем случае, приводит к более быстрой сходимости.

В большинстве алгоритмов обучения ранее, темп обучения используется для определения длины обновления весов (размер шага)
В алгоритмах сопряжённого градиента, размер шага изменяется на каждой итерации. В общем случае, суть методов следующая:

![](https://latex.codecogs.com/svg.latex?%5Cexists%20f%28x%29) **N** переменных, которую нужно минимизировать.

![](https://latex.codecogs.com/svg.latex?%5CDelta_x%20f) - градиент, направление вектора наискорейшего возрастания. Для минимизации, двигаемся в направлении антиградиента: ![](https://latex.codecogs.com/svg.latex?%5CDelta_%7Bx0%7D%20%3D%20-%5CDelta_xf%28x_0%29)

![](https://latex.codecogs.com/svg.latex?%5Calpha) - размер шага, подбирается бинарным поиском: ![](https://latex.codecogs.com/svg.latex?%5Calpha_0%20%3A%3D%20%5Carg%5Cmin_%5Calpha%20f%28x_0&plus;%5Calpha%5CDelta%20x_0%29)

На первой итерации движемся в направлении градиента ![](https://latex.codecogs.com/svg.latex?%5CDelta%20x_0). На следующих итерациях движемся вдоль направлений сопряжения ![](https://latex.codecogs.com/svg.latex?S_n%2C%20S_0%20%3D%20%5CDelta%20x_0)

На каждой итерации:
1. Рассчитаем направление: ![](https://latex.codecogs.com/svg.latex?%5CDelta%20x_n%20%3D%20-%20%5CDelta_x%20f%28x_n%29)
2. По формуле **Hestenes** и **Stiefel** вычислим : ![](https://latex.codecogs.com/svg.latex?%5Cbeta_n%20%3D%20%5Cfrac%7B%5CDelta%20x%5ET_n%28%5CDelta%20x_n%20-%20%5CDelta%20x_%7Bn-1%7D%29%7D%7BS%5ET_%7Bn-1%7D%28%5CDelta%20x_n%20-%20%5CDelta%20x_%7Bn-1%7D%29%7D)
3. Обновим направление ![](https://latex.codecogs.com/svg.latex?S_n%20%3D%20%5CDelta%20x_n%20&plus;%20%5Cbeta_nS_%7Bn-1%7D)
4. Оптимизируем направление (воспользуемся золотым сечением для поиска правой границы в спуске) ![](https://latex.codecogs.com/svg.latex?%5Calpha_n%20%3D%20%5Carg%5Cmin_%5Calpha%20f%28x_n%20&plus;%20%5Calpha%20S_n%29)
5. обновим позицию ![](https://latex.codecogs.com/svg.latex?x_%7Bn&plus;1%7D%20%3D%20x_n%20&plus;%20%5Calpha_n%20S_n)


### Реализация

Датасет был сгенерирован методом из **sklearn** **make_gaussian_quantiles**, который создаёт изотропные гауссовские выборки по квантилям. Датасет для классификации создаётся из многомерного нормального распределения и определения классов, разделённых вложенными концентрическими многомерными сферами так, что количество элементов в каждом классе почти одинаково
```python
    x1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=200, n_features=2,
                                     n_classes=2, random_state=1)
    x2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=1)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, - y2 + 1))
```

Был реализован класс **NeuralNetwork**, содержащий необходимый функционал (методы оригинального **BackProp**) 

```python
    def forward(self, x):
        # propagate inputs through the network
        self.I = [0]
        self.O = [x]
        ret = x
        for i in range(self.len - 1):
            I = ret.dot(self.W[i]) + self.b[i]
            O = self.f[i](I)
            self.I.append(I)
            self.O.append(O)
            ret = O
        # print(self.I)
        # quit()
        return ret
```

```python
```
