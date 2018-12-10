
Задачу обучения по прецедентам ![](https://latex.codecogs.com/svg.latex?Y%20%3D%20%5Cmathbb%7BR%7D) принято называть *восстановлением регрессии*. Постановка задачи аналогична. 

Модель алгоритмов задана в виде парам-кого семейства функций ![](https://latex.codecogs.com/svg.latex?%5Cinline%20f%28x%2C%5Calpha%20%29), а ![](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha%20%5Cin%20%5Cmathbb%7BR%7D%5Ep) - вектор параметров.


Функционал качества аппроксимации целевой зависимости на выборке - сумма квадратов ошибок - *остаточная сумма квадратов, RSS*: 

![](https://latex.codecogs.com/svg.latex?%5Cinline%20Q%28%5Calpha%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%3D1%7D%5Elw_i%28f%28x_i%2C%5Calpha%29-y_i%29%5E2)
, где ![](https://latex.codecogs.com/svg.latex?%5Cinline%20w_i) - вес, важность объекта i. 


Обучение по методу наименьших квадратов - поиск вектора параметров ![](https://latex.codecogs.com/svg.latex?%5Calpha^*), где достигается минимум среднего квадрата ошибки на выборке: 

![](https://latex.codecogs.com/svg.latex?%5Calpha%5E*%20%3D%20%5Carg%5Cmin_%7B%5Calpha%5Cin%5Cmathbb%7BR%7D%5Ep%7D%20Q%28%5Calpha%2C%20X%5El%29)

Решение оптимизационной задачи - использование необходимого условия минимума. Оно же и принимается за искомый вектор ![](https://latex.codecogs.com/svg.latex?%5Calpha^*).

## Непараметрическая регрессия. Ядерное сглаживание.

Непараметрическое восстановление регрессии основано на той же идее, непараметрическое восстановление плотности распределения.


Значение *a(x)* вычисляется для каждого объекта по нескольким ближайшим к нему объектам выборки. Для оценки близости на множестве должна быть задана функция расстояния ![](https://latex.codecogs.com/svg.latex?%5Crho%28x%2Cx%5E%7B%27%7D%29)

## Формула Надарая–Ватсона

Для вычисления значения ![](https://latex.codecogs.com/svg.latex?a%28x%29%20%3D%20%5Calpha%20%5Cforall%20x%20%5Cin%20X), воспользуемся МНК: 

![](https://latex.codecogs.com/svg.latex?Q%28%5Calpha%3B%20X%5El%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bl%7Dw_i%28x%29%28%5Calpha-y_i%29%5E2%20%5Crightarrow%20%5Cmin_%7B%5Calpha%5Cin%5Cmathbb%7BR%7D%7D)

Зададим веса, убывающие по мере увеличения расстояния. Для этого введём невозрастающую, гладкую, ограниченную
функцию ![](https://latex.codecogs.com/svg.latex?K%3A%5B0%2C%20%5Cinfty%29%5Crightarrow%20%5B0%2C%20%5Cinfty%29) - ядро: 

![](https://latex.codecogs.com/svg.latex?w_i%28x%29%20%3D%20%5Cleft%28%20K%5Cfrac%7B%5Crho%28x%2Cx_i%29%7D%7Bh%7D%5Cright%29)

*h* - ширина ядра/окна сглаживания. Чем меньше *h*, тем быстрее будут убывать веса ![](https://latex.codecogs.com/svg.latex?%5Cinline%20w_i(x)) по мере удаления *xi* от *x*

Приравняв нулю производную ![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Calpha%7D%3D0) получим формулу ядерного сглаживания Надарая–Ватсона:

![](https://latex.codecogs.com/gif.latex?a_h%28x%3B%20X%5El%29%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5El%20y_i%20w_i%28x%29%7D%7B%5Csum_%7Bi%3D1%7D%5El%20w_i%28x%29%7D%20%3D%20%5Cfrac%7B%20%5Csum_%7Bi%3D1%7D%5El%20y_i%20K%20%5Cleft%28%5Cfrac%7B%5Crho%28x%2Cx_i%29%7D%7Bh%7D%5Cright%29%7D%7B%5Csum_%7Bi%3D1%7D%5El%20K%20%5Cleft%28%5Cfrac%7B%5Crho%28x%2Cx_i%29%7D%7Bh%7D%5Cright%29%7D)

,т.е, значение есть среднее ![](https://latex.codecogs.com/gif.latex?y_i) по объектам ![](https://latex.codecogs.com/gif.latex?x_i), ближайшим к ![](https://latex.codecogs.com/gif.latex?x)

#### Реализация

Был взят датасет *[diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)* из *scikit-learn**

Подбор параметра *h* происходит с помощью LOO(leave-one-out) CV. 
Для решения задачи был реализован класс Regression, содержащий необходимый функционал.

метод для конечного обучения модели (поиск минимальной ширины окна)
```python
        h = prec
        # if self._ident is 0:
        curr_loo = self.__loo(x, y, h)
        min_loo = curr_loo
        min_h = h
        while curr_loo > prec and h > step:
            h -= step
            print(h)
            curr_loo = self.__loo(x, y, h)
            if min_loo > curr_loo:
                min_loo = curr_loo
                min_h = h
        self._h = min_h
        self._x = x
        self._y = y
```
метод **kern_smooth** для подсчёта исходной формулы на итерации.
```python
            for xi, yi in zip(x, y):
                dist = dist(cur_x, xi)
                ker_v = ker(dist / h)
                denor += ker_v
                numer += yi * ker_v
        if denor > 0:
            res = numer / denor
        return res
```
Пресловутый LOO CV
```python
        res = 0
        count = 1
        for xi, yi in zip(x, y):
            new_x = np.delete(x, xi)
            new_y = np.delete(y, yi)
            smooth = self.__kern_smooth(xi, new_x, new_y, self._kernel, self._dist, h)
            val = (smooth - yi) ** 2
            res += val
            count += 1
```
Были применены квартическое ядро и ядро Епанечникова. (Ядро почти не влияет на результат, разница между ними минимальна). Стартовые параметры: *precision=0.007, step=0.04, featr=3, n=20*


| h_min  |  method | core          | SSE     |
|-------|---------|---------------|---------|
|  0.019(9)     | NW      | quartic       | 6714.07 |
|   0.03    | NW      | epanenchnekov | 6737.99 |

![](NW.png)
