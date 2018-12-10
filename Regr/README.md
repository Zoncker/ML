# Восстановление регрессии
Задачу обучения по прецедентам ![](https://latex.codecogs.com/svg.latex?Y%20%3D%20%5Cmathbb%7BR%7D) принято называть *восстановлением регрессии*. Постановка задачи аналогична. 

Модель алгоритмов задана в виде парам-кого семейства функций ![](https://latex.codecogs.com/svg.latex?%5Cinline%20f%28x%2C%5Calpha%20%29), а ![](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha%20%5Cin%20%5Cmathbb%7BR%7D%5Ep) - вектор параметров.


Функционал качества аппроксимации целевой зависимости на выборке - сумма квадратов ошибок - *остаточная сумма квадратов, RSS*: 

![](https://latex.codecogs.com/svg.latex?%5Cinline%20Q%28%5Calpha%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%3D1%7D%5Elw_i%28f%28x_i%2C%5Calpha%29-y_i%29%5E2)
, где ![](https://latex.codecogs.com/svg.latex?%5Cinline%20w_i) - вес, важность объекта i. 


Обучение по методу наименьших квадратов - поиск вектора параметров ![](https://latex.codecogs.com/svg.latex?%5Calpha^*), где достигается минимум среднего квадрата ошибки на выборке: 

![](https://latex.codecogs.com/svg.latex?%5Calpha%5E*%20%3D%20%5Carg%5Cmin_%7B%5Calpha%5Cin%5Cmathbb%7BR%7D%5Ep%7D%20Q%28%5Calpha%2C%20X%5El%29)

Решение оптимизационной задачи - использование необходимого условия минимума. Если функция ![](https://latex.codecogs.com/gif.latex?f%28x%2C%5Calpha%29) достаточное число раз дифференцируема по ![](https://latex.codecogs.com/gif.latex?%5Calpha), то в точке минимума выполняется система *p* уравнений относительно *p* неизвестных:


![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Calpha%7D%28%5Calpha%2C%20X%5El%29%20%3D%202%5Csum%5El_%7Bi%3D1%7Dw_i%28f%28x_i%2Ca%29-y_i%29%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5Calpha%7D%28x_i%2C%20%5Calpha%29%20%3D%200)

Решение же и принимается за искомый вектор ![](https://latex.codecogs.com/svg.latex?%5Calpha^*).

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

###  LOWESS (locally weighted scatterplot smoothing)

NW - оценка очень чувствительна к одиночным выбросам. Т.е, чем больше величина ошибки 

![](https://latex.codecogs.com/gif.latex?%5Cvarepsilon_i%20%3D%20%7Ca_h%28x_i%3BX%5El/%5C%7Bx_i%5C%7D%29-y_i%7C)

тем в большей степени объект является выбросом, и тем меньге должен быть его вес. Домножаем веса ![](https://latex.codecogs.com/svg.latex?%5Cinline%20w_i(x)) на коэффициенты ![](https://latex.codecogs.com/gif.latex?%5Cgamma_i%20%3D%20%5Ctilde%7BK%7D%28%5Cvarepsilon_i%29)

#### Алгоритм LOWESS:
обучающая выборка ![](https://latex.codecogs.com/gif.latex?X%5El)

Выход:

коэффициенты ![](https://latex.codecogs.com/gif.latex?%5Cgamma_i,i=1,%5Cdots,l)

1. инициализация: ![](https://latex.codecogs.com/gif.latex?%5Cgamma_i=1,i=1,%5Cdots,l)
2. повторять
3. вычислить оценки скользящего контроля на каждом объекте: ![](https://latex.codecogs.com/gif.latex?a_h%28x%3B%20X%5El%29%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5El%20y_i%20w_i%28x%29%7D%7B%5Csum_%7Bi%3D1%7D%5El%20w_i%28x%29%7D%20%3D%20%5Cfrac%7B%20%5Csum_%7Bi%3D1%7D%5El%20y_i%20K%20%5Cleft%28%5Cfrac%7B%5Crho%28x%2Cx_i%29%7D%7Bh%7D%5Cright%29%7D%7B%5Csum_%7Bi%3D1%7D%5El%20K%20%5Cleft%28%5Cfrac%7B%5Crho%28x%2Cx_i%29%7D%7Bh%7D%5Cright%29%7D,i=1,%5Cdots,l)
4. вычислить коэффициенты ![](https://latex.codecogs.com/gif.latex?%5Cgamma_i) :
![](https://latex.codecogs.com/gif.latex?%5Cgamma_i%20%3D%20%5Ctilde%7BK%7D%28%7Ca_i-y_i%7C%29%2C%20i%20%3D1%2C%5Cdots%2Cl)
5. Пока коэффициенты не стабилизируются. 

#### Реализация

Различия с NW методом заключаются в дополнительном учёте новых коэффициентов ![](https://latex.codecogs.com/gif.latex?%5Cgamma_i).

| hopt  |  method | core          | SSE     |
|-------|---------|---------------|---------|
|  0.019(9)     | NW      | quartic       | 6714.07 |
|   0.03    | NW      | epanenchnekov | 6737.99 |
|   0.05   | LOWESS     | quartic | 6152.48 |


![](res.png)


## Многомерная линейная регрессия
Имеется набор *n* вещественных признаков ![](https://latex.codecogs.com/gif.latex?f_j%28x%29%20%2C%20j%3D1%2C%5Cdots%2Cn)
. Решение системы  

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Calpha%7D%28%5Calpha%2C%20X%5El%29%20%3D%202%5Csum%5El_%7Bi%3D1%7Dw_i%28f%28x_i%2Ca%29-y_i%29%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5Calpha%7D%28x_i%2C%20%5Calpha%29%20%3D%200)

существенно упрощается, если модель алгоритмов линейна по ![](https://latex.codecogs.com/gif.latex?%5Calpha%20%5Cin%20%5Cmathbb%7BR%7D%5En):

![](https://latex.codecogs.com/gif.latex?f%28x%2C%20%5Calpha%29%20%3D%20%5Csum_%7Bj%3D1%7D%5En%5Calpha_jf_j%28x%29)

Вводятся матричные обозначения: матрицу информации *F* , целевой вектор *y*, вектор параметров *α* и диагональную матрицу весов *W* :


![](https://latex.codecogs.com/gif.latex?F%20%3D%20%5Cbegin%7Bpmatrix%7D%20f_1%28x_1%29%20%26%20%5Cdots%20%26%20f_n%28x_n%29%5C%5C%20%5Cdots%20%26%20%5Cdots%20%26%20%5Cdots%5C%5C%20f_1%28x_l%29%20%26%20%5Cdots%20%26%20f_n%28x_l%29%20%5Cend%7Bpmatrix%7D%2C%20y%20%3D%20%5Cbegin%7Bpmatrix%7D%20y_1%5C%5C%20%5Cdots%5C%5C%20y_l%5C%5C%20%5Cend%7Bpmatrix%7D%2C%20%5Calpha%20%3D%20%5Cbegin%7Bpmatrix%7D%20%5Calpha_1%5C%5C%20%5Cdots%5C%5C%20%5Calpha_l%5C%5C%20%5Cend%7Bpmatrix%7D%2C%20%5Cbegin%7Bpmatrix%7D%20%5Csqrt%7Bw_1%7D%20%26%20%260%20%5C%5C%20%26%20%5Cddots%20%26%20%5C%5C%200%26%20%26%20%5Csqrt%7Bw_l%7D%20%5Cend%7Bpmatrix%7D)


В матричных обозначениях функционал среднего квадрата ошибки принимает вид 

![](https://latex.codecogs.com/gif.latex?Q%28a%29%20%3D%20%5Cleft%20%5C%7C%20W%28F%5Calpha-y%29%20%5Cright%20%5C%7C%5E2)

### Нормальная система уравнений
Необходимое условие минимума в матричном виде:
 

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Calpha%7D%28%5Calpha%29%20%3D%202F%5ET%28F%5Calpha-y%29%3D0%20%5Crightarrow%20F%5ETF%5Calpha%20%3D%20F%5ETy)

Эта система линейных уравнений относительно α называется нормальной системой
для задачи наименьших квадратов. Матрица ![](https://latex.codecogs.com/gif.latex?F%5ETF) имеет размер **n × n** и совпада-
ет с ковариационной матрицей набора признаков ![](https://latex.codecogs.com/gif.latex?f_1%2C%20%5Cdots%2C%20f_n)

### SVD. Линейная регрессия

Если число признаков не превышает число объектов, ![](https://latex.codecogs.com/gif.latex?n%20%5Cleq%20l) и среди столбцов **F** нет л.з., то **F** можно представить сингулярным разложением (singular value decomposition): 

![](https://latex.codecogs.com/gif.latex?F%20%3D%20VDU%5ET)

свойства:
1. **l × n** матрица **V** ортогональна, ![](https://latex.codecogs.com/gif.latex?V%5ETV%20%3D%20I_n) , и составлена из **n** собственных векторов матрицы ![](https://latex.codecogs.com/gif.latex?FF%5ET), соответствующих ненулевым собственным значениям;
2. **n × n** матрица **U** ортогональна, ![](https://latex.codecogs.com/gif.latex?U%5ETU%20%3D%20I_n) , и составлена из собственных векторов матрицы  ![](https://latex.codecogs.com/gif.latex?F%5ETF);
3. **n × n** матрица **D** диагональна, ![](https://latex.codecogs.com/gif.latex?D%20%3D%20diag%28%5Csqrt%7B%5Clambda_1%7D%2C%5Cdots%2C%20%5Csqrt%28%5Clambda_n%29%29), ![](https://latex.codecogs.com/gif.latex?%5Clambda_1%2C%5Cdots%2C%20%5Clambda_n) - собственные значения матриц ![](https://latex.codecogs.com/gif.latex?F%5ETF%2C%20FF%5ET)

Имея сингулярное разложение,получаем решение задачи наименьших квадратов в явном виде, не прибегая к трудоёмкому обращению матриц: ![](https://latex.codecogs.com/gif.latex?UD%5E%7B-1%7DV%5ET) 

Вектор МНК- решения и МНК- аппроксимация *y* соответственно:

![](https://latex.codecogs.com/gif.latex?%5Calpha^*%20%3D%20F%5E&plus;y%20%3D%20UD%5E%7B-1%7DV%5ETy)

![](https://latex.codecogs.com/gif.latex?F%5Calpha%5E*%20%3D%20PFy%20%3D%20%28VDU%5ET%29UD%5E%7B-1%7DV%5ETy%20%3D%20VV%5ETy%20%3D%20%5Csum_%7Bj%3D1%7D%5Env_j%28v_j%5ETy%29)

#### Реализация
Был использован датасет [boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston) из всё того же scikit-learn
Реализуем решение нормальной системы:
```python
def linear_regression(x_train, y_train, x_test):
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)

    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = inv(product)
    w = dot(dot(theInverse, Xt), y)

    predictions = []
    x_test = np.array(x_test)
    for i in x_test:
        components = w[1:] * i
        predictions.append(sum(components) + w[0])
    predictions = np.asarray(predictions)
    return predictions
```
строим матрицы ошибок и график значений одного признака (он имеет наибольшую величину корреляции) для визуализации распределения (для подтверждения соответствия после применения SVD)

после получения первой функции, применяем к тем же данным SVD:
```python
    A = f_plot.values

    temp = A.T.dot(A)
    S, V = np.linalg.eig(temp)
    S = np.diag(np.sqrt(S))

    U = A.dot(V).dot(np.linalg.inv(S))
    reconstructed_2 = U.dot(S).dot(V.T)
    df_2 = pd.DataFrame(reconstructed_2, columns=f_plot.columns)
    
    train_2 = df_2[:train_size]
    test_2 = df_2[train_size:]

    x_train_2 = train_2.drop('Target', axis=1)
    y_train_2 = train_2['Target']

    x_test_2 = test_2.drop('Target', axis=1)
    y_test_2 = test_2['Target']

    res_2 = linear_regression(x_train_2, y_train_2, x_test_2)
```

Результаты:
![](linreg.png)
![](div.png)

| method | SSE |
|--------------|---------|
| SVD linreg     | 3553.86033953874|
| casual linreg |8654.375722902236 |


### Проблема мультиколлинеарности
Если ковариационная матрица ![](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Csum%20%3D%20F%5ETF) имеет неполный ранг, то её обращение невозможно. 

Часто встречается случай мультиколлинеарности, **Σ** имеет полный ранг, но близка к некоторой матрице неполного ранга.

Тогда **Σ** — матрица неполного псевдоранга,и плохо обусловлена. Столбцы почти линейно зависимы, условие л.з выполняется приближённо. Аналогично для **F** .

Геометрически - что объекты выборки сосредоточены около линейного подпространства меньшей размерности ***m < n***. 
Признаком мультиколлинеарности является наличие у матрицы **Σ** собственных значений, близких к нулю.

Число обусловленности **Σ**:

![](https://latex.codecogs.com/gif.latex?%5Cmu%28%5Csum%29%20%3D%20%5Cleft%20%5C%7C%20%5Csum%20%5Cright%20%5C%7C%5Cleft%20%5C%7C%20%5Csum%5E%7B-1%7D%20%5Cright%20%5C%7C%20%3D%20%5Cfrac%7B%5Cmax_%7Bu%3A%20%5Cleft%20%5C%7C%20u%20%5Cright%20%5C%7C%3D%201%7D%20%5Cleft%20%5C%7C%20%5Csum%20u%20%5Cright%20%5C%7C%20%7D%7B%5Cmin_%7Bu%3A%20%5Cleft%20%5C%7C%20u%20%5Cright%20%5C%7C%3D%201%7D%20%5Cleft%20%5C%7C%20%5Csum%20u%20%5Cright%20%5C%7C%20%7D%20%3D%20%5Cfrac%7B%5Clambda_%7B%5Cmax%7D%7D%7B%5Clambda_%7B%5Cmin%7D%7D)

(шутки ради, TeX - код этой формулы
\mu(\sum) = \left \| \sum \right \|\left \| \sum^{-1} \right \| = \frac{\max_{u: \left \| u \right \|= 1} \left \| \sum u \right \| }{\min_{u: \left \| u \right \|= 1} \left \| \sum u \right \| } = \frac{\lambda_{\max}}{\lambda_{\min}} )

где ![](https://latex.codecogs.com/gif.latex?%5Clambda_%7B%5Cmin%7D%2C%20%5Clambda_%7B%5Cmax%7D) - максимальное и минимальное собственные значения матрицы **Σ**, все нормы евклидовы.

### Ridge Regression

Для решения проблемы мультиколлинеарности припишем к функционалу **Q** дополнительное слагаемое, штрафующее большие значения нормы вектора весов ![](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20%5Calpha%20%5Cright%20%5C%7C): 


![](https://latex.codecogs.com/gif.latex?Q_%5Ctau%28%5Calpha%29%20%3D%20%5Cleft%20%5C%7C%20F%5Calpha%20-%20y%20%5Cright%20%5C%7C%5E2%20&plus;%20%5Ctau%5Cleft%20%5C%7C%20%5Calpha%20%5Cright%20%5C%7C%5E2)

где ![](https://latex.codecogs.com/gif.latex?%5Ctau) неотрицательный параметр. 

В случае мультиколлинеарности имеется бесконечно много векторов **α**, доставляющих функционалу Q значения, близкие к минимальному. 

Штрафное слагаемое - регуляризатор, благодаря которому выбирается решение с минимальной нормой. 

Приравнивая нулю производную ![](https://latex.codecogs.com/gif.latex?Q_%5Ctau%28%5Calpha%29) по параметру **α**, находим:


![](https://latex.codecogs.com/gif.latex?%5Calpha%5E*_%5Ctau%20%3D%20%28F%5ETF&plus;%5Ctau%20I_n%29%5E%7B-1%7DF%5ETy)

Перед обращением матрицы к ней добавляется «гребень» — диагональная матрица ![](https://latex.codecogs.com/gif.latex?%5Ctau%20I_n).

Добавление гребня к матрице ![](https://latex.codecogs.com/gif.latex?F%5ETF) увеличивает все её собственные значения на **τ** , но не изменяет cобственных векторов. 

Матрица становится хорошо обусловленной, оставаясь в то же время «похожей» на исходную.

Регуляризованная МНК - аппроксимация через SVD вектора *y*: 


![](https://latex.codecogs.com/gif.latex?F%5Calpha%5E*_%5Ctau%20%3D%20VDU%5ET%20%5Calpha%5E*_%5Ctau%20%3D%20V%20diag%5Cleft%28%5Cfrac%7B%5Clambda_j%7D%7B%5Clambda_j%20&plus;%20%5Ctau%7D%5Cright%29V%5ETy%20%3D%20%5Csum_%7Bj%3D1%7D%5Env_j%28v_j%5ETy%29%5Cfrac%7B%5Clambda_j%7D%7B%5Clambda_j%20&plus;%20%5Ctau%7D)


МНК-аппроксимация - разложение **y** по базису собственных векторов ![](https://latex.codecogs.com/gif.latex?FF%5ET).

Проекции на собственные векторы сокращаются, уменьшается и норма вектора коэффициентов.

Отсюда ещё одно название метода — сжатие (shrinkage) или сокращение весов (weight decay)
По мере увеличения **τ** ![](https://latex.codecogs.com/gif.latex?%5Calpha%5E*_%5Ctau) становится более устойчивым/ понижение эффективной размерности решения.

При использовании регуляризации эффективная размерность принимает значение от 0 до n, не обязательно целое, и убывает при возрастании **τ** : 

![](https://latex.codecogs.com/gif.latex?tr%20F%28F%5ETF%20%3D%20%5Ctau%20I_n%29%5E%7B-1%7D%20%3D%20tr%20diag%20%5Cleft%20%28%20%5Cfrac%7B%5Clambda_j%7D%7B%5Clambda_j%20&plus;%20%5Ctau%7D%20%5Cright%20%29%20%3D%20%5Csum_%7Bj%3D1%7D%5En%20%5Cfrac%7B%5Clambda_j%7D%7B%5Clambda_j%20&plus;%20%5Ctau%7D%20%3C%20n)

Подбирать  **τ**  можно по CV, но это слишком долгая процедура. На практике  **τ** - в диапазоне (0.1, 0.4)

### Реализация
```python
def solve_ridge_regression(X, y):
    wRR_list = []
    df_list = []
    for i in range(0, 5001, 1):
        lam_par = i
        xtranspose = np.transpose(X)
        xtransx = np.dot(xtranspose, X)
        if xtransx.shape[0] != xtransx.shape[1]:
            raise ValueError('Needs to be a square matrix for inverse')
        lamidentity = np.identity(xtransx.shape[0]) * lam_par
        matinv = np.linalg.inv(lamidentity + xtransx)
        xtransy = np.dot(xtranspose, y)
        wRR = np.dot(matinv, xtransy)
        _, S, _ = np.linalg.svd(X)
        df = np.sum(np.square(S) / (np.square(S) + lam_par))
        wRR_list.append(wRR)
        df_list.append(df)
    return wRR_list, df_list
```
```python
def getRMSEValues(X_test, y_test, wRRArray, max_lamda, poly):
    RMSE_list = []
    for lamda in range(0, max_lamda+1):
        wRRvals = wRRArray[lamda]
        y_pred = np.dot(X_test, wRRvals)
        RMSE = np.sqrt(np.sum(np.square(y_test - y_pred))/len(y_test))
        RMSE_list.append(RMSE)
    plotRMSEValue(max_lamda, RMSE_list, poly=poly)
```

## Результаты

![](ridge_rmse.png)
![](dfl.png)

Из решения мы знаем, что когда df (λ) является максимальным значением, это соответствует λ, равному 0, что является решением наименьших квадратов. 
Шестой признак  и чётвёртый имеют наибольшую величину, что указывает на то, что они являются наиболее важными характеристиками при определении стоимости недвижимости на этих данных.

Таким образом, мы можем сказать, что для решения наименьших квадратов признаки 4 и 6 являются наиболее важными, которые влияют на решение, но поскольку мы всегда стараемся упорядочить веса, чтобы они были маленькими, их веса получают наибольшее штраф, если включить гиперпараметр (λ) для ridgereg и изменить его

Поскольку λ всегда больше 0, увеличение значений λ приводит к уменьшению степеней свободы и регуляризованных весов для всех ковариат в нашем решении.
