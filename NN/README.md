# Machine Learning course
Some ML labs and theory

-----
Nearest Neighbors
-----
Модуль [**sklearn.neighbors**](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
предоставляет пользователю функционал *neighbors-based* методов обучения с учителем и без. 
Метод ближайших соседов **(NN)** без учителя - основа для других подходов, например,
[***многомерного обучение***](http://scikit-learn.org/stable/modules/manifold.html) 
и [***спектральной кластеризации***](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html). 
Методы с учителем приводят к классификации для данных с дискретными ярлыками и регрессии - для непрерывных.

Подход основывается на поиске предопределённого количества обучающих элементов, ближайших по расстоянию к новой точке, и
получить прогноз для "соседей". Количество оных может быть определено пользователем (k - nearest neighbor learning), или
варьироваться, основываясь на локальной плотности точек (radius-based neighbor learning). Расстояние, в свою очередь,
может быть любой метрической мерой: стандартная Эвклидовая - самая популярная. Все методы NN также известны как 
non-generalizing (не обобщающие), поскольку они попросту "запоминают" все обучающие данные (возможна преобразование 
в быстро индексирующую структуру, Ball Tree, KD - Tree etc.)

Несмотря на простоту, NN - методы успешны в большом количестве проблем классификации и регрессии: рукописные цифры и 
изображения со спутников. Будучи непараметрическим методом, он часто успешен в ситуациях классификации, где граница 
решения очень неровна.

Классы в [**sklearn.neighbors**](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors) могут 
принимать как массивы NumPy, так и scipy.sparse матрицы на вход. Для плотных матриц поддерживается большое количество 
метрик расстояния. Для разрежённых матриц доступны произвольные метрики Минковского(для поиска).

[NearestNeighbors](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)
предоставляет унифицированный интерфейс для трёх алгоритмов (BallTree, KDTree и грубый алгоритм, основанный на методах в
[metrics.pairwise](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise) для вычислений
попарных расстояний в заданной метрике). Выбор алгоритма для поиска ближайших соседей контролируется через ключ 'algorithm'
: ['auto', 'ball_tree', 'kd_tree', 'brute']. В случае 'auto', алгоритм предпринимает попытку определить наилучший подход 
исходя из обучающей выборки, переданной методу fit. Обучение модели на разрежённых матрицах приведёт к перезаписи этого
параметра к brute force. 
Подробнее:

```python
class sklearn.neighbors.NearestNeighbors(n_neighbors=5,
                                         radius=1.0, 
                                         algorithm=’auto’, 
                                         leaf_size=30, 
                                         metric=’minkowski’, 
                                         p=2, 
                                         metric_params=None, 
                                         n_jobs=None, 
                                         **kwargs)
```
Параметры конструктора:

| parameter         | type                                      | req | default     | description                                                                                                                                                                                                                                                                                                          |
|-------------------|-------------------------------------------|-----|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **n_neighbors**   | int                                       | no  | 5           | Количество соседей для поиска (с помощью метода kneighbors)                                                                                                                                                                                                                                                          |
| **radius**        | float                                     | no  | 1.0         | Область пространства параметров для использования в методе radius_neigbors.                                                                                                                                                                                                                                          |
| **algorithm**     | {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’} | no  |             | Алгоритм нахождения ближайших соседей.                                                                                                                                                                                                                                                                               |
| **leaf_size**     | int                                       | no  | 30          | Размер листа, переданный BallTree/ KDTree. Может повлиять на скорость построения и на объёмы памяти для хранения дерева. Оптимальное значение зависит от конкретной задачи                                                                                                                                           |
| **metric**        | string/callable                           | yes | ‘minkowski’ | Метрика для использования в расчётах. Любая метрика из scikit-learn/scipy.spatial.distance. Если метрика - вызываемая функция, то она вызывается на каждую пару строк и записанную результирующую величину. Вызов должен принимать два массива на вход и возвращать одно значение, определяющее значение между ними. |
| **p**             | int                                       | no  | 2           | Параметр для метрик Минковского из sklearn.metrics.pairwise.pairwise_distances. p=1 экивалетно использованию manhattan_distance (l1), euclidean_distance (l2) для p = 2. Для произвольного p minkowski_distance (l_p)                                                                                                |
| **metric_params** | dict                                      | no  | None        | дополнительные ключи для функции метрики.                                                                                                                                                                                                                                                                            |
| n_jobs            | int/None                                  | no  | None        | Количество параллельных работ по поиску. None - 1.                                                                                                                                                                                                                                                                   |

Подходящие значения для метрик:

Из scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

Из scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]


| method                     | params                              | description                                          |
|----------------------------|-------------------------------------|------------------------------------------------------|
| **fit**                    | (X[, y])                            | Обучает модель с X - обучающей выборкой.             |
| **get_params**             | ([deep])                            | Получение параметров для оценки                      |
| **kneighbors**             | ([X, n_neighbors, return_distance]) | Вычисление k-соседей точки                           |
| **kneighbors_graph**       | ([X, n_neighbors, mode])            | Вычисление взвешенного графа k соседей для точек в X |
| **radius_neighbors**       | ([X, radius, return_distance])      | Вычисление соседей в данном радиусе точки/точек      |
| **radius_neighbors_graph** | ([X, radius, mode])                 | Вычисление взвешенного графа соседей для точек в X   |
| **set_params**             | (**params)                          | Установка параметров                                 |

***Пример***

Для простой задачи нахождения ближайших соседей может быть использован алгоритм без учителя из [**sklearn.neighbors**](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
```python
from sklearn.neighbors import NearestNeighbors
import numpy as np


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

print(indices, distances)

array([[0, 1],
       [1, 0],
       [2, 1],
       [3, 4],
       [4, 3],
       [5, 4]]...)
       
array([[0.        , 1.        ],
       [0.        , 1.        ],
       [0.        , 1.41421356],
       [0.        , 1.        ],
       [0.        , 1.        ],
       [0.        , 1.41421356]])      
```

Т.к запрашиваемый сет совпадает с обучающим, ближайший сосед каждой точки - она сама, и расстояние = 0.




### Brute Force

Методы быстрых вычислений ближайших соседей активно исследуются до сих пор. 
Наиболее *"наивный"* поиск соседей включает в себя *"грубые"* расчёты расстояний между всеми парами точек в выборке:
для **N** образцов с размерностью **D** этот подход оценивается в ![](https://latex.codecogs.com/svg.latex?O%5BD%20N%5E2%5D). Рационально использовать данный подход
для малых объёмов данных, потому как по мере роста выборки, согласно оценке, стремительно возрастает и сложность, делая 
метод неприменимым.

### K-Dimensional Tree

Для замещения неэффективности использования метода выше были созданы разнообразные структуры данных, основанные на ***деревьях***.
В общем случае, эти структуры пытаются уменьшить необходимое количество вычислений расстояний, эффективно кодируя 
информацию о совокупном расстоянии для выборки. Допустим, **A** очень далека от **B**, а **B** очень близка к **C**. 
Т.е, известно, что **A** очень далека от **C** без явного вычисления расстояния. (Транзитивность?). Как следствие,
стоимость вычилсений падает до  ![](https://latex.codecogs.com/svg.latex?O%5BD%20N%20%5Clog%28N%29%5D) или ниже. 
Разница с брутфорсом значительная (на **N** ).


![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/3dtree.png/250px-3dtree.png)
![](http://www.vlfeat.org/demo/kdtree_uniform_median.jpg)
![](https://i.stack.imgur.com/wzKb2.png)

Для реализации преимущества обобщённых вычислений были созданы **K - размерные деревья**, которые обобщают 
***двумерные квад-деревья*** и трёхмерные ***окт-деревья*** для произвольной размерности. 

KD - бинарное дерево, которое рекурсивно разбивает пространство параметров вдоль осей выборки, разделяя его на вложенные
неоднородные (ортотропные) области, которые заполнены объектами выборки. Такое дерево строится очень быстро, 
поскольку разбиение производится только вдоль осей выборки, вычисления D-мерных расстояний не требуются. Единожды
построенное, ближайший сосед любой точки может быть найден за ![](https://latex.codecogs.com/svg.latex?O%5B%5Clog%28N%29%5D)
Хоть данный подход и очень быстр для маломерных **(D<20)** случаев, он становится неэффективным с существеным ростом **D**,
 имеет место быть т.н. ***проклятие размерности***.

 Конструктор и методы класса аналогичны BallTree.
 
Примеры:

Код k ближайших соседей

```python 
import numpy as np
np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)              
dist, ind = tree.query(X[:1], k=3)                
print(ind)  # indices of 3 closest neighbors
[0 3 1]
print(dist)  # distances to 3 closest neighbors
[ 0.          0.19662693  0.29473397]
```

Перевод дерево в формат сериализации pickle. Состояние дерева сохранено в операции pickle,
его не нужно пересоздавать до анпиклинга (десериализации)

```python
import numpy as np
import pickle
np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)        
s = pickle.dumps(tree)                     
tree_copy = pickle.loads(s)                
dist, ind = tree_copy.query(X[:1], k=3)     
print(ind)  # indices of 3 closest neighbors
[0 3 1]
print(dist)  # distances to 3 closest neighbors
[ 0.          0.19662693  0.29473397]
```
Поиск соседей внутри заданного радиуса

```python
>>> import numpy as np
>>> np.random.seed(0)
>>> X = np.random.random((10, 3))  # 10 points in 3 dimensions
>>> tree = KDTree(X, leaf_size=2)     
>>> print(tree.query_radius(X[:1], r=0.3, count_only=True))
3
>>> ind = tree.query_radius(X[:1], r=0.3)  
>>> print(ind)  # indices of neighbors within distance 0.3
[3 0 1]
```

Расчёт плотности распределения гауссовским ядром

```python
import numpy as np
np.random.seed(1)
X = np.random.random((100, 3))
tree = KDTree(X)                
tree.kernel_density(X[:3], h=0.1, kernel='gaussian')
array([ 6.94114649,  7.83281226,  7.2071716 ])
```

Получить значение двухточечной автокорелляционной функции 

```python
import numpy as np
np.random.seed(0)
X = np.random.random((30, 3))
r = np.linspace(0, 1, 5)
tree = KDTree(X)                
tree.two_point_correlation(X, r)
#array([ 30,  62, 278, 580, 820])
```
 
### Ball-Tree

Данный метод решает проблему неэффективности KD - деревьев для многомерных случаев. Используется структура **ball tree**.
Вместо разбиения данных вдоль осей прямоугольной системы, проводится разбиение данных в последовательность вложенных гиперсфер.
Это делает операцию построения дерева несколько более дорогостоящей в сравнении с KD, но приводит к намного более 
удобной структуре, даже при работе с большими размерностями. 

Дерево-шар рекурсивно разбивает данные в ячейки, определённые центроидом **C** и радиусом **r**, такими, что каждая точка 
ячейки лежит внутри гиперсферы от **r** и **C**. Количество точек - кандидатов уменьшено использованием ***неравенства
треугольника***: ![](https://latex.codecogs.com/svg.latex?%7Cx&plus;y%7C%20%5Cleq%20%7Cx%7C%20&plus;%20%7Cy%7C)
Таким образом, одного расчёта расстояния между тестовой точкой и центроидом достаточно для определения нижней и верхней 
грани расстояний ко всем точкам ячейки. Благодаря сферической геометрии ячеек, метод превосходит KD в случае больших
размерностей, пусть и реальная производительность существенно зависит от структуры обучающей выборки.

![](https://slideplayer.com/slide/11882225/66/images/98/Ball+tree+example.jpg)

class sklearn.neighbors.BallTree

Параметры конструктора

| parameter | type                                          | required | default                 | description                                                                                                                                                                                                        |
|-----------|-----------------------------------------------|----------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **X**         | array-like,  shape = [n_samples,  n_features] | yes      |                         | n_samples - количество точек в датасете, n_features - размерность пространства параметров.  |
| **leaf_size** | positive int                                  |          | 40                      | Количество точек, при котором переключиться  на брутфорс. Изменение параметра не изменит результата запроса, но серьёзно повлияет на его скорость и требуемую память для хранения дерева ~ n_samples/leaf_size.    |
| **metric**    | string/DistanceMetric object                  |          | minkowski, p=2 (euclid) | метрика расстояний для использования с деревом                                                                                                                                                                     

Код k ближайших соседей

```python
from sklearn.neighbors import BallTree
import numpy as np
np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = BallTree(X, leaf_size=2)              
dist, ind = tree.query(X[:1], k=3)                
print(ind)  # indices of 3 closest neighbors
#[0 3 1]
print(dist)  # distances to 3 closest neighbors
#[ 0.          0.19662693  0.29473397]
```
Перевод дерево в формат сериализации pickle. Состояние дерева сохранено в операции pickle,
его не нужно пересоздавать до анпиклинга (десериализации)

```python
import numpy as np
import pickle
from sklearn.neighbors import BallTree
np.random.seed(0)
tree = BallTree(X, leaf_size=2)        
s = pickle.dumps(tree)                     
tree_copy = pickle.loads(s)                
dist, ind = tree_copy.query(X[:1], k=3)     
print(ind)  # indices of 3 closest neighbors
# [0 3 1]
print(dist)  # distances to 3 closest neighbors
# [ 0.          0.19662693  0.29473397]
```

Поиск соседей внутри заданного радиуса

```python
import numpy as np
from sklearn.neighbors import BallTree
np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = BallTree(X, leaf_size=2)     
print(tree.query_radius(X[:1], r=0.3, count_only=True))
#3
ind = tree.query_radius(X[:1], r=0.3)  
print(ind)  # indices of neighbors within distance 0.3
#[3 0 1]
```

Расчёт плотности распределения гауссовским ядром
```python
import numpy as np
from sklearn.neighbors import BallTree
np.random.seed(1)
X = np.random.random((100, 3))
tree = BallTree(X)                
tree.kernel_density(X[:3], h=0.1, kernel='gaussian')
#array([ 6.94114649,  7.83281226,  7.2071716 ])
```
Получить значение двухточечной автокорелляционной функции 

```python
import numpy as np
from sklearn.neighbors import BallTree
np.random.seed(0)
X = np.random.random((30, 3))
r = np.linspace(0, 1, 5)
tree = BallTree(X)                
tree.two_point_correlation(X, r)
#array([ 30,  62, 278, 580, 820])
```
Методы

| method                | params                                 | descr                                                                                                          |
|-----------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------|
| kernel_density        | (self, X, h[, kernel, atol, …])        | Рассчитывает плотность ядра  на точках из X с данным ядром, используя метрику, указанную  при создании дерева. |
| query_radius          | (X, r, count_only = False):            | определение радиуса запроса                                                                                    |
| query                 | (X[, k, return_distance, dualtree, …]) | запрос к дереву на k ближайших  соседей                                                                        |
| two_point_correlation | X, r, dualtree                         | Высчитывает двухточечную  корреляционную функцию                                                               |

## Метрики. DistanceMetric class

Класс предоставляет интерфейс к "быстрым" функциям метрик расстояния. Различные метрики могут быть обращены через get_metric метод

```python
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')
X = [[0, 1, 2],
         [3, 4, 5]]
dist.pairwise(X)
#array([[ 0.        ,  5.19615242],
#       [ 5.19615242,  0.        ]])
```

Метрики для действительных векторных пространств

| identifier    | class name          | args    | distance function           |
|---------------|---------------------|---------|-----------------------------|
| **"euclidean"**   | EuclideanDistance   |         | sqrt(sum((x - y)^2))        |
| **"manhattan"**   | ManhattanDistance   |         | sum(&#124;x - y&#124;)                |
| **"chebyshev"**   | ChebyshevDistance   |         | max(&#124;x - y&#124;)                |
| **"wminkowski"**  | WMinkowskiDistance  | p, w    | sum(&#124;w * (x - y)&#124;^p)^(1/p)  |
| **"seuclidean"**  | SEuclideanDistance  | V       | sqrt(sum((x - y)^2 / V))    |
| **"mahalanobis"** | MahalanobisDistance | V or VI | sqrt((x - y)' V^-1 (x - y)) |
| **"minkowski"**   | MinkowskiDistance   | p       | sum(&#124;x - y&#124;^p)^(1/p)        |

Метрики для двумерных векторных пространств
Метрика требует данные в виде  [latitude, longitude], оба входа и выхода в радианах.

| identifier   | class name        | distance function                                           |
|--------------|-------------------|-------------------------------------------------------------|
| **“haversine”**  | HaversineDistance | 2 arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy))) |

Метрики для целых векторных пространств (работают также и для действительных чисел)

| identifier | class name         | distance function                    |
|------------|--------------------|--------------------------------------|
| **“canberra”**  | CanberraDistance   | sum(&#124;x - y&#124; / (&#124;x&#124; + &#124;y&#124;))           |
| **“braycurtis”** | BrayCurtisDistance | sum(&#124;x - y&#124;) / (sum(&#124;x&#124;) + sum(&#124;y&#124;)) |
| **“hamming”**    | HammingDistance    | N_unequal(x, y) / N_tot              |

Пользовательская функция расстояния

| identifier   | class name        | distance function                                           |
|--------------|-------------------|-------------------------------------------------------------|
| **“pyfunc”**  | PyFuncDistance | func |

Функция должна принимать одномерный массив NumPy, и возвращает расстояние. Для использования внутри BallTree, расстояние
должно быть настоящей метрикой, оно должно удовлетворять следующим свойствам : 

* Неотрицательность: d(x, y) >= 0
* Идентичность: d(x, y) = 0 если и только если x == y
* Симметричность: d(x, y) = d(y, x)
* Неравенство треугольника: d(x, y) + d(y, z) >= d(x, z)

Из - за расходов на вызов функции объекта в пайтоне, это будет относительно медленно, но будет иметь такое же масштабирование,
как и у других расстояний.

Методы

| method   | descr        | 
|--------------|-------------------|
| **dist_to_rdist**  | Конвертирование настоящего расстояния в уменьшенное |
| **get_metric**  | Получить метрику по идентификатору |
| **pairwise**  | Расчёт попарного расстояния между x и y |
| **rdist_to_dist**  | обратное 1 |

## NN - классификация 

Классификация с NN - пример необобщающего или основанного-на-экземпляре обучения: из названия понятно, что алгоритм не 
пытается строить обобщённую внутреннюю модель, а просто хранит экземпляры обучающей выборки. Классификация расчитывается
из простого большинства голосов ближайших соседей каждой точки,  класс запрашиваемой точки - тот, чьих представителей 
наиболее среди N ближайших соседей. 
scikit-learn реализует два классификатора NN: [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) - обучение на k ближайших соседях каждой точки, k задано пользователем, когда же [RadiusNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier) - обучение на соседях каждой точки внутри определённого пользователем радиуса r (float). Оптимальный подбор k очень зависит от данных,
но, в общем случае, большее k подавляет эффекты шумов, но делает границы классификации менее точными. В случаях, где данные
не унифицированно собраны (неоднородны?), RadiusNeighborsClassifier может быть лучшим выбором. Пользователь выбирает r такой,
что точки в разрежённых соседствах используют меньшее количество соседей для классификации. Для многомерных параметрических
пространств этот метод малоэффективен вследствие проклятия размерности (тыц)

Базовая NN - классификация использует унифицированные веса. При некоторых обстоятельствах лучше взвешивать соседей по мере 
близости к объекту: чем ближе - тем весомее. Значение по умолчанию weights = 'uniform', присваивает унифицированные веса
соседям, weights = 'distance' - обратно пропорционально относительно расстояния до исходной точки. Возможно использование 
пользовательской функции расстояния для расчёта весов.

### KNeighborsClassifier

Конструктор

| parameter         | type                                      | req | default     | description                                                                                                                                                                                                                                                                                                          |
|-------------------|-------------------------------------------|-----|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **n_neighbors**   | int                                       | no  | 5           | Количество соседей для поиска (с помощью метода kneighbors)                                                                                                                                                                                                                                                          |
| **algorithm**     | {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’} | no  |             | Алгоритм нахождения ближайших соседей.                                                                                                                                                                                                                                                                               |
| **leaf_size**     | int                                       | no  | 30          | Размер листа, переданный BallTree/ KDTree. Может повлиять на скорость построения и на объёмы памяти для хранения дерева. Оптимальное значение зависит от конкретной задачи                                                                                                                                           |
| **metric**        | string/callable                           | yes | ‘minkowski’ | Метрика для использования в расчётах. Любая метрика из scikit-learn/scipy.spatial.distance. Если метрика - вызываемая функция, то она вызывается на каждую пару строк и записанную результирующую величину. Вызов должен принимать два массива на вход и возвращать одно значение, определяющее значение между ними. |
| **p**             | int                                       | no  | 2           | Параметр для метрик Минковского из sklearn.metrics.pairwise.pairwise_distances. p=1 экивалетно использованию manhattan_distance (l1), euclidean_distance (l2) для p = 2. Для произвольного p minkowski_distance (l_p)                                                                                                |
| **metric_params** | dict                                      | no  | None        | дополнительные ключи для функции метрики.                                                                                                                                                                                                                                                                            |
| **n_jobs**        | int/None                                  | no  | None        | Количество параллельных работ по поиску. None - 1.                                                                                                                                                                                                                                                                   |
| **weights**       | string/callable                           | no  | default = ‘uniform’        | Функция весов, используемая для прогнозов.                                                                                                                                                                                                                                                             |

Методы

| method                | params                                 | descr                                                                                                          |
|-----------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **fit**        | (X, y)       | обучение модели на данных |
| **get_params**          | ([deep])            | Получение параметров оценщика                                                                                   |
| **predict**                 | (X) | Классификация по массиву тестовых векторов X                                                                      |
| **predict_proba**                 | (X) | Получение вероятности для тестовых данных X                                                                     |
| **score** | (X, y[, sample_weight])                         | Получение средней точности на указанных тестовых данных и ответах                                                               |
| **set_params** | (**params)                        | Установка параметров оценщика 
| **kneighbors** | ([X, n_neighbors, return_distance])                      | Поиск k соседей точки  
| **kneighbors_graph** | (**params)                        | Расчёт взвешенного графа для k соседей точек X    

Пример
```python
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
print(neigh.predict([[1.1]]))
[0]
print(neigh.predict_proba([[0.9]]))
[[0.66666667 0.33333333]]
``` 

Пример RadiusNeighborsClassifier
```python
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import RadiusNeighborsClassifier
neigh = RadiusNeighborsClassifier(radius=1.0)
neigh.fit(X, y) 

print(neigh.predict([[1.5]]))
#[0]
```

### Nearest Centroid Classifier

[NearestCentroid](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid)
классификатор - простой алгоритм, который представляет каждый класс центроидом его членов. Это делает алгоритм похожим по эффекту 
на фазу обновления ярлыков в алгоритме k-средних (k-means). Также у алгоритма нет параметров на выбор, что делает его
отличным основополагающим (baseline) классификатором. Конечно, алгоритм страдает от невыпуклых классов, так же как и когда 
классы имеют существенно различные дисперсии, поскольку предполагается равная дисперсия во всех измерениях. см. LDA.

Конструктор

| param   | type        | descr                                           |
|--------------|-------------------|-------------------------------------------------------------|
| **“metric”**  | string, or callable | Метрика, используемая для расчёта расстояний между объектами в массиве признаков.|
| **“shrink_threshold”**  | float, optional (default = None) | Граница обрезки центроидов для удаления признаков. |

Если метрика - string или callable, то одна из опций должна быть разрешённым метрическим параметром metrics.pairwise.pairwise_distances. 
Центроиды для образцов, соответствующих каждому классу - это точки, из которых суммы расстояний всех образцов класса минимизированы.
В случае метрики Manhattan центроид - это медиана, для всех остальных - среднее значение.

Методы

| method                | params                                 | descr                                                                                                          |
|-----------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **fit**        | (X, y)       | обучение NC модели на данных | 
| **get_params**          | ([deep])            | Получение параметров оценщика                                                                                   |
| **predict**                 | (X) | Классификация по массиву тестовых векторов X                                                                      |
| **score** | (X, y[, sample_weight])                         | Получение средней точности на указанных тестовых данных и ответах                                                               |
| **set_params** | (**params)                        | Установка параметров оценщика    

Пример

```python
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
NearestCentroid(metric='euclidean', shrink_threshold=None)
print(clf.predict([[-0.8, -1]]))
#[1]
```

#### Nearest Shrunken Centroid

У NC есть параметр shrink_threshold, который реализует NSC. Значения каждого признака для каждого центроида разделены 
внутриклассовой дисперсией этого признака. Значения признака после уменьшаются на shrink_threshold. Стоит заметить, что
если значение конкретного признака пересечёт ноль, оно установится им же, т.е этот признак более не влияет на результат 
классификации, это удобно применять для удаления шумных признаков.


## NN - регрессия

Данный метод может быть использован в случаях, когда ярлыки данных представлены непрерывными, а не дискретными значениями.
Ярлык присвоенный запрашиваемой точке определяется из локальной интерполяции - среднего ярлыков ближайших соседей. По аналогии с классификацией, в
scikit-learn представлены два метода:  [KNeighborsRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)
и [RadiusNeighborsRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor)
Конструкторы классов и методы аналогичны таковым в классификации.

Примеры KNeighborsRegressor и RadiusNeighborsRegressor (разница в примерах лишь в выборе метода)

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y) 

print(neigh.predict([[1.5]]))


### Kernel Density Estimation

Оценка плотности переступает грань между обучением без учителя, разработкой признаков и моделированием данных. Некоторые
наиболее популярные и используемые техники вычисления плотности являются смесями моделей: Гауссова смесь [sklearn.mixture.GaussianMixture](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture)
, и NN-based подходы, такие как Ядерная оценка плотности [sklearn.neighbors.KernelDensity](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity)
. Гауссовы смеси лучше обсуждать в контексте кластеризации, поскольку техника очень полезна как схема кластеризации без учителя

Оценка плотности - очень простая концепция, и многи люди уже знакомы с одним распространённым методом оценки плотности - 
гистограммами. Это очень удобный метод визуализации данных, где значения заданы и количество точек данных внутри каждого зна
чения подсчитаны.

![Визуализация примера ядерной оценки плотности](http://scikit-learn.org/stable/_images/sphx_glr_plot_kde_1d_0011.png)

Визуализация примера ядерной оценки плотности

Ядерная оценка плотности использует BallTree/KDTree для эффективных и быстрых запросов. Несмотря на то ,что примеры выше используют
одномерный датасет для простоты, ЯОП можно произвести в любом количестве плоскостей, хотя на практике проклятие размерности
приводит к деградации производительности при больших размерностях.

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_kde_1d_0031.png)

100 точек построены из бимодального(двухвершинного) распределения, и их ЯОП показана для трёх выбранных ядер. Заметно, как
форма ядра может повлиять на гладкость выходного распределения. Сниппет реализации:

```python
from sklearn.neighbors.kde import KernelDensity
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
kde.score_samples(X)
```

В коде было задано гауссовское ядро. Математически, ядро - положительная функция ![](http://latex.codecogs.com/svg.latex?K%28x%3Bh%29),
управляемая параметром ширины ***h***. Учитывая такую форму ядра, формула ЯОП точки ***y*** внутри группы точек ![](http://latex.codecogs.com/svg.latex?x_i%3B%20i%3D1%5Ccdots%20N)
![](http://latex.codecogs.com/svg.latex?%5Crho_K%28y%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20K%28%28y%20-%20x_i%29%20/%20h%29)
Ядро играет роль параметра сглаживания, контролируя баланс между смещением и дисперсией. Большая ширина ведёт к очень сглаженному
(с высоким смещением) распределению плотности. Маленькая, в свою очередь, ведёт к негладкой (с большой дисперсией) плотности
распределения.

ЯОП может быть использована с любой подходящей метрикой расстояний(см. DistanceMetric), но результаты подходяще нормализованы
только для Эвклидовой метрики. Одна полезная метрика - Haversine distance, которая вычисляет угловое расстояние между точками
сферы.

Конструктор класса

| params        | type   | description                                                                                                                 |
|---------------|--------|-----------------------------------------------------------------------------------------------------------------------------|
| **bandwidth**    | float  | пропускная способность (ширина)  ядра                                                                                                |
| **algorithm**     | string | Алгоритм построения дерева, [‘kd_tree’&#124;’ball_tree’&#124;’auto’] auto default                                                     |
| **kernel**        | string | [‘gaussian’&#124;’tophat’&#124;’epanechnikov’&#124; ’exponential’&#124;’linear’&#124;’cosine’]  Default is ‘gaussian’.                               |
| **metric**        | string | Метрика расстояний. Не все метрики  валидны для всех алгоритмов. Нормали зация плотности корректна для  эвклидовой метрики. |
| **atol**          | float  | Желаемая абсолютная толерантность результата. Большая толерантность в общем приведёт  к более быстрому исполнению. Def = 0  |
| **rtol**          | float  | относительная толерантность. Default is 1E-8                                                                                |
| **breadth_first** | bool   | если true (def), то использует метод в ширину.(bfs) Иначе - dfs                                                             |
| **leaf_size**     | int    |                                                                                                                             |
| **metric_params** | dict   | Дополнительные параметры дерева для метрики.                                                                                |

atol: Определяет максимальную приемлимую ошибку решения,как значение измеряемого состояния приближается к нулю. Если
абсолютная ошибка превышает приемлемую, решение сокращает время шага.

Методы

| method                | params                                 | descr                                                                                                          |
|-----------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **fit**        | (X[, y, sample_weight])        | обучение ЯОП модели на данных |
| **get_params**          | ([deep])            | Получение параметров оценщика                                                                                   |
| **sample**                 | ([n_samples, random_state]) | Генерация случайных образцов из модели                                                                      |
| **score** | (X[, y])                         | Расчёт общей логарифмической вероятности по модели                                                               |
| **score_samples** | (X)                        | Расчёт модели плотности на данных    
| **set_params** | (**params)                        | Установка параметров оценщика    


### Служебные классы

#### sklearn.neighbors.radius_neighbors_graph

Получение взвешенного графа соседей с радиусом для точек в X. На выходе имеем разрежённую матрицу **A** в формате CSR с 
shape = [n_samples, n_samples]; A[i,j] присвоен вес грани, которая соединяет **i** c **j**

Конструктор

| param         | type                                                    | desc                                                                                                           |
|---------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **X**             | array-like or BallTree, shape = [n_samples, n_features] | Образцы, numpy arr или готовый BallTree                                                                        |
| **radius**        | float                                                   | радиус соседства                                                                                               |
| **mode**          | {‘connectivity’, ‘distance’}                            | тип возвращаемой матрицы. conn - соедин. матрица с 1/0, dist - расстояния между соседями относительно метрики. |
| **metric**        | string, default ‘minkowski’                             |                                                                                                                |
| **p**             | int, default 2                                          | Сила метрики Минковского. p = 1  = manhattan_distance (l1),  euclidean_distance (l2) для p = 2                 |
| **metric_params** | dict optional                                           | доп. параметры                                                                                                 |
| **include_self**  | bool default=False                                      | Отмечать каждый образец как сосед самому себе.                                                                 |
| **n_jobs**        | int or None, optional (default=None)                    |                                                                                                                |

```python
X = [[0], [3], [1]]
from sklearn.neighbors import radius_neighbors_graph
A = radius_neighbors_graph(X, 1.5, mode='connectivity',
                              include_self=True)
A.toarray()
#array([[1., 0., 1.],
#       [0., 1., 0.],
#       [1., 0., 1.]])
```

#### sklearn.neighbors.kneighbors_graph

Получение взвешенного графа k-соседей для точек из X. На выходе имеем разрежённую матрицу **A** в формате CSR с 
shape = [n_samples, n_samples]; A[i,j] присвоен вес грани, которая соединяет **i** c **j**

Конструктор

| param         | type                                                    | desc                                                                                                           |
|---------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **X**             | array-like or BallTree, shape = [n_samples, n_features] | Образцы, numpy arr или готовый BallTree                                                                        |
| **n_neighbors**   | int                                                     | n соседей                                                                                               |
| **mode**          | {‘connectivity’, ‘distance’}                            | тип возвращаемой матрицы. conn - соедин. матрица с 1/0, dist - расстояния между соседями относительно метрики. |
| **metric**        | string, default ‘minkowski’                             |                                                                                                                |
| **p**             | int, default 2                                          | Сила метрики Минковского. p = 1  = manhattan_distance (l1),  euclidean_distance (l2) для p = 2                 |
| **metric_params** | dict optional                                           | доп. параметры                                                                                                 |
| **include_self**  | bool default=False                                      | Отмечать каждый образец как сосед самому себе.                                                                 |
| **n_jobs**        | int or None, optional (default=None)                    |                                                                                                                |

```python
X = [[0], [3], [1]]
from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(X, 1.5, mode='connectivity',
                              include_self=True)
A.toarray()
#array([[1., 0., 1.],
#       [0., 1., 0.],
#       [1., 0., 1.]])
```
