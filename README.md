# Topic-Modeling-PyCharm-Issues
Репозиторий с проектом к домашнему заданию по курсу "Истории из прода", JetBrains, YouTrack.

#### Data overview
 В задании необходимо проанализировать содержания обращений пользователей в YouTrack, касающиеся 
версий 2020.2 и 2020.3. Соберем 2 датасета, с суммаризированными данными, относящимися к этим релизам.


 **Версия** | Количество обращений | 
---|---| 
 2020.2 | 625
2020.3 | 522

Проведем лемматизацию собранных для текста суммаризаций, отбросим наиболее 
очевидные частотные слова, знаки препинания и посмотрим на топ из самых часто-встречающихся слов в запросах:

Для версии 2020.2:

<img src = images/top_10_02.png width="500" height="330">

Для версии 2020.3:

<img src = images/top_10_03.png width="500" height="330">


#### Topic modeling

Проведем тематическое моделирование с помощью `LDA` модели из
библиотеки `gensim`. Для выбора количества тем, воспользуемся
метрикой `c_npmi`. 

Код для обучения моедли, выбора оптимального количества 
тем и обнаружения ключевых слов для темы 
представлен в файле [model.py](model.py). На пример запуска и 
ключевые слова можно посмотреть в [ноутбуке](notebook.ipynb).

По результатам выбора оптимального числа тем
оказалось, что оптимальным является разделение `issues`
на 7 тем.

Темы, чаще всего упоминающиеся в issues относительно
каждого из релизов, а также их пересечение представлено
на диаграмме:

<img src = images/topic_modeling_res.png width="500" height="330">

Таким образом, отвечая на поставленный в задаче вопрос:

   - После выхода релиза 2020.3 пользователи стали меньше жаловаться на `jupyter` и `django`, 
     `инспекции` и `автодополнение кода`.
   - С выходом релиза 2020.3 появилось больше жалоб относительно `тестов`, `атрибутов классов`, 
     `импортирования библиотек` и `работы с кансолью`.

Если сравнивать тематическое моделирование с простым подсчетом самых частотных слов в обращениях, кажется, что
1 оказалось более информативно (в данном случае) просто за счет получения большего количества слов.
Сами слова в выделенных темах, не то чтобы очень логично связаны, встречается много повторяющихся. 
Возможно, примерно к тем же выводам можно было бы прийти, 
подобрав более удачный трешхолд по количеству наиболее популярных слов в запросах.
