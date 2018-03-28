# scikit-learn在Kaggle Titanic数据集上的简单实践（新手向）

博客链接：[scikit-learn在Kaggle Titanic数据集上的简单实践（新手向）](https://blog.csdn.net/aaronjny/article/details/79735998)

Titanic乘客生存预测是Kaggle上的一项入门竞赛，即给定一些乘客的信息，预测该乘客是否在Tatanic灾难中幸存下来。

> 什么是Kaggle？

> 给出百度百科的定义作为参考：Kaggle是由联合创始人、首席执行官安东尼·高德布卢姆（Anthony Goldbloom）2010年在墨尔本创立的，主要为开发商和数据科学家提供举办机器学习竞赛、托管数据库、编写和分享代码的平台。

今天，我们使用scikit-learn框架在Titanic数据集上做一些基础实践。

-------------------------

## 分析数据集

**1.获取数据集**

Titanic数据集分为两部分：

- 训练数据集-包含特征信息和存活与否的标签

- 测试数据集-只包含特征信息

数据集可以从kaggle上下载（点击下面的链接进行下载），格式为csv：

- [train.csv](https://www.kaggle.com/c/3136/download/train.csv)

- [test.csv](https://www.kaggle.com/c/3136/download/test.csv)


**2.分析数据集**

数据集下载到本地后，我们使用pandas读取，查看一下数据集信息：

```python
import pandas as pd

# 读取数据集
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')
# 打印信息
train_data.info()
```

输出信息如下：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```


对上面的特征进行一下解释：

- PassengerId 乘客编号

- Survived 是否幸存

- Pclass 船票等级

- Name 乘客姓名

- Sex 乘客性别

- SibSp、Parch 亲戚数量

- Ticket 船票号码

- Fare 船票价格

- Cabin 船舱

- Embarked 登录港口


这里，根据常识和经验，简单选择一下用于训练的特征：`['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`。当然了，如果使用特征工程来筛选和创造训练特征的话，效果要比这个好得多。

```python
# 选择用于训练的特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x_train = train_data[features]
x_test = test_data[features]

y_train=train_data['Survived']
```

现在，我们查看一下筛选后的数据集：

```python
# 检查缺失值
x_train.info()
print '-'*100
x_test.info()
```

输出的结果如下:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 7 columns):
Pclass      891 non-null int64
Sex         891 non-null object
Age         714 non-null float64
SibSp       891 non-null int64
Parch       891 non-null int64
Fare        891 non-null float64
Embarked    889 non-null object
dtypes: float64(2), int64(3), object(2)
memory usage: 48.8+ KB
------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 7 columns):
Pclass      418 non-null int64
Sex         418 non-null object
Age         332 non-null float64
SibSp       418 non-null int64
Parch       418 non-null int64
Fare        417 non-null float64
Embarked    418 non-null object
dtypes: float64(2), int64(3), object(2)
memory usage: 22.9+ KB
```

可以发现，在这些数据中，是存在缺失值的，比如说训练数据中的`Age`,`Embarked`，测试数据中的`Age`,`Fare`,`Embarked`。我们要将其填充完整。

**3.补全数据集**

`Age`和`Fare`是数值型数据，可以使用其平均值来补全空值，尽量减小补全值对结果的影响。

```python
# 使用平均年龄来填充年龄中的nan值
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(),inplace=True)

# 使用票价的均值填充票价中的nan值
x_test['Fare'].fillna(x_test['Fare'].mean(),inplace=True)
```

`Embarked`是类别数据，取出现次数最多的类别来补全空值。

```python
# 使用登录最多的港口来填充登录港口的nan值
print x_train['Embarked'].value_counts()
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S',inplace=True)
```

能够看到，出现次数最多的类别是‘S’:

```
S    644
C    168
Q     77
```

**4.将特征值转化为特征向量**

想要进行训练，还需要将这些特征值转化成特征向量才行。类别类型的特征，也需要转化成类似于`one-hot`的格式。

```python
# 将特征值转换成特征向量
dvec=DictVectorizer(sparse=False)

x_train=dvec.fit_transform(x_train.to_dict(orient='record'))
x_test=dvec.transform(x_test.to_dict(orient='record'))

# 打印特征向量格式
print dvec.feature_names_
```

能够看到，转换之后的特征向量的格式大致上是这样的：

```
['Age', 'Embarked=C', 'Embarked=Q', 'Embarked=S', 'Fare', 'Parch', 'Pclass', 'Sex=female', 'Sex=male', 'SibSp']
```

比如，我们打印训练数据的第一条：

```
print x_train[0]
```

其输出为：

```
[22.    0.    0.    1.    7.25  0.    3.    0.    1.    1.  ]
```

该特征向量的值与上面的名称是一一对应的。


**5.选择模型**

sklearn里面集成了很多经典模型，本文选了几个进行了测试。这些模型很多都具备可调节的超参数，本文均使用默认配置。

对模型的验证使用十倍交叉验证。

```python
# 支持向量机
svc = SVC()
# 决策树
dtc = DecisionTreeClassifier()
# 随机森林
rfc = RandomForestClassifier()
# 逻辑回归
lr = LogisticRegression()
# 贝叶斯
nb = MultinomialNB()
# K邻近
knn = KNeighborsClassifier()
# AdaBoost
boost = AdaBoostClassifier()

print 'SVM acc is', np.mean(cross_val_score(svc, x_train, y_train, cv=10))
print 'DecisionTree acc is', np.mean(cross_val_score(dtc, x_train, y_train, cv=10))
print 'RandomForest acc is', np.mean(cross_val_score(rfc, x_train, y_train, cv=10))
print 'LogisticRegression acc is', np.mean(cross_val_score(lr, x_train, y_train, cv=10))
print 'NaiveBayes acc is', np.mean(cross_val_score(nb, x_train, y_train, cv=10))
print 'KNN acc is', np.mean(cross_val_score(knn, x_train, y_train, cv=10))
print 'AdaBoost acc is', np.mean(cross_val_score(boost, x_train, y_train, cv=10))
```

十倍交叉验证的平均结果如下：

```
SVM acc is 0.726437407786
DecisionTree acc is 0.777898081943
RandomForest acc is 0.81717483827
LogisticRegression acc is 0.795800987402
NaiveBayes acc is 0.692726705255
KNN acc is 0.708359153331
AdaBoost acc is 0.810419929633
```

可以看出，在当前的特征选择和模型配置下，随机森林、逻辑回归和AdaBoost表现较好，KNN和朴素贝叶斯表现较差。这样的准确率和精心的特征工程和模型调优相比，肯定是差得远的，毕竟大佬们都已经有1.0准确率的了，可怕……不过本文只是做个实践，表现差点倒是无所谓，这里选择使用AdaBoost完成后续内容。

**6.进行预测**

使用AdaBoost分类器来进行生存预测，并保存预测结果。

```
# 训练
boost.fit(x_train, y_train)
# 预测
y_predict = boost.predict(x_test)
# 保存结果
result = {'PassengerId': test_data['PassengerId'],
          'Survived': y_predict}
result = pd.DataFrame(result)
result.to_csv('submission.csv',index=False)
```

在kaggle上提交了一下，准确率只有0.75119。果然，无脑莽是不行的=。=想要模型的效果好，认真的特征工程和模型调优，还是少不了的。。。

emmm,今天的实践就到这里，完整的代码请点击[这里(GitHub)](https://github.com/AaronJny/simple_titanic)