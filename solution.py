# -*- coding: utf-8 -*-
# @Time    : 18-3-28 下午6:59
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# 读取数据集
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# 选择用于训练的特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x_train = train_data[features]
x_test = test_data[features]

y_train = train_data['Survived']

# 检查缺失值
print '训练数据信息：'
x_train.info()
print '-'*30
print '测试数据信息：'
x_test.info()

# 使用登录最多的港口来填充登录港口的nan值
print '\n\n\n登录港口信息：'
print x_train['Embarked'].value_counts()
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

# 使用平均年龄来填充年龄中的nan值
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

# 使用票价的均值填充票价中的nan值
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

# 将特征值转换成特征向量
dvec = DictVectorizer(sparse=False)

x_train = dvec.fit_transform(x_train.to_dict(orient='record'))
x_test = dvec.transform(x_test.to_dict(orient='record'))

# 打印特征向量格式
print '\n\n\n特征向量格式'
print dvec.feature_names_

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


print '\n\n\n模型验证:'
print 'SVM acc is', np.mean(cross_val_score(svc, x_train, y_train, cv=10))
print 'DecisionTree acc is', np.mean(cross_val_score(dtc, x_train, y_train, cv=10))
print 'RandomForest acc is', np.mean(cross_val_score(rfc, x_train, y_train, cv=10))
print 'LogisticRegression acc is', np.mean(cross_val_score(lr, x_train, y_train, cv=10))
print 'NaiveBayes acc is', np.mean(cross_val_score(nb, x_train, y_train, cv=10))
print 'KNN acc is', np.mean(cross_val_score(knn, x_train, y_train, cv=10))
print 'AdaBoost acc is', np.mean(cross_val_score(boost, x_train, y_train, cv=10))

# 训练
boost.fit(x_train, y_train)
# 预测
y_predict = boost.predict(x_test)
# 保存结果
result = {'PassengerId': test_data['PassengerId'],
          'Survived': y_predict}
result = pd.DataFrame(result)
result.to_csv('submission.csv',index=False)
