# 构建价格模型 #

利用多种不同属性（比如价格）对数值型数据进行预测时，贝叶斯分类器，决策树，支持向量机都不是最佳算法。

目的：

1. 分类器算法接受训练，根据之前见过的样本数据作出数字类的预测，而且显示出预测的概率分布情况，以帮助用户对预测过程加以解释。
2. 如何利用算法来构造价格预测模型。


解答2缘由：

1. 经济学家认为，价格（尤其是拍卖价格）是一种利用集体智慧来决定商品真实价值的非常好的方法；在一个拥有众多买家和卖家的大型市场中，通常对于交易双方而言，商品的价格最终将会达到一个最优解。
2. 对价格预测也是测试分类器算法的一种很好的手段。

进行数值型预测的一项关键工作是确定哪些变量是重要的，以及如何将它们组合在一起。


## 构造一个样本数据集 ##

情景：

**根据一个人为假设的简单模型来构造一个有关葡萄酒价格的数据集**。

酒的价格是根据**酒的等级**及**其储藏的年代**共同来决定的。

该模型假设葡萄酒有“峰值年（peakage）”的现象，即：较之峰值年而言，年代稍早一些的酒的品质会比较好一些，而紧随其后的则品质稍差些。

一瓶高等级的葡萄酒将从高价位开始，尔后价格逐渐走高直至其“峰值年"，而一瓶低等级的葡萄酒则会从一个低价位开始，价格一路走低。


	# 根据等级，已存储年龄得出酒价
	def wineprice(rating,age):
	  peak_age=rating-50
	  
	  # Calculate price based on rating
	  price=rating/2
	  if age>peak_age:
	    # Past its peak, goes bad in 10 years
	    price=price*(5-(age-peak_age)/2)
	  else:
	    # Increases to 5x original value as it
	    # approaches its peak
	    price=price*(5*((age+1)/peak_age))
	  if price<0: price=0
	  return price

	# 生成（等级，已存储年龄）特征，（酒价）分类数据集
	def wineset1():
	  rows=[]
	  for i in range(300):
	    # Create a random age and rating
	    rating=random()*50+50
	    age=random()*50
	
	    # Get reference price
	    price=wineprice(rating,age)
	    
	    # Add some noise # 模拟税收，价格局部变动
	    price*=(random()*0.2+0.9)
	
	    # Add to the dataset
	    rows.append({'input':(rating,age),
	                 'result':price})
	  return rows

运行代码


	>>> 
	 RESTART: C:\Users\Administrator.USER-20180302VA\Desktop\Lab\ProgrammingCollectiveIntelligence\C08\numpredict.py 
	>>> wineprice(95.0, 3.0)
	21.111111111111114
	>>> wineprice(95.0, 8.0)
	47.5
	>>> wineprice(99.0, 1.0)
	10.102040816326529
	>>> data=wineset1()
	>>> data[0]
	{'input': (75.49955179321756, 23.42460727367927), 'result': 184.60154078303577}
	>>> data[1]
	{'input': (90.98280059701668, 0.051264242473414434), 'result': 5.570848286414397}
	>>> 


## k-最近邻分配算法 ##

对于葡萄酒定价问题而言，最简单的做法与人们尝试手工进行定价时所采用的做法是一样的一一即，找到几瓶情况最为相近的酒，并假设其价格大体相同。

使用k-最近邻算法（k-nearest neighbors，kNN）通过寻找与当前所关注的商品情况相似的一组商品，对这些商品的价格求均值，进而作出价格预测。

### 近邻数 ###

kNN中的k代表的，是为了求得最终结果而参与求平均运算的商品数量。

k的值根据实际情况选择，不能过大或过小。

### 定义相似度 ###

选用**欧几里得距离算法**衡量两件商品之间相似程度的方法。

	def euclidean(v1,v2):
	  d=0.0
	  for i in range(len(v1)):
	    d+=(v1[i]-v2[i])**2
	  return math.sqrt(d)

运行代码

	>>> data[0]['input']
	(75.49955179321756, 23.42460727367927)
	>>> data[1]['input']
	(90.98280059701668, 0.051264242473414434)
	>>> euclidean(data[0]['input'], data[1]['input'])
	28.036479058090837
	>>> 

### kNN代码 ###

kNN是一种实现起来相对简单的算法。虽然这种算法的计算量很大（computationally intensive），但是其优点在于每次有新数据加入时，都无需重新训练。

	# 计算给定商品与原数据集中任一其他商品间的距离
	def getdistances(data,vec1):
	  distancelist=[]
	  
	  # Loop over every item in the dataset
	  for i in range(len(data)):
	    vec2=data[i]['input']
	    
	    # Add the distance and the index
	    distancelist.append((euclidean(vec1,vec2),i))
	  
	  # Sort by distance
	  distancelist.sort()
	  return distancelist

	# 利用其中前k项结果求出平均值
	def knnestimate(data,vec1,k=5):
	  # Get sorted distances
	  dlist=getdistances(data,vec1)
	  avg=0.0
	  
	  # Take the average of the top k results
	  for i in range(k):
	    idx=dlist[i][1]
	    avg+=data[idx]['result']
	  avg=avg/k
	  return avg

运行代码

	>>> knnestimate(data,(95.0, 3.0))
	17.225810699279158
	>>> knnestimate(data,(99.0, 3.0))
	22.031353661602076
	>>> knnestimate(data,(95.0, 5.0))
	29.10861771952609
	>>> wineprice(99.0, 5.0)
	30.306122448979593
	>>> knnestimate(data,(95.0, 3.0), k=1)
	22.54614350103098
	>>> 

## 为近邻分配权重 ##

### 反函数 ###

上述算法有可能会选择距离太远的近邻，因此，一种补偿的办法是根据距离的远近为其赋予相应的权重。

![](image/01.png)

	def inverseweight(dist,num=1.0,const=0.1):
	  return num/(dist+const)

该函数实现简单，但是最为主要的潜在缺陷在于，它会为近邻项赋予很大的权重，而稍远的一项，其权重“衰减”得很快。

### 减法函数 ###

![](image/02.png)


	def subtractweight(dist,const=1.0):
	  if dist>const: 
	    return 0
	  else: 
	    return const-dist

该函数克服了前述对近邻权重分配过大的潜在问题，但是权重可能为0，因此我们有可能找不到距离足够近的项，将其视为近邻，即对于某些项，算法根本就无法做出预测。

### 高斯函数 ###

该方法克服前述方法的局限。就是有点复杂，算起来较慢

![](image/03.png)

	def gaussian(dist,sigma=5.0):
	  return math.e**(-dist**2/(2*sigma**2))

![](image/04.png)

运行代码

	>>> subtractweight(0.1)
	0.9
	>>> inverseweight(0.1)
	5.0
	>>> gaussian(0.1)
	0.9998000199986667
	>>> gaussian(1.0)
	0.9801986733067553
	>>> subtractweight(1.0)
	0.0
	>>> inverseweight(1.0)
	0.9090909090909091
	>>> gaussian(3.0)
	0.835270211411272
	>>> 

PS. 运行结果与书本有出入，公式前后并不一致。

### 加权kNN ###

	def weightedknn(data,vec1,k=5,weightf=gaussian):
	  # Get distances
	  dlist=getdistances(data,vec1)
	  avg=0.0
	  totalweight=0.0
	  
	  # Get weighted average
	  for i in range(k):
	    dist=dlist[i][0]
	    idx=dlist[i][1]
	    weight=weightf(dist)
	    avg+=weight*data[idx]['result']
	    totalweight+=weight

	  if totalweight==0: return 0
	  avg=avg/totalweight
	  return avg

运行代码

	>>> weightedknn(data,(99.0, 5.0))
	27.992611805843243
	>>> 

## 交叉验证 ##

交叉验证是将数据拆分成训练集与测试集的一系列技术的统称。

将训练集传入算法随着正确答案的得出(在本章的例子中即为价格)，就得到了一组用以进行预测的数据集。

随后，要求算法对测试集中的每一项数据都作出预测。其所给出的答案，将与正确答案进行对比，算法会计算出一个整体分值，以评估其所做预测的准确程度

---

数据集分成两部分，5%数据用于测试，95%用于训练。

	def dividedata(data,test=0.05):
	  trainset=[]
	  testset=[]
	  for row in data:
	    if random()<test:
	      testset.append(row)
	    else:
	      trainset.append(row)
	  return trainset,testset

---

	def testalgorithm(algf,trainset,testset):
	  error=0.0
	  for row in testset:
	    guess=algf(trainset,row['input'])
	    error+=(row['result']-guess)**2
	    #print row['result'],guess
	  #print error/len(testset)
	  return error/len(testset)

testalgorithm的a1gf接受一个算法函数作为参数，而该算法函数则接受一个数据集和个査询项作为参数。 

testalgorithm会循环遍历测试集中的每一行，并利用a1gf得出最佳的猜测结果。随后，它会从实际结果中减去猜测所得的结果。

对数字求平方是一种常见的做法，因为它会突显较大的差值。这意味着，一个在大多数时候都非常接近于正确值，但是偶尔会有较大偏离的算法，要比始终都比较接近于正确值的算法稍逊一些。一般而言，这种情况是我们所期望的，不过有时也有例外，那就是：如果算法在余下的大多数时候准确度都非常的高，那么偶尔犯一个大错误还是可以接受的。如果是这种情况，那么我们可以对函数稍做修改，只要将差值的绝对值累加起来即可。

---

	def crossvalidate(algf,data,trials=100,test=0.1):
	  error=0.0
	  for i in range(trials):
	    trainset,testset=dividedata(data,test)
	    error+=testalgorithm(algf,trainset,testset)
	  return error/trials

该函数对数据采取若干不同的划分，并在每个划分上执行testalgorithm函数，然后再将所有的结果都累加起来，以求得最终的评分值。

---

运行代码

	>>> data=wineset1()
	>>> crossvalidate(knnestimate, data)# kNN默认选5个近邻
	298.7994273869
	>>> def knn3(d,v):return knnestimate(d, v, k=3)
	
	>>> crossvalidate(knn3, data)
	252.18174022304578
	>>> def knn1(d,v):return knnestimate(d, v, k=1)
	
	>>> crossvalidate(knn1, data)
	352.0428832264139
	>>> 

从上看出，选择太少近邻或太多近邻导致效果不彰。k值为5比1或3的好。

	>>> crossvalidate(weightedknn,data)
	280.68594920470093
	>>> def knninverse(d, v):return weightedknn(d, v, weightf=inverseweight)
	
	>>> crossvalidate(knninverse,data)
	296.12994220636386
	>>> 

至于选择哪个，自己执生。


## 不同类型的变量 ##

## 对缩放结果进行优化 ##

## 不对称分布 ##

## 使用真实数据——eBayAPI ##

## 何时使用k-最邻近算法 ##

## 小结 ##

