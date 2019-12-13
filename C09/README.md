# 高阶分类-核方法与SVM #

本章例子是关于如何为约会网站的用户寻找配对。

给定两人的信息，我们能否预测出他们将会成为一对好朋友呢？

这问题涉及许多变量，既有数值型的，也有名词性的，还有大量的非线性关系。

重要事实：将一个复杂数据集扔给一个算法，然后希望它能够学会如何进行精确分类，这几乎是不可能的。

选择正确的算法，然后对数据进行适当地预处理，这是要获得满意的分类结果所需的。

## 婚介数据集 ##

字段信息：

1. 年龄
2. 是否吸烟？
3. 是否要孩子？
4. 兴趣列表
5. 家庭住址

---

数据文件

- [agesonly.csv](agesonly.csv)


	24,30,1
	30,40,1
	22,49,0
	43,39,1
	23,30,1
	...


- [matchmaker.csv](matchmaker.csv)


	39,yes,no,skiing:knitting:dancing,220 W 42nd St New York NY,43,no,yes,soccer:reading:scrabble,824 3rd Ave New York NY,0
	23,no,no,football:fashion,102 1st Ave New York NY,30,no,no,snowboarding:knitting:computers:shopping:tv:travel,151 W 34th St New York NY,1
	50,no,no,fashion:opera:tv:travel,686 Avenue of the Americas New York NY,49,yes,yes,soccer:fashion:photography:computers:camping:movies:tv,824 3rd Ave New York NY,0
	...

上面每一行数据都对应于一对男女的信息，最后一列表示似乎配对成功

加载数据集函数

	class matchrow:
	  def __init__(self,row,allnum=False):
	    if allnum:
	      self.data=[float(row[i]) for i in range(len(row)-1)]
	    else:
	      self.data=row[0:len(row)-1]
	    self.match=int(row[len(row)-1])
	
	def loadmatch(f,allnum=False):
	  rows=[]
	  for line in file(f):
	    rows.append(matchrow(line.split(','),allnum))
	  return rows

运行代码

	>>> 
	 RESTART: C:\Users\Administrator.USER-20180302VA\Desktop\Lab\ProgrammingCollectiveIntelligence\C09\advancedclassify.py 
	>>> agesonly=loadmatch("agesonly.csv",allnum=True)
	>>> matchmaker=loadmatch('matchmaker.csv')
	>>> 

## 数据中的难点 ##

上述数据集有两个值得注意的地方：

1. 变量的相互作用
2. 非线性的特点

生成一个涉及男女年龄对比情况的散布图，配对的.，否则为+。

	from pylab import *
	def plotagematches(rows):
	  xdm,ydm=[r.data[0] for r in rows if r.match==1],\
	          [r.data[1] for r in rows if r.match==1]
	  xdn,ydn=[r.data[0] for r in rows if r.match==0],\
	          [r.data[1] for r in rows if r.match==0] 
	  
	  plot(xdm,ydm,'bo')
	  plot(xdn,ydn,'b+')
	  
	  show()

![](image/01.png)

尽管很显然还有许多其他因素会对两个人是否成功匹配构成影响，但上图却是根据简化了的只包含年龄信息的数据集绘制而成的。

观图可得：

1. 它还给出了一条明显的边界，表明人们不会去寻找远远超出其年龄范围内的人进行配对。

2. 图上的边界看上去似乎还有些曲折，并且年龄越大边界就越不清晰，这表明人们的年龄越见长就越能忍受更大的年龄差距。

### 决策树分类器 ###

尝试使用决策树分类器对数据进行分类。

决策树算法是根据数值边界来对数据进行划分的。

借助带有两个变量函数精确表达分界线dividing line时，问题也就随之而来。


两人的年龄差作为变量进行预测。得到：

![](image/02.png)

上述结果对于解释决策的过程显然**没有任何用处**。

这棵决策树也许对自动分类会有帮助，但是这样做**太麻烦也太死板**了。

假如考虑除年龄之外的其他变量，结果甚至有可能会变得令人更加难以理解。

>PS.决策树对数值分类的缺陷

为了明白决策树到底做了些什么，看一下散布图，以及根据决策树生成的决策边界。

![](image/03.png)

决策边界是这样一条线：位于这条线一侧的毎一个点会被赋予某个分类，而位于另一侧的每一个点会被赋予另一个分类。从图中可看到，决策树的约束条件使边界线呈现出垂直或水平向的分布。

此处有两个要点：

1. 在没有弄清楚数据本身的含义及如何将其转换成更易于理解的形式之前，轻率地使用提供给数据是错误的。**建立散布图有助于找到数据真正的划分方式**。
2. 尽管决策树有其自身的优势，但是在确定向题的分类时，**如果存在多个数值型输入，且这些输入彼此间所呈现的关系并不简单，决策树则常常不是最有效的方法**。

## 基本的线性分类 ##

**线性分类的工作原理**是寻找每个分类中所有数据的平均值，并构造一个代表该分类中心位置的点，然后即可通过判断距离哪个中心点位置最近来对新的坐标点进行分类。

实现一个函数计算分类的均值点average point


	def lineartrain(rows):
	  averages={}
	  counts={}
	  
	  for row in rows:
	    # Get the class of this point # 配对结果
	    cl=row.match
	    
	    averages.setdefault(cl,[0.0]*(len(row.data)))
	    counts.setdefault(cl,0)
	    
	    # Add this point to the averages
	    for i in range(len(row.data)):
	      averages[cl][i]+=float(row.data[i])
	      
	    # Keep track of how many points in each class
	    counts[cl]+=1
	    
	  # Divide sums by counts to get the averages
	  for cl,avg in averages.items():
	    for i in range(len(avg)):
	      avg[i]/=counts[cl]
	  
	  return averages

运行代码

	>>> agesonly=loadmatch("agesonly.csv",allnum=True)
	>>> matchmaker=loadmatch('matchmaker.csv')
	>>> lineartrain(agesonly)
	{0: [26.914529914529915, 35.888888888888886], 1: [35.48041775456919, 33.01566579634465]}
	>>> 

年龄数据的分布图有助于理解线性分类的作用

![](image/04.png)

图中的X表示由lineartrain计算求得的均值点。划分数据的直线位于两个X的中间位置。

这意味着，所有位于直线左侧的坐标点都更接近于表示“不相匹配(no match)”的均值点，而所有位于右侧的坐标点则都更接近于表示“相匹配(match)”的均值点。

任何时候当我们遇到一对新的年龄数据时，如果想要推测二者是否相匹配，只须将其想象成上图中的一个坐标点，并判断其更接近于哪个均值点即可。

判定一个坐标点距离均值点的远近程度：

1. 利用欧几里得距离公式，先计算坐标点到每个分类的均值点的距离，然后从中选择距离较短者
2. 向量和点积

---

**向量**

![](image/05.png)

**点积**指两个向量将第一个向量中的每个值与第二个向量中的对应值相乘，然后再将所得的每个乘积相加，最后得到一个结果。

	# 计算点积
	def dotproduct(v1,v2):
	  return sum([v1[i]*v2[i] for i in range(len(v1))])

点积也可以利用两个向量的长度乘积，再乘以两者夹角的余弦求得。注意，角度大于90时，余弦值为负数，点积也为负数。

![](image/06.png)

在图中，看到有两个均值点，分别对应于“相匹配”(M0)和“不相匹配”(M1)两种情况，以及一个位置介于M0与M1中间的C。另外还有两个点，X0和X1，它们是即将要被分类的两个例子。除此以外，图上还显示了连接M0到M1的向量，以及连接X1到C和X2到C的两个向量。

在图中，X1更接近于M，因此它应该被划归为“相匹配”。我们注意到，介于向量X1→C和M0→M1的夹角为45度，小于90度，因此X1→C与M→M1的点积结果为正数。

而由于向量X2→C和M0→M1的指向相反，因此介于两者间的夹角大于90度。即X2→C和M0→M1的点积结果为负数

夹角大者点积为负，夹角小者点积为正，因此只须通过观察点积结果的正负号，就可以判断出新的坐标点属于哪个分类。

寻找分类公式

![](image/07.png)

![](image/08.png)

	# 计算公式
	def dpclassify(point,avgs):
	  b=(dotproduct(avgs[1],avgs[1])-dotproduct(avgs[0],avgs[0]))/2
	  y=dotproduct(point,avgs[0])-dotproduct(point,avgs[1])+b
	
	  return 0 if y > 0 else 1

运行代码

	>>> agesonly=loadmatch("agesonly.csv",allnum=True)
	>>> avgs=lineartrain(agesonly)
	>>> dpclassify([30,30],avgs)
	1
	>>> dpclassify([30,25],avgs)
	1
	>>> dpclassify([25,40],avgs)
	0
	>>> dpclassify([48,20],avgs)
	1
	>>> 

请记住这是一个线性分类器，所以它只找出了一条分界线来。

这意味着，如果找不到一条划分数据的直线来，或者如果实际存在多条直线时，就如同前面那个年龄对比的例子中所呈现的那样，那么此时分类器将会得到错误的答案。

在那个例子中，48岁与20岁的年龄对比实际上应该是“不相匹配”的结果，但是因为我们只找到了一条直线，而相应的坐标点又落在了这条直线的右侧，所以函数得出了“相匹配”的结论。

## 分类特征 ##

## 对数据进行缩放处理 ##

## 理解核方法 ##

## 支持向量机 ##

## 使用LIBSVM ##

## 基于Facebook的匹配 ##