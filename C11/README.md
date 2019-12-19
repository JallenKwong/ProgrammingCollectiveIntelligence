# 智能进化 #

在先前的章节中，在面对每一个问题时，都采用了一种最适合与解决该问题的算法。

在某些例子，还需对参数进行调整，或者借助于优化手段来找出一个满意的参数集来。

**目的**：编写一个程序，尝试自动构造出解决某一问题的最佳程序来。因而本质上看，将构造出**一个能构造算法的算法**。

本章将采用一种称为**遗传编程genetic programming**

本章与其他章节不同，不再使用API或公共数据集，同时因为，能够根据与大量人群的交互对自身作出修改的程序，是一种极为有趣且类型迥异的集体智慧。


本章例子：

1. 根据给定的数据集重新构造一个数学函数
2. 在一个简单的棋类游戏中自动生成一个AI玩家。

计算能力computational power才是真正制约遗传编程问题解决能力的**唯一因素**。


## 什么是遗传编程 ##

**遗传编程**是受到生物进化理论的启发而形成的一种机器学习技术。

通常工作方式是：

1. 以一大堆程序（**种群**）开始——这些程序可以随机产生的，也可以是人为设计的hand-designed，并且它们被认为是在某种程度上的一组优解good solutions；
2. 随后，这些程序将会在一个由**用户定义的任务user-defined task**中展开竞争。<br>此处所谓的任务或许是一种竞赛game，各个程序在竞赛中彼此直接展开竞争，或者也有可能是一种个体测试，其目的是要测出哪个程序的执行效果更好。
3. 待竟争结東之后，会得到一个针对所有程序的评价列表，该列表按程序的表现成绩从最好到最差顺次排列。

接下来，也正是进化得以体现的地方，算法可以采取**两种**不同的方式对表现最好的程序实施复制和修改：

1. 比较简单的一种方式称为**变异**(mutation)，算法会对程序的某些部分以随机的方式稍作修改，希望借此能够产生一个更好的题解来。
2. 另一种修改程序的方式称为**交叉**(crossover)，有时也称为配对(breeding)，其做法是：先将某个最优程序的一部分去掉，然后再选择其他最优程序的某一部分来替代之。

这样的复制和修改过程会产生出许多新的程序来，这些程序基于原来的最优程序，但又不同于它们。

---

在每一个复制和修改的阶段，算法都会借助于一个适当的函数对程序的质量作出**评估**。

因为种群的大小始终是保持不变的，所以许多表现极差的程序都会从种群中被剔除出去，从而为新的程序腾出空间。

新的种群被称为“下一代”，而整个过程则会一直不断地重复下去。因为最优秀的程序一直被保留了下来，而且算法又是在此基础上进行复制和修改的，所以我们有理由相信，每一代的表现都会比前一代更加出色；这在很大程度上有点类似于人类世界中，年轻一代比他们的父辈更聪明。

---

创建新一代的过程直到**终止条件**满足才会结束，具体问题的不同，可能的终止条件也不同。

- 找到了最优解。
- 找到了表现足够好的解。
- 题解在历经数代之后都没有得到任何改善。
- 繁衍的代数达到了规定的限制。

对于某些问题而言，

- 比如确定一个数学函数，令其将一组输入正确地映射到某个输出一要找到最优解是有可能的。但是对于其他向题，
- 比如棋类游戏，也许根本就不存在最优解，这是因为题解的质量依赖于程序的对抗者所采取的策略。

![遗传编程的一个大体过程](image/01.png)

### 遗传编程与遗传算法 ###

**遗传算法genetic algorithms**是一种**优化技术**，它汲取了生物进化中优胜劣汰的思想。

就优化技术而言，不论是何种形式的优化，算法或度量都是事先选择好了的，而我们所要做的工作只是**尝试为其找到最佳参数**。

和优化算法一样，遗传编程也需要一种方法来度量题解的优劣程度；但与优化算法**不同**的是，此处的题解并**不仅仅是一组用于给定算法的参数而已**。

相反，在遗传编程中，连同算法本身及其所有的参数，都是按照优胜劣汰的进化规律( evolutionary pressure)自动设计得到的。

### 遗传编程的成功之处 ###

遺传编程自20世纪80年代以来就一直存在着，但是它的**计算量非常庞大**，而且以那个时候可以获得的计算能力而言，人们是不可能将其用于稍复杂一些的问题的。然而，随着计算机的执行速度越来越快，人们已经逐渐能够将遺传编程应用到复杂问题上了。正因为如此、许多以前的专利发明，借助遺传编程得到了再次挖掘和进一步的改善，而近年来也有不少可以获得专利的新发明，都是借助计算机利用遗传编程设计出来的。

人们已经将遺传编程技术广泛应用于许多领域：

1. NASA的天线设计
2. 光子晶体领域光学领域
3. 量子计算系统，
4. 应用于许多竟技类游戏程序的开发上，比如:国际象棋和西洋双陆棋。<br>1998年，来自卡耐基梅隆大学的研究者率领一支机器人队伍间入了Robo Cup机器人世界杯赛，并且在众多参赛者中排名居中，这支队伍就是完全利用遗传编桯技术打造的。
5. ...

## 将程序以树形方式表示 ##

为了构造出能够用以测试、变异和配对的程序，我们需要一种方法能够在 Python代码中描述和运行这些程序。这种描述方法自身必须是易于修改的，而且更重要的一点是，它必须保证所描述的是一个实实在在的程序这意味着，**试图将随机生成的字符串作为 Python代码的做法是行不通的**。

为了描述遗传编程中的程序，应用最为普遍的是**树形表示法**。

大多数编程语言，在编译或解释时，首先都会被转换成一棵解析树，这棵树非常类似于此处我们将要用到的树。(Lisp編程语言及其变体，本质上就是一种直接访问解析树的方法)。

![](image/02.png)

树上的节点有可能是**枝节点**，代表了应用于其子节点之上的**某一项操作**，也有可能是**叶节点**，比如一个带常量值的参数。

例如:

1. 图上的圆形节点代表了应用于两个分支(本例中为Y值和5)之上的求和操作。一旦求出了此处的计算值，就会将计算结果赋予上方的节点处，相应地，这一计算过程会一直向下传播。
2. 树上有一个节点的操作为“if"，这表示:如果该节点左侧分支的计算结果为真，则它将返回中间的分支，如果不为真，则返回右侧的分支。

对整棵树进行遍历，你就会发现它相当于下面这个Python函数:

	def func(x,y):
		if x > 3:
			return y + 5
		else:
			return y - 2

构造树还有两点需要考虑的是：

1. 构成这棵树的节点可以是非常复杂的函数，如：
	- 距离度量
	- 高斯分布
2. 通过引用树上位置相对较高的节点，可以用**递归**的方式构造树。
 
采用这样的方式来构造树可以实现循环及其它更为复杂的控制结构。

### 在Python中表现树 ###

	from random import random,randint,choice
	from copy import deepcopy
	from math import log
	
	# 一个封装类，对应于“函数型”节点上的函数。
	class fwrapper:
	  def __init__(self,function,childcount,name):
		# 函数本身
	    self.function=function
		# 函数接受的参数的个数
	    self.childcount=childcount
		# 函数名称
	    self.name=name
	
	# 对应于函数型节点（即带子节点的节点）。以一个fwrapper类对其进行初始化。当evaluate被调用时，会对各个子结点进行求值运算，然后再将函数本身应用于求得结果。
	class node:
	  def __init__(self,fw,children):
	    self.function=fw.function
	    self.name=fw.name
	    self.children=children
	
	  def evaluate(self,inp):    
	    results=[n.evaluate(inp) for n in self.children]
	    return self.function(results)
	  def display(self,indent=0):
	    print (' '*indent)+self.name
	    for c in self.children:
	      c.display(indent+1)
	    
	
	# 这个类对应的节点只返回传递给程序的某个参数。其evaluate方法返回的是由idx指定的参数。
	class paramnode:
	  def __init__(self,idx):
	    self.idx=idx
	
	  def evaluate(self,inp):
	    return inp[self.idx]
	  def display(self,indent=0):
	    print '%sp%d' % (' '*indent,self.idx)
	    
	# 返回常量值的节点。其evaluate方法仅返回的是该类被初始化时所传入的值。
	class constnode:
	  def __init__(self,v):
	    self.v=v
	  def evaluate(self,inp):
	    return self.v
	  def display(self,indent=0):
	    print '%s%d' % (' '*indent,self.v)
	    
利用fwrapper构造一组操作函数

	addw=fwrapper(lambda l:l[0]+l[1],2,'add')
	subw=fwrapper(lambda l:l[0]-l[1],2,'subtract') 
	mulw=fwrapper(lambda l:l[0]*l[1],2,'multiply')
	
	def iffunc(l):
	  if l[0]>0: return l[1]
	  else: return l[2]
	ifw=fwrapper(iffunc,3,'if')
	
	def isgreater(l):
	  if l[0]>l[1]: return 1
	  else: return 0
	gtw=fwrapper(isgreater,2,'isgreater')
	
	flist=[addw,mulw,ifw,gtw,subw]

### 树的构造和评估 ###

根据下图构造出程序树

![](image/02.png)


	def exampletree():
	  return node(ifw,[
	                  node(gtw,[paramnode(0),constnode(3)]),
	                  node(addw,[paramnode(1),constnode(5)]),
	                  node(subw,[paramnode(1),constnode(2)]),
	                  ]
	              )

运行代码

	>>> 
	 RESTART: C:\Users\Administrator.USER-20180302VA\Desktop\Lab\ProgrammingCollectiveIntelligence\C11\gp.py 
	>>> exampletree=exampletree()
	>>> exampletree.evaluate([2,3])
	1
	>>> exampletree.evaluate([5,3])
	8
	>>> exampletree.evaluate([5,3,2])
	8
	>>> exampletree.evaluate([5])
	
	Traceback (most recent call last):
	  File "<pyshell#4>", line 1, in <module>
	    exampletree.evaluate([5])
	  File "C:\Users\Administrator.USER-20180302VA\Desktop\Lab\ProgrammingCollectiveIntelligence\C11\gp.py", line 18, in evaluate
	    results=[n.evaluate(inp) for n in self.children]
	  File "C:\Users\Administrator.USER-20180302VA\Desktop\Lab\ProgrammingCollectiveIntelligence\C11\gp.py", line 18, in evaluate
	    results=[n.evaluate(inp) for n in self.children]
	  File "C:\Users\Administrator.USER-20180302VA\Desktop\Lab\ProgrammingCollectiveIntelligence\C11\gp.py", line 31, in evaluate
	    return inp[self.idx]
	IndexError: list index out of range
	>>> 

### 程序的展现 ###

	>>> exampletree.display()
	if
	 isgreater
	  p0
	  3
	 add
	  p1
	  5
	 subtract
	  p1
	  2
	>>> 

## 构造初始种群 ##

尽管为遗传编程手工构造程序是可行的，但是通常的初始种群都是由一组随机程序构成的。

这样做可以使我们的起点变得更低，因为我们**没有必要去设计一组几乎已经将问题完全解决了的程序**。

而且，这样做还可以在初始种群中引入更加丰富的多样性一一由某位编程人员为了解决特定问题而专门设计的一组程序，彼此间很可能会非常相似，尽管这些程序也许会给出几乎正确的答案，但是最终的理想题解很有可能会截然不同。

---

创建一个随机程序的步骤包括：

1. 创建根结点并为其随机指定一个关联函数，
2. 然后再随机创建尽可能多的子节点;
3. 相应地，这些子节点也可能会有它们自己的随机关联子节点。
<br>和大多数对树进行操作的函数一样，这一过程很容易以递归的形式进行定义。


	# pc给出了程序树所输入参数的个数
	# fpr给出了新建节点数据函数型节点的概率
	# ppr给出了当新建节点不是函数型节点时，其属于paramnode节点的概率
	# maxdepth 最大深度，防止分支不断地生长
	def makerandomtree(pc,maxdepth=4,fpr=0.5,ppr=0.6):
	  if random()<fpr and maxdepth>0:
		# 随机选择一个函数列表中的函数
	    f=choice(flist)
	    children=[makerandomtree(pc,maxdepth-1,fpr,ppr) 
	              for i in range(f.childcount)]
	    return node(f,children)
	  elif random()<ppr:
	    return paramnode(randint(0,pc-1))
	  else:
	    return constnode(randint(0,10))

运行代码

	>>> random1=makerandomtree(2)
	>>> random1.evaluate([7,1])
	14
	>>> random1.evaluate([2,4])
	4
	>>> random2=makerandomtree(2)
	>>> random2.evaluate([5,3])
	0
	>>> random2.evaluate([5,20])
	0
	>>> 

如果一个程序的所有叶节点都是常量，则该程序实际上根本不会接受任何形式的输入参数，因此无论传给它什么样的输入，其结果都一样。

呈现树

	>>> random1.display()
	multiply
	 p0
	 if
	  if
	   isgreater
	    p0
	    p1
	   p0
	   p1
	  2
	  p0
	>>> random2.display()
	isgreater
	 1
	 4
	>>> 

## 测试题解 ##

我们寻找测试题解正确与否的方法，若题解不正确，还可以确知它与正确答案的差距。

### 一个简单的数学测试 ###

尝试重新构造一个简单的数学函数。

![](image/03.png)

的确存在一些函数可以将X与Y映射到上述输出结果一栏，但是没有人告诉我们这个函数到底是什么。

有时，我们需要的仅仅是一个公式而已。

	def hiddenfunction(x,y):
	    return x**2+2*y+3*x+5

构造一个数据集。借助得到的数据集，可以开始对生成的程序进行测试。

	def buildhiddenset():
	  rows=[]
	  for i in range(200):
	    x=randint(0,40)
	    y=randint(0,40)
	    rows.append([x,y,hiddenfunction(x,y)])
	  return rows

运行代码

	>>> hiddenset=buildhiddenset()
	>>> 

我们要真正测试的，是遗传编程是否能够在不知情的前提下重新构造出这一函数来。

### 衡量程序的好坏 ###

	# 看这个程序与代表正确的数据集之间的接近程度
	def scorefunction(tree,s):
	  dif=0
	  for data in s:
	    v=tree.evaluate([data[0],data[1]])
	    dif+=abs(v-data[2])
	  return dif

运行代码

	>>> scorefunction(random1,hiddenset)
	128870
	>>> scorefunction(random2,hiddenset)
	137150
	>>> 

随机函数程序得到值与正确值的差值的绝对值累加得到的值越小，题解表现越好。累加值为0则表示该程序得到的每一项结果都是正确的。

## 对程序进行变异 ##

当表现最好的程序被选定之后，它们就会被复制并修改以进入到下一代。

一个树状程序可以有多种修改方式，如

- 改变节点上的函数，也可以改变节点的分支；

![](image/04.png)

- 利用一棵全新的树来替换某一子树

![](image/05.png)

**变异采用的次数不宜过多**。

例如，不宜对整棵树上的大多数节点都实施变异。相反，可以为任何须要进行修改的节点定义一个相对较小的概率。

从树的根节点开始，如果每次生成的随机数小于该概率值，就以如上所述的某种方式对节点进行变异，否则，就再次对子节点进行调用自身。

为了简单起见，此处给出的代码只实现了第二种变异方式。

	def mutate(t,pc,probchange=0.1):
	  if random()<probchange:
	    return makerandomtree(pc)
	  else:
	    result=deepcopy(t)
	    if hasattr(t,"children"):
	      result.children=[mutate(c,pc,probchange) for c in t.children]
	    return result

运行代码

	>>> muttree=mutate(random1,2)
	>>> muttree.display()
	multiply
	 p0
	 if
	  if
	   isgreater
	    add
	     p1
	     p1
	    p1
	   p0
	   p1
	  2
	  p0
	>>> 

测试程序

	>>> scorefunction(random1,hiddenset)
	128870
	>>> scorefunction(muttree,hiddenset)
	126742
	>>> 

请记住，变异是随机进行的，而且不必非得朝着有利于改善题解的方向进行。

我们只是希望其中的一部分变异能够对最终的结果有所改善。

这种变化过程会一直持续下去，并且在经历过数代之后，终将找到最优解。

## 交叉 ##

另一种修改程序的方法称为交叉或配对。

其做法是:

1. 从众多程序中选出两个表现优异者；
2. 将其组合在一起构造出一个新的程序，通常的组合方式是用一棵树的分支取代另一棵树的分支。

执行交又操作的函数以两棵树作为输入，并同时开始向下遍历。当到达某个随机选定的阈值时，该函数便会返回前一棵树的一份拷贝，树上的某个分支会被后一棵树上的一个分支所取代。通过同时对两棵树的即时遍历，函数会在每棵树上大致位于相同层次的节点处实施交叉操作。

![](image/06.png)

	def crossover(t1,t2,probswap=0.7,top=1):
	  if random()<probswap and not top:
	    return deepcopy(t2) 
	  else:
	    result=deepcopy(t1)
	    if hasattr(t1,'children') and hasattr(t2,'children'):
	      result.children=[crossover(c,choice(t2.children),probswap,0) 
	                       for c in t1.children]
	    return result

运行代码

	>>> random1=makerandomtree(2)
	>>> random1.display()
	multiply
	 subtract
	  p0
	  subtract
	   if
	    10
	    7
	    p1
	   isgreater
	    p1
	    8
	 add
	  p0
	  p0
	>>> random2=makerandomtree(2)
	>>> random2.display()
	multiply
	 5
	 2
	>>> cross=crossover(random1,random2)
	>>> cross.display()
	multiply
	 subtract
	  p0
	  subtract
	   if
	    10
	    7
	    p1
	   isgreater
	    p1
	    8
	 5

---

	>>> scorefunction(random1,hiddenset)
	55446
	>>> scorefunction(random2,hiddenset)
	135150
	>>> scorefunction(cross,hiddenset)
	122630
	>>> 

交换两个分支可能会完全改变程序的行为。导致各个程序最终接近于正确答案的原因可能是五花八门的，因此，将两个程序合并后得到的结果可能会与前两者都載然不同。

同样，此处我们的希望是，某些交又操作会对题解有所改进，并且这些题解会被保留到下一代。

## 构筑环境 ##

开始构筑供程序进化用的竞争环境。

![遗传编程的一个大体过程](image/01.png)

本质上，我们的思路是要生成一组随机程序并择优复制和修改，然后一直重复这一过程指导终止条件满足为止。

	def evolve(pc,popsize,rankfunction,maxgen=500,
	           mutationrate=0.1,breedingrate=0.4,pexp=0.7,pnew=0.05):
	  # Returns a random number, tending towards lower numbers. The lower pexp
	  # is, more lower numbers you will get
	  def selectindex():
	    return int(log(random())/log(pexp))
	
	  # Create a random initial population
	  population=[makerandomtree(pc) for i in range(popsize)]
	  for i in range(maxgen):
	    scores=rankfunction(population)
	    print scores[0][0]
	    if scores[0][0]==0: break
	    
	    # The two best always make it
	    newpop=[scores[0][1],scores[1][1]]
	    
	    # Build the next generation
	    while len(newpop)<popsize:
	      if random()>pnew:
	        newpop.append(mutate(
	                      crossover(scores[selectindex()][1],
	                                 scores[selectindex()][1],
	                                probswap=breedingrate),
	                        pc,probchange=mutationrate))
	      else:
	      # Add a random node to mix things up
	        newpop.append(makerandomtree(pc))
	        
	    population=newpop
	  scores[0][1].display()    
	  return scores[0][1]

## 一个简单的游戏 ##

## 更多可能性 ##

## 小结 ##