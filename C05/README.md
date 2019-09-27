# 优化 #

Optimization

本文教授一系列被称为随机优化stochastic optimization的技术来解决协作类问题。

**优化技术特别擅长于处理**：受多种变量的影响，存在许多可能解的问题，以及结果因这些变量的组合而产生很大变化的问题。

这些优化技术有着大量的应用：

1. 物理学——研究分子运动
2. 生物学——预测蛋白质的结构
3. 计算机科学——测定算法的最坏可能运行时间
4. NASA——设计具有正确操作特性的天线

---

**优化算法**是通过尝试许多不同题解并给这些题解打分以确定其质量的方式来找到一个问题的最优解。

优化算法的典型的应用场景是，存在大量可能的题解以至于无法对它们进行一一尝试的情况。

**最简单也是最低效的求解方法**，尝试随机猜测的上千个题解，并从中找出最佳解来。

更有效方法，则是一种对题解可能有改进的方式来对其进行智能化地修正。


---

本文例子

1. 如何计划旅游才最合理且划算？
2. 如何基于人们的偏好来分配有限的资源？
3. 如何用最少的交叉线来可视化社会网络？


## 组团旅游 ##

[本例源代码](optimization.py)

家庭人员信息

数据结构 人s=[(人名，居住地),...]

	import time
	import random
	import math
	
	people = [('Seymour','BOS'),
	          ('Franny','DAL'),
	          ('Zooey','CAK'),
	          ('Walt','MIA'),
	          ('Buddy','ORD'),
	          ('Les','OMA')]
	# 纽约的Laguardia机场
	destination='LGA'

---

[航班数据文件](schedule.txt)

部分航班信息

	#起点，终点，起飞时间，到达时间，价格
	LGA,OMA,6:19,8:13,239
	OMA,LGA,6:11,8:31,249
	LGA,OMA,8:04,10:59,136
	OMA,LGA,7:39,10:24,219
	LGA,OMA,9:31,11:43,210
	OMA,LGA,9:15,12:03,99
	...

---

将上述的数据载入一个字典中

	flights={}
	# 
	for line in file('schedule.txt'):
	  origin,dest,depart,arrive,price=line.strip().split(',')
	  flights.setdefault((origin,dest),[])
	
	  #键（起，止点）- 值（起飞时间，到达时间，价格）
	  # Add details to the list of possible flights
	  flights[(origin,dest)].append((depart,arrive,int(price)))

---

下一个函数用于计算某个给定时间在一天中的分钟数。

	def getminutes(t):
	  x=time.strptime(t,'%H:%M')
	  return x[3]*60+x[4]

## 描述题解 ##

**家庭中的每个成员应该乘坐哪个航班？**


## 成本函数 ##



## 随机搜索 ##



## 爬山法 ##



## 模拟退火算法 ##



## 遗传算法 ##



## 真实的航班搜索 ##



## 涉及偏好的优化 ##



## 网络可视化 ##



## 其他可能的应用场景 ##



## 小结 ##


