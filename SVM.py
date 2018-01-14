# -*- coding: utf-8 -*-
'''
	@file		NR.py
	@brief		支持向量机识别手写数字
	@author		USST.吕旭东
	@date		2018-1
'''

import numpy as np
import mlpy
import cv2


def getnumc(fn):
	fnimg = cv2.imread(fn)
	img = cv2.resize(fnimg, (40, 40))
	alltz = []
	for now_h in xrange(0, 40):
		xtz = []
		for now_w in xrange(0, 40):
			b = img[now_h, now_w, 0]
			g = img[now_h, now_w, 1]
			r = img[now_h, now_w, 2]
			btz = 255-b
			gtz = 255-g
			rtz = 255-r
			if btz > 0 or gtz > 0 or rtz > 0:
				nowtz = 1
			else:
				nowtz = 0
			xtz.append(nowtz)
		alltz += xtz
	return alltz

x = []
y = []
print 'USST 1906-2018'
print '加载数据训练中 ...'
for numi in xrange(0, 10):
	for numij in xrange(1, 5):
		fn = 'nums/'+str(numi)+'-'+str(numij)+'.jpg'
		x.append(getnumc(fn))
		y.append(numi)
x = np.array(x)
y = np.array(y)
svm = mlpy.LibSvm(svm_type='c_svc', kernel_type='poly', gamma=10)
svm.learn(x, y)

# print svm.pred(x)
testx = []
testy = []
print '训练结束，验证执行中 ...'
for iii in xrange(0, 10):
	testfn = 'nums/test/'+str(iii)+'-test.jpg'
	testx.append(getnumc(testfn))
	testy = svm.pred(testx)
print '验证结果如下：'
print testy
tj = 0
for ti in xrange(0,10):
	if testy[ti] == ti:
		tj += 1
acc = tj / 10
print '正确率为：',
print '%3.2f%%' %(acc * 100)
