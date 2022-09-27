#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
classes = ['A','B','C','D','E']
confusion_matrix = np.array([(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1),(9,1,3,4,0,1,1,1,1,1)],dtype=np.float64)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  #按照像素显示出矩阵
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(10)] for i in range(10)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]),va='center',ha='center')   #显示对应的数字

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.savefig('confusion_matrix.svg')
plt.show()

