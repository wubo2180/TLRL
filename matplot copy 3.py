'''
Author: your name
Date: 2022-02-27 22:13:01
LastEditTime: 2022-03-06 22:27:50
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \python workspace\pretrain-gnns\matplot copy 3.py
'''
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 38,
#     "mathtext.fontset":'stix',
}

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator, figure


fig=plt.figure(figsize=(11,9))
plt.grid(True,linestyle='--')

loss_list1=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\Loss\\js\\loss_list1_bbbp_gin.npy')
loss_list2=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\Loss\\js\\loss_list2_bbbp_gin.npy')
x = np.arange(1,30+1)
y=np.arange(0,1,0.1)
# plt.xticks([0,1,2,3,4,5,6],['1','5','10','15','20','25','30'],fontsize=30)
plt.tick_params(labelsize=23)
plt.yticks(y,fontsize=30)
y1 = np.array(loss_list1/100)
y2 = np.array(loss_list2)
# plt.title("Matplotlib demo")
plt.xlabel("Epochs",fontsize=30)
plt.ylabel("Loss",fontsize=30)
plt.plot(x,y1,label='RSS-GNN (JS)',zorder=10,linewidth=2)
plt.plot(x,y2,label='No pre-trained',zorder=10,linewidth=2)
plt.legend(fontsize=25)
plt.savefig('figure5.pdf',bbox_inches='tight')
# plt.show()
fig=plt.figure(figsize=(11,9))
plt.grid(True,linestyle='--')
plt.tick_params(labelsize=23)
acc1=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\\acc\\js\\train_acc_bbbp_gin.npy')
acc2=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\\acc\\js\\val_acc_bbbp_gin.npy')
acc3=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\\no_train_acc_bbbp_gin.npy')
acc4=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\\no_val_acc_bbbp_gin.npy')
x = np.arange(1,100+1)
y1 = np.array(acc1)
y2 = np.array(acc2)
y3 = np.array(acc3)
y4 = np.array(acc4)
# plt.title("Matplotlib demo")
plt.xlabel("Epochs",fontsize=30)
plt.ylabel("ROC-AUC (%)",fontsize=30)
plt.plot(x,y1,label='RSS-GNN (JS) train',zorder=10,linewidth=3)
plt.plot(x,y2,label='RSS-GNN (JS) val',zorder=10,linewidth=3)
plt.plot(x,y3,label='no pre-trained train',linestyle='-',zorder=10,linewidth=3)
plt.plot(x,y4,label='no pre-trained val',linestyle='-',zorder=10,linewidth=3)
plt.legend(fontsize=25)
# plt.plot(x,y2,label='No pre-trained')
plt.savefig('figure6.pdf',bbox_inches='tight')
# plt.show()
# fig=plt.figure(figsize=(11,9))
# plt.xticks(np.arange(1,31),fontsize=30)
# # plt.ylim((0.6, 0.8))
# # plt.set_ylim([60, 80])

# x = np.arange(1,30+1)
# plt.ylim((0, 1))

# plt.yticks(np.arange(0,1, 0.1),fontsize=30)
# plt.ylabel("Loss",fontsize=30)
# plt.xlabel(r"Epochs",fontsize=30)
# plt.plot(x , loss_list1/100, color='g',linestyle='-',marker='o',markersize=16,linewidth=2,label='RSS-GNN (JS)')
# plt.plot(x , loss_list2, color='mediumpurple',linestyle='-',marker='d',markersize=16,linewidth=2,label='RSS-GNN (MMD)')
# # plt.plot(x, CCC_plot, color='black',linestyle='-',marker='s',markersize=16,linewidth=2,label='RSS-GNN (RÉNYI)')
# plt.legend(fontsize=25)
# plt.savefig('figure4.pdf',bbox_inches='tight')
plt.show()