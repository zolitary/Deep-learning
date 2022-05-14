import numpy as np
import matplotlib.pyplot as plt
history=np.load('history.npy',allow_pickle=True).item()
train_loss=[x.data.cpu().numpy() for x in history['train_loss']]
valid_loss =[x.data.cpu().numpy() for x in history['valid_loss']]
plt.figure(figsize=(5,5),dpi=100)
plt.plot(train_loss, label='train_loss')
plt.plot(valid_loss, label="valid_loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('train_loss and valid_loss')

plt.show()

'''
plt.figure(figsize=(5,5),dpi=100)
plt.title('valid_loss')
plt.plot(valid_loss)
plt.show()

print(history['best_epoch'])
plt.subplot(1, 2, 1)
plt.plot(val_acc, label="分类精度")
plt.xlabel('epoch')
plt.ylabel('分类精度')
plt.subplot(1, 2, 2)
plt.plot(val_loss, label='损失值')
plt.xlabel('epoch')
plt.ylabel('损失值')
plt.legend()
plt.show()
'''