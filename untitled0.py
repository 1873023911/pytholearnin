import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels,kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

# 定义模型
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
net = LeNet().to(device)
net.train()

#加载数据集
mnist_train = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transforms.ToTensor())
batch_size = 128
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
mnist_test1=torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transforms.ToTensor())
test_loader=torch.utils.data.DataLoader(mnist_test1, batch_size=1, shuffle=False)


#测试函数

def evaluate_accuracy(data_iter, net,device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) ==y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # ⾃定义的模型, 3.13节之后不会⽤到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X,is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum+= (net(X).argmax(dim=1) ==y).float().sum().item()
            n += y.shape[0]
            return acc_sum / n


def train_ch5(net, train_iter, test_iter, batch_size, optimizer,device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0,time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) ==y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec' % (epoch + 1, train_l_sum / batch_count,train_acc_sum / n, test_acc, time.time() - start))

lr, num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device,num_epochs)


torch.save(net.state_dict(), "ALEXNET_cifar10_tzjzt_3td.pth")

def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image


# 攻击函数
def test( model, device, test_loader, epsilon ):
    # 精度计数器
    correct = 0
    adv_examples = []
    top5_examples = []
    top5_ciexamples=[]
    part_correct=[0]*10
    part_total=[0]*10
    # 循环遍历测试集中的所有示例
    for i, (data, target) in enumerate(test_loader):
        if i>9999:
            break
        part_total[target.item()]+=1
        data, target = data.to(device), target.to(device)
        # 设置张量的requires_grad属性，这对于攻击很关键
        data.requires_grad = True
        # 通过模型前向传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        ci_1=output[0][init_pred].item()
        # 如果初始预测是错误的，不打断攻击，继续
        if init_pred.item() != target.item():
            continue
        # 计算损失
        loss = F.nll_loss(output, target)
        # 将所有现有的渐变归零
        model.zero_grad()
        # 计算后向传递模型的梯度
        loss.backward()
        # 收集datagrad
        data_grad = data.grad.data
        # 唤醒FGSM进行攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # 重新分类受扰乱的图像
        output = model(perturbed_data)
        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        ci_2=output[0][final_pred].item()
        if final_pred.item() == target.item():
            correct += 1
            part_correct[target.item()]+=1
            # 保存0 epsilon示例的特例
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # 稍后保存一些用于可视化的示例
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            if len(top5_ciexamples) < 6:
                top5_ciex=perturbed_data.squeeze().detach().cpu().numpy()
                ex=data.squeeze().detach().cpu().numpy()
                top5_ciexamples.append((ex,init_pred.item(),ci_1,top5_ciex,final_pred.item(),ci_2))
                for i in range(len(top5_ciexamples)-1,0,-1):
                    if top5_ciexamples[i][5]>top5_ciexamples[i-1][5]:
                        kkk=top5_ciexamples[i]
                        top5_ciexamples[i]=top5_ciexamples[i-1]
                        top5_ciexamples[i-1]=kkk
            elif(ci_2>top5_ciexamples[5][5]):
                top5_ciexamples.pop()
                top5_ciex=perturbed_data.squeeze().detach().cpu().numpy()
                ex=data.squeeze().detach().cpu().numpy()
                top5_ciexamples.append((ex,init_pred.item(),ci_1,top5_ciex,final_pred.item(),ci_2))
                for i in range(5,0,-1):
                    if top5_ciexamples[i][5]>top5_ciexamples[i-1][5]:
                        kkk=top5_ciexamples[i]
                        top5_ciexamples[i]=top5_ciexamples[i-1]
                        top5_ciexamples[i-1]=kkk
        if len(top5_examples) < 5:
            top5_ex = perturbed_data.squeeze().detach().cpu().numpy()
            top5_examples.append( (init_pred.item(), final_pred.item(), top5_ex) )
    # 计算这个epsilon的最终准确度
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    for i in range(0,10,1):
        part_acc=part_correct[i]/part_total[i]
        print("Label:{}\tTestAccuracy = {} / {} = {}".format(i,part_correct[i],part_total[i],part_acc))
    
    # 返回准确性和对抗性示例
    return final_acc, adv_examples,top5_examples,top5_ciexamples


epsilons = [0, .01, .02, .03, .04, .05, .06]
accuracies = []
examples = []
top_examples = []
top5_ci_examples=[]
# 对每个epsilon运行测试 
for eps in epsilons:
    acc, ex ,top_ex,top5_ci_ex, = test(net, device,test_loader, eps)
    accuracies.append(acc)
examples.append(ex)
top_examples.append(top_ex)
top5_ci_examples.append(top5_ci_ex)



import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon") 
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
