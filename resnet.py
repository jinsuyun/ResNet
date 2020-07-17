import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms, datasets, models
USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS=300
BATCH_SIZE=64

train_loader=torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',
                     train=True,
                     download=True,
                     transform=transforms.Compose([
                         transforms.RandomCrop(32,padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,0.5,0.5),
                                              (0.5,0.5,0.5))]
                     )),
    batch_size=BATCH_SIZE,shuffle=True
)

test_loader=torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',
                     train=False,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,0.5,0.5),
                                              (0.5,0.5,0.5))
                     ])),
    batch_size=BATCH_SIZE,shuffle=True
)

class BasicBlock(nn.Module):
    def __init__(self,inplanes,planes,stride=1):
        super(BasicBlock,self).__init__()
        # in_planes(입력채널 수), out_planes(출력 채널수 - 몇장 conv1 layer를 만들지), stride, dilation, downsample, previous_dilation
        # kernel_size 커널의크기(필터 크기 - 3*3 필터를 만들고 싶으면 3을 기입)
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.shortcut=nn.Sequential()

        if stride != 1 or inplanes != planes:
            self.shortcut=nn.Sequential(
                nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes)
            )
        self.stride=stride

    def forward(self,x):

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        out+=self.shortcut(x)
        out=self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResNet,self).__init__()

        self.inplanes=64
        self.conv1=nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu=nn.ReLU(inplace=True)

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer2=self._make_layer(planes=64,num_blocks=2,stride=1)
        self.layer3=self._make_layer(planes=128,num_blocks=2,stride=2)
        self.layer4=self._make_layer(planes=256,num_blocks=2,stride=2)
        self.layer5=self._make_layer(planes=512,num_blocks=2,stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.linear=nn.Linear(512,num_classes)



    def _make_layer(self,planes,num_blocks,stride):

        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(BasicBlock(self.inplanes,planes,stride))
            self.inplanes=planes
        return nn.Sequential(*layers)

    def forward(self,x):

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)

        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out

model=ResNet().to(DEVICE)
optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0005)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)

print(model)
def train(model, train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(DEVICE),target.to(DEVICE)
        optimizer.zero_grad()
        output=model(data)
        loss=nn.functional.cross_entropy(output,target)
        loss.backward()
        optimizer.step()

def evaluate(model,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(DEVICE), target.to(DEVICE)
            output=model(data)

            test_loss+=nn.functional.cross_entropy(output,target,reduction='sum').item()

            pred=output.max(1,keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item()

    test_loss /=len(test_loader.dataset)
    test_accuracy=100.*correct/len(test_loader.dataset)
    return test_loss,test_accuracy

for epoch in range(1,EPOCHS+1):
    scheduler.step()
    train(model,train_loader,optimizer,epoch)
    test_loss,test_accuracy=evaluate(model,test_loader)

    print('[{}] Test Loss:{:.4f}, Accuracy: {:.2f}%'.format(epoch,test_loss,test_accuracy))
