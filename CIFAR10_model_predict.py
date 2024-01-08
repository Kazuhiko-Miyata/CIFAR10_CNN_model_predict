import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #畳み込み層:（入力チャンネル数、フィルタ数、フィルタサイズ）
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        #プーリング層:（領域サイズ、領域の間隔）
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)#全結合層
        self.dropout = nn.Dropout(p=0.5)
        #ドロップアウト:（p=ドロップアウト率）
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()
net.cuda()
print(net)
#ここまでは上記のコード

from torch import optim

#交差エントロピー誤差関数
loss_fnc = nn.CrossEntropyLoss()

#最適化アルゴリズム
optimizer = optim.Adam(net.parameters())

#損失のログ
record_loss_train = []
record_loss_test = []

#学習
for i in range(20):#20エポック学習
    net.train()#訓練モード
    loss_train = 0
    for j, (x, t) in enumerate(train_loader):
      #ミニバッチ(x, t)を取り出す
      x, t = x.cuda(), t.cuda()#GPU対応
      y = net(x)
      loss = loss_fnc(y, t)
      loss_train += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    net.eval()#評価モード
    loss_test = 0

    for j, (x, t) in enumerate(test_loader):
      #ミニバッチ(x, t)を取り出す
      x, t = x.cuda(), t.cuda()
      y = net(x)
      loss = loss_fnc(y, t)
      loss_test += loss.item()
    loss_test /= j+1
    record_loss_test.append(loss_test)

    if i%1 ==0:
        print("Epoch:", i, "Loss_Train:",
              loss_train, "Loss_Test:", loss_test)
        
cifar10_loader = DataLoader(cifar10_test, batch_size=1, shuffle=True)
for batch_idx, (images, labels) in enumerate(cifar10_loader):
    print(f"Batch{batch_idx + 1}:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

plt.imshow(images[0].permute(1, 2, 0))
plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
plt.show()

net.eval()
x, t = images.cuda(), labels.cuda(),
y = net(x)
print("正解:", cifar10_classes[labels[0]],
      "予測結果:", cifar10_classes[y.argmax().item()])
