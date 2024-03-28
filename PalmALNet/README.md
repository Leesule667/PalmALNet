# PalmALNet

## load PalmALNet
* from model.build import PalmALNet
* net = PalmALNet(num_classes=1000）

## Loss calculation
* for step, data in enumerate(train_bar):
    * images, labels = data
    * batch_size_, kk1, kk2, kk3 = images.shape
    * optimizer.zero_grad()
    * outputs, x2 = net(images.to(device))
    * loss1 = torch.nn.CrossEntropyLoss(outputs, labels.to(device))
    * loss2 = 0
    * for i in range(batch_size_):
        * for j in range(i, batch_size_):
            * if labels[i] == labels[j]:
                * loss2 += torch.nn.functional.pairwise_distance(x2[i, :], x2[j, :])
    * loss = loss1 + e * loss2
    * loss.backward()
    * optimizer.step()