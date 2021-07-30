import torch
import torch.nn as nn

def train(epoch:int, net:nn.Module, trainloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer):
    print('-'*50)
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0.
    correct = 0.
    total = 0.
    criterion = nn.CrossEntropyLoss()
    n_batches = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx!=0 and batch_idx%10==0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{n_batches}] Loss: {loss.item()}')
    
    print(f'Training Accuracy: {correct / total * 100.}')


def test(epoch:int, net:nn.Module, testloader:torch.utils.data.DataLoader):

    net.eval()
    test_loss = 0.
    correct = 0.
    total = 0.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        
    acc = 100.*correct/total
    print(f'At epoch: {epoch} test accuracy: {acc}')
    return acc
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc