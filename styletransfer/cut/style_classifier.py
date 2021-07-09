import torch
from torch import nn

from data import create_dataset
from models.networks import SimpleClassifier
from options.train_options import TrainOptions

opt = TrainOptions().parse()  # get training options
dataset = create_dataset(opt)
net = SimpleClassifier()
optimizer_C = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
epochs = 100


def compute_C_loss(data):
    """
        Loss for class discriminator.
        """
    c_pred = net(data["B"])
    c_real = torch.argmax(data["B_class"], dim=1)

    from torch.autograd import Variable
    loss = nn.CrossEntropyLoss()

    loss = loss(c_pred, c_real)
    loss = Variable(loss, requires_grad=True)
    return loss


for epoch in range(epochs):
    for data in dataset:
        optimizer_C.zero_grad()
        loss_C = compute_C_loss(data)
        loss_C.backward()
       # print(f"Loss: {loss_C}")
        optimizer_C.step()
    correct = 0
    total = 0
    for data in dataset:
        total = total + 1
        out = net(data["B"])
        correct = correct + (torch.argmax(out) == data["B_class"]).float().sum()
    print(f"Accuracy: {correct/total}")


