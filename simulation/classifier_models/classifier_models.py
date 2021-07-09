# import the necessary packages
from torch import flatten
from torch.nn import Conv2d, Linear, LogSoftmax, MaxPool2d, Module, ReLU, Dropout, BatchNorm2d
from torchvision.transforms import Resize


class LeNet(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        if dim == 256:
            self.fc1 = Linear(in_features=186050, out_features=500)  # 288x256 4950 , 256x256 4050
        else:
            self.fc1 = Linear(in_features=210450, out_features=500)

        self.relu3 = ReLU()

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)

        # return the output predictions
        return output
class Nvidia_model(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(Nvidia_model, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=3,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv3 = Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv4 = Conv2d(in_channels=36, out_channels=48,
                            kernel_size=(3, 3))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv5 = Conv2d(in_channels=48, out_channels=64,
                            kernel_size=(3, 3))
        self.relu5 = ReLU()
        self.maxpool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        if dim == 256:
            self.fc1 = Linear(in_features=1600, out_features=100)  # 288x256 4950 , 256x256 4050
        else:
            self.fc1 = Linear(in_features=1920, out_features=100)
        self.relu6 = ReLU()

        # initialize first (and only) set of FC => RELU layers
        self.fc2 = Linear(in_features=100, out_features=50)
        self.relu7 = ReLU()

        # initialize our softmax classifier
        self.fc3 = Linear(in_features=50, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)

        x = self.fc2(x)
        x = self.relu7(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc3(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

class LeNet_mod(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(LeNet_mod, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=5,
                            kernel_size=(10, 10), dilation=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(5, 5), stride=(2, 2))

        self.bn = BatchNorm2d(5)

        # initialize first set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=5, out_channels=10,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bn2 = BatchNorm2d(10)

        # initialize first set of CONV => RELU => POOL layers
        self.conv3 = Conv2d(in_channels=10, out_channels=20,
                            kernel_size=(5, 5))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bn3 = BatchNorm2d(20)

        # initialize second set of CONV => RELU => POOL layers
        self.conv4 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bn4 = BatchNorm2d(50)

        self.drop_layer = Dropout(p=0.4)

        # initialize first (and only) set of FC => RELU layers
        if dim == 256:
            self.fc1 = Linear(in_features=4050, out_features=500)  # 288x256 4950 , 256x256 4050
        else:
            self.fc1 = Linear(in_features=4950, out_features=500)
        self.relu3 = ReLU()

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.bn(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.bn2(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.bn3(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.bn4(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.drop_layer(x)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)

        # return the output predictions
        return output
class LeNet_resized(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(LeNet_resized, self).__init__()

        if dim == 256:
            self.resize = Resize((96, 96))
        else:
            self.resize = Resize((108, 96))

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        if dim == 256:
            self.fc1 = Linear(in_features=22050, out_features=500)  # 288x256 4950 , 256x256 4050
        else:
            self.fc1 = Linear(in_features=25200, out_features=500)

        self.relu3 = ReLU()

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.resize(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)

        # return the output predictions
        return output


class Nvidia_model_resized(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(Nvidia_model_resized, self).__init__()
        if dim == 256:
            self.resize = Resize((96, 96))
        else:
            self.resize = Resize((108, 96))
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=3,
                            kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv3 = Conv2d(in_channels=24, out_channels=36, kernel_size=(3, 3))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv4 = Conv2d(in_channels=36, out_channels=48,
                            kernel_size=(3, 3))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv5 = Conv2d(in_channels=48, out_channels=64,
                            kernel_size=(3, 3))
        self.relu5 = ReLU()
        self.maxpool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=64, out_features=32)  # 288x256 4950 , 256x256 4050

        self.relu6 = ReLU()

        # initialize our softmax classifier
        self.fc3 = Linear(in_features=32, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.resize(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc3(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output
class LeNet_resized_cons(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(LeNet_resized_cons, self).__init__()

        if dim == 256:
            self.resize = Resize((96, 96))
        else:
            self.resize = Resize((108, 96))

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bn = BatchNorm2d(20)

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        if dim == 256:
            self.fc1 = Linear(in_features=22050, out_features=500)  # 288x256 4950 , 256x256 4050
        else:
            self.fc1 = Linear(in_features=25200, out_features=500)

        self.relu3 = ReLU()

        self.do = Dropout(p=0.4)

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.resize(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x= self.bn(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.do(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)

        # return the output predictions
        return output

class Nvidia_model_resized_cons(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(Nvidia_model_resized_cons, self).__init__()
        if dim == 256:
            self.resize = Resize((96, 96))
        else:
            self.resize = Resize((108, 96))
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=3,
                            kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bn = BatchNorm2d(3)


        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv3 = Conv2d(in_channels=24, out_channels=36, kernel_size=(3, 3))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv4 = Conv2d(in_channels=36, out_channels=48,
                            kernel_size=(3, 3))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv5 = Conv2d(in_channels=48, out_channels=64,
                            kernel_size=(3, 3))
        self.relu5 = ReLU()
        self.maxpool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=64, out_features=32)  # 288x256 4950 , 256x256 4050

        self.relu6 = ReLU()
        self.do = Dropout(p=0.4)

        # initialize our softmax classifier
        self.fc3 = Linear(in_features=32, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.resize(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x= self.bn(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)

        x = self.do(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc3(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

class LeNet_resized_cons_no_bn(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(LeNet_resized_cons_no_bn, self).__init__()

        if dim == 256:
            self.resize = Resize((96, 96))
        else:
            self.resize = Resize((108, 96))

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        if dim == 256:
            self.fc1 = Linear(in_features=22050, out_features=500)  # 288x256 4950 , 256x256 4050
        else:
            self.fc1 = Linear(in_features=25200, out_features=500)

        self.relu3 = ReLU()

        self.do = Dropout(p=0.4)

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.resize(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)


        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.do(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)

        # return the output predictions
        return output
class Nvidia_model_resized_cons_no_bn(Module):
    def __init__(self, numChannels, classes, dim):
        # call the parent constructor
        super(Nvidia_model_resized_cons_no_bn, self).__init__()
        if dim == 256:
            self.resize = Resize((96, 96))
        else:
            self.resize = Resize((108, 96))
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=3,
                            kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv3 = Conv2d(in_channels=24, out_channels=36, kernel_size=(3, 3))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv4 = Conv2d(in_channels=36, out_channels=48,
                            kernel_size=(3, 3))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv5 = Conv2d(in_channels=48, out_channels=64,
                            kernel_size=(3, 3))
        self.relu5 = ReLU()
        self.maxpool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=64, out_features=32)  # 288x256 4950 , 256x256 4050

        self.relu6 = ReLU()
        self.do = Dropout(p=0.4)

        # initialize our softmax classifier
        self.fc3 = Linear(in_features=32, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.resize(x)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)


        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)

        x = self.do(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc3(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

