from BasicNeuralNetwork import TrainMLP, TestMLP
from ConvolutionalNeuralNetwork import TrainCNN, TestCNN

def main():

    # training = TrainCNN()
    # training.start_training("cnn_mnist.pth")

    # training = TrainMLP()
    # training.start_training("mlp_mnist.pth")

    test = TestCNN("cnn_mnist.pth")
    test.test_example("raw/IMG_4.jpg", show_example=True)

    # test = TestMLP("mlp_mnist.pth")
    # test.test_example("raw/IMG_4.jpg", show_example=True)

if __name__ == '__main__':
    main()