import torch

class ResBlockConvUpsample2d(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel) -> None:
        super(ResBlockConvUpsample2d, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(negative_slope=0.2), 
            torch.nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channel))
        self.h_layer = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channel, out_channel, 1, bias=False),
            torch.nn.BatchNorm2d(out_channel))
        self.leakyrelu_layer = torch.nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, inputs):
        h_outputs = self.h_layer(inputs)
        outputs = self.conv_layer(inputs)
        outputs = self.leakyrelu_layer(outputs + h_outputs)
        return outputs

class NetG(torch.nn.Module):

    def __init__(self) -> None:
        super(NetG, self).__init__()
        self.embedding_layer = torch.nn.Embedding(2, 128)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 256 * 4 * 4))
        self.conv_layer = torch.nn.Sequential(
            ResBlockConvUpsample2d(256, 128),
            ResBlockConvUpsample2d(128, 64),
            ResBlockConvUpsample2d(64, 32),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(32, 3, 3, padding=1, bias=False),
            torch.nn.Tanh())
    
    def forward(self, inputs, labels):
        embedded = self.embedding_layer(labels)
        inputs = inputs * embedded
        outputs = self.fc_layer(inputs)
        outputs = outputs.reshape((-1, 256, 4, 4))
        outputs = self.conv_layer(outputs)
        return outputs