import torch
class NetG(torch.nn.Module):

    def __init__(self) -> None:
        super(NetG, self).__init__()
        self.embedding_layer = torch.nn.Embedding(2, 128)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(256, 256 * 4 * 4, bias=False))
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, 3, 3, padding=1, bias=False),
            torch.nn.Tanh())
    
    def forward(self, inputs, labels):
        labels = self.embedding_layer(labels)
        inputs = torch.cat([inputs, labels], dim=1)
        outputs = self.fc_layer(inputs)
        outputs = outputs.reshape((-1, 256, 4, 4))
        outputs = self.conv_layer(outputs)
        return outputs
