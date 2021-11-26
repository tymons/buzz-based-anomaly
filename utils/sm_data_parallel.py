from torch import nn


class SmDataParallel(nn.Module):
    def __init__(self, model, gpu_ids):
        super(SmDataParallel, self).__init__()
        self.model = nn.DataParallel(model, device_ids=gpu_ids).cuda()

    def forward(self, *input_data):
        return self.model(*input_data)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
