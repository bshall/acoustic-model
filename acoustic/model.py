import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

URLS = {
    "hubert-discrete": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-discrete-d49e1c77.pt",
    "hubert-soft": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-soft-0321fd7e.pt",
}


class AcousticModel(nn.Module):
    def __init__(self, discrete: bool = False, upsample: bool = True):
        super().__init__()
        self.encoder = Encoder(discrete, upsample)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x, mels)

    @torch.inference_mode()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder.generate(x)


class Encoder(nn.Module):
    def __init__(self, discrete: bool = False, upsample: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(100 + 1, 256) if discrete else None
        self.prenet = PreNet(256, 256, 256)
        self.convs = nn.Sequential(
            nn.Conv1d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.ConvTranspose1d(512, 512, 4, 2, 1) if upsample else nn.Identity(),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prenet = PreNet(128, 256, 256)
        self.lstm1 = nn.LSTM(512 + 256, 768, batch_first=True)
        self.lstm2 = nn.LSTM(768, 768, batch_first=True)
        self.lstm3 = nn.LSTM(768, 768, batch_first=True)
        self.proj = nn.Linear(768, 128, bias=False)

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        mels = self.prenet(mels)
        x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
        res = x
        x, _ = self.lstm2(x)
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        x = res + x
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, xs: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(xs.size(0), 128, device=xs.device)
        h1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c3 = torch.zeros(1, xs.size(0), 768, device=xs.device)

        mel = []
        for x in torch.unbind(xs, dim=1):
            m = self.prenet(m)
            x = torch.cat((x, m), dim=1).unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)


class PreNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _acoustic(
    name: str,
    discrete: bool,
    upsample: bool,
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    acoustic = AcousticModel(discrete, upsample)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(URLS[name], progress=progress)
        consume_prefix_in_state_dict_if_present(checkpoint["acoustic-model"], "module.")
        acoustic.load_state_dict(checkpoint["acoustic-model"])
        acoustic.eval()
    return acoustic


def hubert_discrete(
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    r"""HuBERT-Discrete acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _acoustic(
        "hubert-discrete",
        discrete=True,
        upsample=True,
        pretrained=pretrained,
        progress=progress,
    )


def hubert_soft(
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    r"""HuBERT-Soft acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _acoustic(
        "hubert-soft",
        discrete=False,
        upsample=True,
        pretrained=pretrained,
        progress=progress,
    )
