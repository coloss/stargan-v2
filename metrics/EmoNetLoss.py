import torch
from torchvision.transforms.transforms import Resize
from pathlib import Path
import sys
import inspect
import torch.nn.functional as F


def get_emonet(device=None, load_pretrained=True):
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path_to_emonet = Path(__file__).absolute().resolve().parent.parent.parent / "emonet"
    if not(str(path_to_emonet) in sys.path  or str(path_to_emonet.absolute()) in sys.path):
        print(f"Adding EmoNet path '{path_to_emonet}'")
        sys.path += [str(path_to_emonet)]

    from emonet.models import EmoNet
    # n_expression = 5
    n_expression = 8

    # Create the model
    net = EmoNet(n_expression=n_expression).to(device)

    # if load_pretrained:
    state_dict_path = Path(
        inspect.getfile(EmoNet)).parent.parent.parent / 'pretrained' / f'emonet_{n_expression}.pth'
    print(f'Loading the EmoNet model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=False)
    if not load_pretrained:
        print("Created an untrained EmoNet instance")
        net.reset_emo_parameters()

    net.eval()
    return net


class EmoNetLoss(torch.nn.Module):
# class EmoNetLoss(object):

    def __init__(self, device=None, emonet=None, unnormalize=False, feat_metric='l1'):
        super().__init__()
        self.emonet = emonet or get_emonet(device).eval()
        self.emonet.requires_grad_(False)
        # self.emonet.eval()
        # self.emonet = self.emonet.requires_grad_(False)
        # self.transforms = Resize((256, 256))
        self.size = (256, 256)

        self.feat_metric = feat_metric
        self.valence_loss = F.l1_loss
        self.arousal_loss = F.l1_loss
        # self.expression_loss = F.kl_div
        self.expression_loss = F.l1_loss
        self.input_emotion = None
        self.output_emotion = None
        self.unnormalize = unnormalize

    @property
    def network(self):
        return self.emonet

    def to(self, *args, **kwargs):
        self.emonet = self.emonet.to(*args, **kwargs)
        # self.emonet = self.emonet.requires_grad_(False)
        # for p in self.emonet.parameters():
        #     p.requires_grad = False

    def eval(self):
        self.emonet = self.emonet.eval()
        # self.emonet = self.emonet.requires_grad_(False)
        # for p in self.emonet.parameters():
        #     p.requires_grad = False

    def train(self, mode: bool = True):
        # super().train(mode)
        if hasattr(self, 'emonet'):
            self.emonet = self.emonet.eval() # evaluation mode no matter what, it's just a loss function
            # self.emonet = self.emonet.requires_grad_(False)
            # for p in self.emonet.parameters():
            #     p.requires_grad = False

    def forward(self, images):
        return self.emonet_out(images)

    def emonet_out(self, images):
        if self.unnormalize:
            images = self._transform(images)
        images = F.interpolate(images, self.size, mode='bilinear')
        # images = self.transform(images)
        return self.emonet(images, intermediate_features=True)

    def _transform(self, img):
        # stargan outputs images in range (-1,1), emonet expects them in (0,1)
        img = (img + 1) / 2
        return img

    def emo_feat_loss(self, x, y):
        if self.feat_metric == 'l1':
            return F.l1_loss(x, y)
        elif self.feat_metric == 'l2':
            return F.mse_loss(x, y)
        elif self.feat_metric == 'cos':
            return (1. - F.cosine_similarity(x, y, dim=1)).mean()
        raise ValueError(f"Invalid feat_metric: '{self.feat_metric}'")

    def compute_loss(self, input_images, output_images):
        # input_emotion = None
        # self.output_emotion = None
        input_emotion = self.emonet_out(input_images)
        output_emotion = self.emonet_out(output_images)
        self.input_emotion = input_emotion
        self.output_emotion = output_emotion

        emo_feat_loss_1 = self.emo_feat_loss(input_emotion['emo_feat'], output_emotion['emo_feat'])
        emo_feat_loss_2 = self.emo_feat_loss(input_emotion['emo_feat_2'], output_emotion['emo_feat_2'])
        valence_loss = self.valence_loss(input_emotion['valence'], output_emotion['valence'])
        arousal_loss = self.arousal_loss(input_emotion['arousal'], output_emotion['arousal'])
        expression_loss = self.expression_loss(input_emotion['expression'], output_emotion['expression'])
        return emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss

    @property
    def input_emo(self):
        return self.input_emotion

    @property
    def output_emo(self):
        return self.output_emotion

#
# if __name__ == "__main__":
#     net = get_emonet(load_pretrained=False)