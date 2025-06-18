from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "/Users/hechenyi/ML-learning/dataset/raw-img/elefante/e83cb00a2ef1083ed1584d05fb1d4e9fe777ead218ac104497f5c978a4eebdbd_640.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
# tensor_image是tonsor数据类型，包含有神经网络以及其方向传播的配置，例如gradient,is_cpu,data等配置
writer = SummaryWriter("logs")

tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img,1)
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
writer.add_image("img_norm", img_norm)
writer.close()
print(tensor_img.shape)
