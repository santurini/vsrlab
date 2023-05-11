import ptlflow
import torch
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter

model = ptlflow.get_model('flowformer', pretrained_ckpt='sintel').cuda()

io_adapter = IOAdapter(model, (480, 720))

inputs = io_adapter.prepare_inputs(torch.rand(8, 480, 720, 3).numpy())

input_images = inputs["images"][0]
video1 = input_images[:-1]
video2 = input_images[1:]
input_images = torch.stack((video1, video2), dim=1)
inputs["images"] = input_images.cuda()

with torch.no_grad():
   predictions = model(inputs)

print(predictions["flows"].shape)
