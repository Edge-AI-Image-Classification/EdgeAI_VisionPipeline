import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet50 import ResNet50
from pytorch_nndct.apis import torch_quantizer
# This code was modified from Dr. Kate Bowers in-class CSI4110/5110 4-8 pytorch-vitis demo

def load_model(model_path, device):
    model = ResNet50(num_classes=102)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

resnet50=load_model(r"resnet200.pth",'cpu')
resnet50.eval()  # Set the model to evaluation mode


# Define the image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


# Load and preprocess the image
img_path = 'pyramid.png' 
img_test = Image.open(img_path)
img_test = transform(img_test)
img_test = img_test.unsqueeze(0)  # Add batch dimension


# Run the image through the model
with torch.no_grad():
   output = resnet50(img_test)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
   print(f'{i} {top5_prob[i].item() * 100:.2f}%')


input("Press Enter to continue...")


# prepare images for calibration
print("Prepare images for calibration...")
calib_image_path = r"images/image_000"
calib_images = []


# go through each subfolder 0 thru 999
for i in range(10):
   first_image = calib_image_path + str(i+1)+".jpg"     
   img = Image.open(first_image)              
   if img.mode != 'RGB':                      # if black and white, convert to RGB
     img = img.convert('RGB')
   img = transform(img)                       # pre-process
   calib_images.append(img)                   # append to calibration images array
   print(first_image)


print("Converting to batch...")
calib_images_batch = torch.stack(calib_images[0:999])


print("Quantizing...")
quantizer = torch_quantizer("calib", resnet50, (calib_images_batch))
quant_model = quantizer.quant_model


print("Evaluating quantized model...")


device = torch.device("cpu")
quant_model.eval()
quant_model = quant_model.to(device)
output = quant_model(img_test)


probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
   print(f'{i} {top5_prob[i].item() * 100:.2f}%')

print("Exporting...")


quantizer.export_quant_config()


print("Deploying...")


# create batch with 1 test image for the final confirmation step of the model
test_images = []
test_images.append(img_test)
test_images_batch = torch.stack(test_images)


# create quantizer with "test" (i.e. evaluation and export) setting
input = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer("test", resnet50, (input))
quant_model = quantizer.quant_model
device = torch.device("cpu")
quant_model.eval()
quant_model = quant_model.to(device)
output = quant_model(img_test)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
   print(f'{i} {top5_prob[i].item() * 100:.2f}%')


quantizer.export_xmodel(deploy_check=True)
quantizer.export_onnx_model()