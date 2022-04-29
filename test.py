import torch
from torchvision import transforms
from PIL import Image
model_path = '/home/shalom/pytorch_classifier/classifier_weights/resnet18_fuelingsplitclsfr_416x416.pt'
cat=['fueling','normal']
img_path='/home/shalom/Downloads/infrence_image.png'
def classify(model_path,img):
    #img = Image.open(img_path)
    print(type(img))
    transform = transforms.Compose([
                transforms.Resize((416,416)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load(model_path)
    #model = ort.InferenceSession("resnet18.onnx") 
    model.eval()
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze_(0)
    #input = Variable(image_tensor)
    input = image_tensor.to(device)
    output = model(input)
    #output = model.run(1,input)
    probabilities = torch.nn.functional.softmax(output,dim=1)
    top_prob, top_catid = torch.topk(probabilities,1)
    #print(cat[top_catid])
    img.close()
    print(probabilities)
    return top_catid
img=Image.open(img_path).convert('RGB')
print(cat[classify(model_path,img)])