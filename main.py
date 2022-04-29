import os
from datetime import datetime
from PIL import Image
import json
from torchvision import transforms
import torch
start_time = datetime.now()
source_loc = '/home/ninad/Documents/violations_dashboard/mobile_online'
dest = '/home/ninad/Documents/violations_dashboard/segreg_data'
base_model_path='/home/shalom/pytorch_classifier/resnet18feb24_fsmclsfr_416x416.pt'
classifier = ['FSM','person']
dnet_loc = '/home/shalom/yolov4/darknet'
yolov3_cfg_loc = '/home/shalom/yolov4/darknet/yolov3.cfg'
yolov3_weights_loc='/home/shalom/yolov4/darknet/yolov3.weights'

def loc_txt(source,desti):
    data = []
    for i in os.listdir(source):
        if i[-3:] in ['jpg']:
            data.append(source+'/'+i+'\n')
    f = open(desti+'/'+desti+'.txt','a')
    f.writelines(data)
    f.close()

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
    print(probabilities)
    return top_catid,str(top_prob).split()[0][-9:-3]

def read_objects(obj_cor_json_loc,dest,model_path,thresh=60):
    obj_cor_json_file = open(obj_cor_json_loc)
    data = json.load(obj_cor_json_file)
    for i in classifier:
        if not os.path.exists(dest+'/'+i):
            os.makedirs(dest+'/'+i)
    for i in data:
        image = Image.open(i['filename'])
        fname = i['filename'].split('/')[-1]
        obn = 0
        for j in i['objects']:
            if j['class_id'] == 0 and j['confidence']> thresh/100:
                cord = j["relative_coordinates"]
                x = cord["center_x"]
                y = cord["center_y"]
                w = cord["width"]
                h = cord["height"]
                a = image.size
                hh = float(h)*float(a[1])
                ww = float(w)*float(a[0])
                x2 = float(x)*float(a[0]) + (ww/2)
                y2 = float(y)*float(a[1]) + (hh/2)
                x1 = float(x)*float(a[0])-(ww/2)
                y1 = float(y)*float(a[1])-(hh/2)
                crop = image.crop((x1,y1,x2,y2))
                top_catid,top_prob = classify(model_path,crop)
                crop.save(dest+'/'+classifier[top_catid]+'/'+fname[:-4]+str(obn)+'.jpg')
        image.close()

def detect_person(dnet_loc,source_loc,dest,base_model_path,classes):
    temp_fol_all = source_loc.split('/')[-1]
    if not os.path.exists(temp_fol_all):
        os.mkdir(temp_fol_all)
    loc_txt(source_loc,temp_fol_all)

    working_path = os.getcwd()
    imgs_location_txt = working_path+'/'+temp_fol_all+'/'+source_loc.split('/')[-1]+'.txt'
    json_loc = working_path+'/'+temp_fol_all+'/'+source_loc.split('/')[-1]+'.json'
    os.chdir(dnet_loc)
    os.system('./darknet detector test cfg/coco.data '+yolov3_cfg_loc+' '+yolov3_weights_loc+' -dont_show -ext_output -out '+json_loc+' < '+imgs_location_txt)
    os.system('killall -9 darknet')
    os.chdir(working_path)
    os.remove(imgs_location_txt)
    read_objects(json_loc,dest,base_model_path,60)
detect_person(dnet_loc,source_loc,dest,base_model_path,classifier)
print(datetime.now()-start_time)

#obj_loc = "/home/shalom/taskss/task11/script_for_segregation/vid_im/vid_im.json"
#read_objects(obj_loc,base_model_path)