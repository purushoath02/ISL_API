import torch
from utils import *
import cv2 
from configs import TransformerConfig
from models import Transformer

if __name__ == "__main__":
    path_video = "/home/red/Downloads/INCLUDE/Places/35. Bank/MVI_3335.MOV"
    
    model_path = "/home/red/Downloads/transformer_augs(1).pth"
    config = TransformerConfig(size="large")
    model = Transformer(config=config, n_classes=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model"])
    model.eval()
    #print(model)
    cap = cv2.VideoCapture(path_video)
    Prep = Prep()
    label_map = Prep.load_label_map()
    count = 0 
        
    temp =None
    while cap.isOpened():
        
        ret, img = cap.read()
        # if count < 10:
        #     continue
        
        frame = Prep.process_frame(img,ret)
        
        data = Prep.get_data(frame)
        # if temp != None: 
        #     data = torch.FloatTensor(np.concatenate((temp,data)))
        # else: 
        #     temp = data
        print(data.shape)
        output_tensor = prediction().get_prediction(data, model, Prep.load_label_map())
        print(output_tensor)
        # prediction= []
       
        # with torch.no_grad():
        #     #print(data)
        #     # output = model(data.cuda()).detach().cpu()
        #     output = model(data.cuda()).detach().cpu()
        #     output = torch.argmax(torch.softmax(output, dim=-1), dim=-1).numpy()
        #     #prediction.append(output)
        #     print(output)
        #     #prediction ={"predicted_label": label_map[output]}
        # del frame, data
        # #print(prediction)
        if count > 30:
            break
        count+=1
        
    cap.release