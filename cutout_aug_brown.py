import argparse
import os,glob,cv2,random
import numpy as np

if __name__ == '__main__':

        
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, type=str, help='path for folder containing images')  
    parser.add_argument('--txt', required=True, type=str,  help='path for folder containing labels')
    parser.add_argument('--out', required=True, type=str, help='path for folder to contain augmentation output')
    parser.add_argument('--perc', default=10, type=float, help='percenatge of bounding box to perform augmentation on')
    parser.add_argument('--ext', default='jpg', type=str, help='Extention of images')
    parser.add_argument('--class_id', type=str, help='class_id of object')
    

    opt = parser.parse_args()

    img_path=opt.img
    txt_path=opt.txt
    output_path=opt.out
    perc=opt.perc/100
    ext=opt.ext
    obj_class = opt.class_id


    if not os.path.exists(output_path):
        os.mkdir(output_path)

    iml=glob.glob(os.path.join(img_path,'*.'+ext))

    for file in iml:
        
        img=cv2.imread(file)
        
        height=img.shape[0]
        width=img.shape[1]
        
        mean= (31,102,156)
        
        txt=os.path.join(txt_path,file.split(os.sep)[-1][:-3]+'txt')

        if not os.path.exists(txt):
            continue
        
        with open(txt) as f:
            lines=f.readlines()

        if obj_class is not None:
            L=[]
            for line in lines:
                if line.split()[0]==obj_class:
                    L.append(line)
            lines = L
        
        random.shuffle(lines)
            
        s={}
        for line in lines:
            if line.split()[0] not in s.keys():
                s[line.split()[0]] = 1
            else:
                s[line.split()[0]]+=1
                
        m={key:random.randint(1,s[key]) for key in s.keys()}
        
        bb_list=[]
        for line in lines:
            cls=line.split()[0]

            if m[cls]>=1:
                bb_list.append([float(x) for x in line.split()][1:])
            m[cls]-=1
            
        bb=np.array(bb_list)

        x1=bb[:,0]-bb[:,2]/2
        x2=bb[:,0]+bb[:,2]/2
        y1=bb[:,1]-bb[:,3]/2
        y2=bb[:,1]+bb[:,3]/2

        bb_cv=np.column_stack((x1,y1,x2,y2))
        
        bb_cv[:,[0,2]]=bb_cv[:,[0,2]]*width
        bb_cv[:,[1,3]]=bb_cv[:,[1,3]]*height
        
        bb_cv=bb_cv.astype(int)
        
        for row in bb_cv:
            
            h_diff=row[3]-row[1]
            w_diff=row[2]-row[0]

            lx=min(h_diff,w_diff)
            
            if int(perc*h_diff*w_diff) <= lx**2:
                b_h=b_w=int(np.sqrt(perc*h_diff*w_diff))

            else:
                b_h=b_w=lx

                rem=int((perc*h_diff*w_diff)/lx)

                if h_diff==lx:
                    b_w=rem
                else:
                    b_h=rem
                    
            h_range=h_diff-b_h
            w_range=w_diff-b_w
            
            mask_x1=random.randint(row[0],row[0]+w_range)
            mask_y1=random.randint(row[1],row[1]+h_range)

            mask_x2=mask_x1+b_w
            mask_y2=mask_y1+b_h
            
            img[mask_y1:mask_y2,mask_x1:mask_x2]=mean
            
        cv2.imwrite(os.path.join(output_path,file.split(os.sep)[-1][:-4]+'.jpg'),img)
    