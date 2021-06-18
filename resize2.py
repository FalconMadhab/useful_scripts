import PIL
import os
import os.path
from PIL import Image

f = r'sedan_pure'
d = r'resized_sedan'
count=0
desired_size = 448
for file in os.listdir(f):
    count=count+1
    f_img = f+"/"+file
    d_img = d+"/"+file
    print(f_img, count)
    img = Image.open(f_img)
    old_size = img.size
    print(old_size)
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # print("New Size:",new_size)
    im = img.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
    print("After pasting:",new_im.size)
    new_im.save(d_img)

'''
from PIL import Image, ImageOps

desired_size = 384
im_pth = "/home/ninad/000125.jpg"

im = Image.open(im_pth)
old_size = im.size  # old_size[0] is in (width, height) format

ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
# use thumbnail() or resize() method to resize the input image

# thumbnail is a in-place operation

# im.thumbnail(new_size, Image.ANTIALIAS)

im = im.resize(new_size, Image.ANTIALIAS)
# create a new image and paste the resized on it

new_im = Image.new("RGB", (desired_size, desired_size))
new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))

new_im.show()
new_im.save("test.jpg")
'''