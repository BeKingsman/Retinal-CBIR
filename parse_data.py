import os, shutil

cwd = os.getcwd()
if "images" in os.listdir(cwd):
        shutil.rmtree(os.path.join(cwd,"images"))

os.mkdir("images")



def parse_kaggle():
        kaggle_path = os.path.join(os.path.join(cwd,"datasets"),"kaggle")
        for cat in os.listdir(kaggle_path):
                if cat != ".DS_Store":
                        cat_prefix = cat.split(".")[-1]
                        images_path = os.path.join(kaggle_path,cat)
                        images= os.listdir(images_path)
                        for i in range(len(images)):
                                path = os.path.join(images_path,images[i])
                                dst_path = os.path.join(os.path.join(cwd,"images"),cat_prefix+"-"+str(i+1)+"."+images[i].split(".")[-1])
                                shutil.copy(path,dst_path)
                                print(dst_path)
                                



parse_kaggle()