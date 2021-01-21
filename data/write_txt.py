import os

def wirte(VOC_CLASSES,xml_files_obj,txt_path):
    f = open(txt_path, 'w')
    cnt=0
    for i in xml_files_obj:
        img_path=i['img_path']
        img_objs=i['img_objs']
        f.write(img_path)
        for obj in img_objs:
            obj_name = obj['obj_name']
            obj_box = obj['obj_box']
            class_ind = VOC_CLASSES.index(obj_name)
            f.write(' %s+%s+%s+%s+%s'%(str(obj_box[0]),str(obj_box[1]),str(obj_box[2]),str(obj_box[3]),class_ind))
        f.write('\n')
        cnt+=1
        if cnt%100==0:
            print('write complete %d',cnt)
    f.close()
    return 0