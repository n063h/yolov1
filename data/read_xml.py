import xml.etree.ElementTree as ET
import os

def get_obj(filename):
    tree = ET.parse(filename)
    obj = []
    for object in tree.findall('object'):
        obj_name = object.find('name').text
        box = object.find('bndbox')
        obj_box = [int(float(box.find('xmin').text)),
                              int(float(box.find('ymin').text)),
                              int(float(box.find('xmax').text)),
                              int(float(box.find('ymax').text))]
        obj.append({'obj_name':obj_name,'obj_box':obj_box})
    return obj

def read(voc_dir):
    xml_dir=voc_dir+'/'+'Annotations'
    img_dir=voc_dir+'/'+'JPEGImages'
    xml_files = os.listdir(xml_dir)
    obj=[]
    cnt=0
    for xml_file in xml_files:
        xml_file_path=xml_dir+'/'+xml_file
        id=xml_file.split('.')[0]
        img_path=img_dir+'/'+id+'.jpg'
        img_objs=get_obj(xml_file_path)
        obj.append({'img_path':img_path,'img_objs':img_objs})
        cnt += 1
        if cnt % 100 == 0:
            print('read complete %d', cnt)
    return obj


