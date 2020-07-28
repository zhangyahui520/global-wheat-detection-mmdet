import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import argparse

def generate_test_annotation(test_data_path, annotation_path):
    test_anno_dict = {}
    test_anno_dict["info"] = "created by chen "
    test_anno_dict["license"] = ["license"]
    id = 0
    test_anno_list = []
    for img in os.listdir(test_data_path):
        if img.endswith('jpg'):
            id += 1
            img_info = {}
            img_size = Image.open(os.path.join(test_data_path, img)).size
            img_info['height'] = img_size[1]
            img_info['width'] = img_size[0]
            img_info['id'] = id            
            img_info['file_name'] = img
            test_anno_list.append(img_info)
    test_anno_dict["images"] = test_anno_list
    test_anno_dict["categories"] = [
    {
      "id": 1,
      "name": "wheat"
    }
  ]
    with open(annotation_path, 'w+') as f:
        json.dump(test_anno_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate global wheat detection test json file for submitting')
    parser.add_argument('--test-data-path', help='test image dir', type=str,default='../global_wheat_detection/test/')
    parser.add_argument('--save-json-path', help='save json path', type=str,default='../global_wheat_detection/annotations/test_wheat.json')
    args = parser.parse_args()
    print(args)
    print("generate test json label file.")
    generate_test_annotation(args.test_data_path,args.save_json_path)
