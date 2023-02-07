from PIL import Image
import numpy as np
import os


im_names = ['Savannas', 'Grasslands', 'Urban and built-up', 'Croplands', 'Forests', 'Waterbodies & Wetlands']

def mergeDictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3


class DistStore():
    
    def __init__(self, class_list):
        
        self.class_list = class_list
        self.class_dict = {self.class_list[i]:i for i in range(len(self.class_list))}
        self.dist_store = {i:0 for i in range(len(self.class_list))}
        print(self.class_dict, self.dist_store)
        
    def update_vals(self, lc, vals):
            
        try:
            class_id = self.class_dict[lc]
            if self.dist_store[class_id] == 0:
                self.dist_store[class_id] = vals
            else:
                self.dist_store[class_id] = mergeDictionary(self.dist_store[class_id], vals)
                
        except Exception as e:
            print(e, lc)
            
        

def calc_dists(dist_storer, channel):
    for country in os.listdir("../dimagery/"):
        print(country)
        c = 1
        for school in os.listdir("../dimagery/" + country)[0:100]:
            print(c, len(os.listdir("../dimagery/" + country)), end = "\r")
            school_id = school.split("_")[1].split(".")[0]
            lc = adm0[adm0["school_id"] == school_id].drop_duplicates(subset = "school_id")["LC_Type1_name"].squeeze()
            im = np.array(Image.open("../dimagery/" + country + "/" + school).convert("RGB"))
            b_vals = im[:, :, channel].flatten()
            bunique, bcounts = np.unique(b_vals, return_counts=True)
            dist_storer.update_vals(lc, dict(zip(bunique, bcounts)))
            c += 1
    
    return dist_storer


r_store = calc_dists(DistStore(im_names), 0)
        
        
DistStore(im_names)