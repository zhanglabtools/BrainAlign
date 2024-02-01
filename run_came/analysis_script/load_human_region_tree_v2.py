# -- coding: utf-8 --
# @Time : 2022/12/18 20:53
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : load_mouse_region_tree.py



import json
from treelib import Node, Tree

tree = Tree()
tree.create_node('Human brain', 'Human brain')

def add_node(tree_dict, key_input):
    #print(type(tree_dict[key_input]))
    if tree_dict['children'] == []:
        return
    #elif isinstance(tree_dict['children'], list):
    #    for key in tree_dict['children']:
    #        tree.create_node(key, key, parent=key_input)
    else:
        for dict in tree_dict['children']:
            tree.create_node(dict['name'], dict['name'], parent=key_input)
            add_node(dict, dict['name'])

if __name__ == '__main__':
    #tree_path = '../brain_mouse_2020sa/DSURQE_tree.json'
    tree_path = '../brain_human_mouse/human_structure.json'
    f = open(tree_path, 'rb')
    tree_dict = json.load(f)

    #print(tree_dict)
    #print('1st level:', tree_dict.keys())
    #print('2nd level:', tree_dict['msg'])
    #print('3rd level:', tree_dict['msg'][0].keys())


    #print('4th level:', tree_dict['msg'][0]['children'].keys())
    for dict in tree_dict['msg'][0]['children']:
        print(dict['name'])
    tree_dict_new = tree_dict['msg'][0]['children']

    print(tree_dict_new)


    '''
    print('5th level:', tree_dict['msg'][0]['children']['Basic cell groups and regions'].keys(),
          tree_dict['msg'][0]['children']['fiber tracts'].keys(),
          tree_dict['msg'][0]['children']['ventricular systems'].keys())
    print('6th level:', tree_dict['msg'][0]['children']['Basic cell groups and regions']['children'].keys(),
          tree_dict['msg'][0]['children']['fiber tracts']['children'].keys(),
          tree_dict['msg'][0]['children']['ventricular systems']['children'].keys())
    print('7th level:', tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebrum']['children'].keys(),
          tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Brain stem']['children'].keys(),
          tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebellum']['children'].keys())
    print('8th level:')
    for region in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebrum']['children'].keys():
        print(tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebrum']['children'][region]['children'].keys())
    for region in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Brain stem'][
        'children'].keys():
        print(tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Brain stem']['children'][
                  region]['children'].keys())
    for region in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebellum'][
        'children'].keys():
        print(tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebellum']['children'][region]['children'].keys())

    print('9th level:')
    region_num = 0
    for region in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebrum'][
        'children'].keys():
        for region_1 in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebrum']['children'][
                  region]['children'].keys():
            region_list = tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebrum']['children'][
                  region]['children'][region_1]['children'].keys()
            print(region_list)
            region_num += len(region_list)
    for region in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Brain stem'][
        'children'].keys():
        for region_1 in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Brain stem']['children'][
                  region]['children'].keys():
            if region_1 != 'Midbrain-other':
                region_list = tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Brain stem']['children'][region]['children'][region_1]['children'].keys()

            else:
                region_list = [region_1]
            print(region_list)
            region_num += len(region_list)
    for region in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebellum'][
        'children'].keys():
        for region_1 in tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebellum']['children'][
                  region]['children'].keys():
            region_list = tree_dict['msg'][0]['children']['Basic cell groups and regions']['children']['Cerebellum']['children'][
                  region]['children'][region_1]['children'].keys()
            print(region_list)
            region_num += len(region_list)

    print('region number = ', region_num)
    '''


    for dict in tree_dict_new:
        tree.create_node(dict['name'], dict['name'], parent='Human brain')
        add_node(dict, dict['name'])

    tree.show()



