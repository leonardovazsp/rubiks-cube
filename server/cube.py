import os
import copy
import pickle
import random
import requests
import numpy as np

def find_ip(server_type):
    '''
    Scan through the network to find the server.
    '''

    ip = None
    for i in range(100, 120):
        ip = '192.168.1.' + str(i)
        try:
            print('Trying ' + ip)
            r = requests.get('http://' + ip + ':8000/get_device/', timeout=0.5)
            if r.text == server_type:
                print('Found ' + server_type + ' server at ' + ip)
                return 'http://' + ip + ':8000'
        except:
            pass
    raise Exception('Could not find ' + server_type + ' server')

class RubiksCube:
    def __init__(self):
        self.url = find_ip('master')
        self.moves_list = ['right', 'left', 'top', 'bottom', 'front', 'back', 'right_rev', 'left_rev', 'top_rev', 'bottom_rev', 'front_rev', 'back_rev']

    def _process_label(self, label):
        """
        Process the labels such that the positions of the labels are consistent with the positions of the images.
        """
        label_1 = label[[0, 1, 4]]
        label_2 = label[[2, 3, 5]]
        label_2_ = copy.deepcopy(label_2)
        label_2[0] = np.rot90(label_2_[1], 2)
        label_2[1] = np.rot90(label_2_[0], 2)
        label_2[2] = np.rot90(label_2_[2], 1)
        return label_1, label_2
    
    def _send_request(self, move=None):
        """
        Send a request to the API to rotate the cube and get the image and label
        Two images and labels are received via the API in pickle format
        If move is None, rotate the cube randomly otherwise rotate the cube according to the move
        """
        assert move in self.moves_list or move is None, 'Invalid move. Valid moves are: ' + str(self.moves_list) + ' or None'

        if move is None:
            move = random.choice(self.moves_list)

        r = requests.post(self.url, data=move)
        img_1, img_2, label = pickle.loads(r.content)
        label_1, label_2 = self._process_label(label)
        return img_1[:, :, ::-1], img_2[:, :, ::-1], label_1, label_2

    def get_imgs(self):
        """
        Get get the images and labels from the API without moving the cube
        """

        r = requests.get(self.url + 'get_images/')
        img_1, img_2 = pickle.loads(r.content)
        return img_1[:, :, ::-1], img_2[:, :, ::-1]
    
    def get_state(self):
        """
        Get the state of the cube
        """
        r = requests.get(self.url + 'get_cube_state/')
        state = pickle.loads(r.content)
        return state