import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat

from constants import category_to_id, mp3d_category_id
import utils.pose as pu

coco_categories = [0, 3, 2, 4, 5, 1]

class Multi_Agent_Env(habitat.Env):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, config_env):

        super().__init__(config_env)

        # Initializations
        self.episode_no = 0
   
        fileName = 'data/matterport_category_mappings.tsv'

        text = ''
        lines = []
        items = []
        self.hm3d_semantic_mapping={}

        with open(fileName, 'r') as f:
            text = f.read()
        lines = text.split('\n')

        for l in lines:
            items.append(l.split('    '))

        for i in items:
            if len(i) > 3:
                self.hm3d_semantic_mapping[i[2]] = i[-1]

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        self.episode_no += 1

        obs = super().reset()
        self.scene = self.sim.semantic_annotations()
  
        for i in range(len(obs)):
            obs[i]['semantic'] = self._preprocess_semantic(obs[i]["semantic"])

        return obs

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        obs = super().step(action)


        for i in range(len(obs)):
            obs[i]['semantic'] = self._preprocess_semantic(obs[i]["semantic"])

        return obs

    def _preprocess_semantic(self, semantic):
        # print("*********semantic type: ", type(semantic))
        se = list(set(semantic.ravel()))
        # print(se) # []
        for i in range(len(se)):
            if se[i] >= len(self.scene.objects):
                hm3d_category_name = "Unknown"
            elif self.scene.objects[se[i]].category.name() in self.hm3d_semantic_mapping:
                hm3d_category_name = self.hm3d_semantic_mapping[self.scene.objects[se[i]].category.name()]
            else:
                hm3d_category_name = self.scene.objects[se[i]].category.name()

            if hm3d_category_name in mp3d_category_id:
                # print("sum: ", np.sum(sem_output[sem_output==se[i]])/se[i])
                semantic[semantic==se[i]] = mp3d_category_id[hm3d_category_name]-1
            else :
                semantic[
                    semantic==se[i]
                    ] = 0
    
        # se = list(set(semantic.ravel()))
        # print("semantic: ", se) # []
        semantic = np.expand_dims(semantic.astype(np.uint8), 2)
        return semantic

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
