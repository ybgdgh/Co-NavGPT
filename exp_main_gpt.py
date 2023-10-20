from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from agents.llm_agents import LLM_Agent
from envs.habitat.multi_agent_env import Multi_Agent_Env
from constants import color_palette, coco_categories, hm3d_category, category_to_id
import utils.visualization as vu
from arguments import get_args

from skimage import measure
import skimage.morphology
from PIL import Image

import cv2
import habitat
import openai
from openai.error import OpenAIError

import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

system_prompt = """Objective: Control two robots equipped with cameras and 5-meter range depth sensors. They must collaboratively explore an indoor home scene on a 2D map to locate a specific target object.

Map Information:
Origin: Top-left corner.
Coordinates: Represented in pixels.

Representation Details:
Robot Position: Format - (x, y).
Scene Objects: Defined by vertices in clockwise order. Format - <(x1, y1), (x2, y2)...>.
Walls: Lines with start and end points. Format - <(x1_start, y1_start, x1_end, y1_end)>.
Frontiers: Defined by centroid and pixel count. Format - <centroid:(x, y), number: n>.

Strategy Considerations:
Robots share observed data and choose a frontier to explore.
Assign robots based on the relationships between different objects, the structure of the explored areas robots' positions, frontier proximity, and their previous movement directions.
Minimize frequent switches between frontiers. Use centroid for frontier selection.
A robot should maintain its exploration direction unless an efficient switch is evident.

Instructions: Given the scene details, decide the best frontier each robot should explore next. Provide ONLY the decision in the [output:] format without additional explanations or additional text.

Example: 

[input:]
Task: locate the chairs

Position: 
robot_0: (240, 240)
robot_1: (200, 150) 

Scene Objects:
sofa: <(280, 200), (280, 150), (330, 150), (330, 180), (300, 180), (300, 200)> 
bed: <(220, 250), (240, 250), (250, 250), (250, 220)> 
chest_of_drawers: <(200, 240), (210, 240), (210, 250), (200, 250)> 
tv_monitor: <(220, 200), (240, 200), (240, 210), (220, 210)> 
table: <(200, 150), (230, 150), (230, 170), (200, 170)> 

Walls:
wall_0: <(190, 180, 300, 180)> 
wall_1: <(160, 180, 160, 250)> 
wall_2: <(160, 250, 30, 250)> 

Previous Movements:
robot_0: <centroid:(195, 320), number: 60>
robot_1: <centroid:(170, 180), number: 30> 

Unexplored Frontier: 
frontier_0: <centroid:(180, 175), number: 40> 
frontier_1: <centroid:(195, 280), number: 80>

[output:]
robot_0: frontier_1
robot_1: frontier_0 

Please give the output based on the following input:\n"""

openai.organization = "xxx"
openai.api_key = "xxx" 
openai.Model.list()
# GPT Type
gpt_name = [
    'text-davinci-003',
    'gpt-3.5-turbo',
    'gpt-4',
]

def Visualize(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, goal_points):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, l_step)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    no_cat_mask = sem_map == 20
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i


    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)

    vis_image[50:530, 15:495] = sem_map_vis

    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)

    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Vis-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        cv2.imwrite(fn, vis_image)

def Frontiers(full_map_pred):
    # ------------------------------------------------------------------
    ##### Get the frontier map and filter
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    full_w = full_map_pred.shape[1]
    local_ex_map = np.zeros((full_w, full_w))
    local_ob_map = np.zeros((full_w, full_w))

    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(local_ex_map,contour,-1,1,1)

    # clear the boundary
    local_ex_map[0:2, 0:full_w]=0.0
    local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
    local_ex_map[0:full_w, 0:2]=0.0
    local_ex_map[0:full_w, full_w-2:full_w]=0.0

    target_edge = local_ex_map-local_ob_map
    # print("local_ob_map ", self.local_ob_map[200])
    # print("full_map ", self.full_map[0].cpu().numpy()[200])

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    wall_edge = local_ex_map - target_edge

    # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     dst = np.zeros(wall_edge.shape)
    #     cv2.drawContours(dst, contours, -1, 1, 1)

    # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)
    Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

    # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_area_list = []
    dict_cost = {}
    for i in range(1, len(props)):
        if props[i].area > 4:
            dict_cost[i] = props[i].area

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        for i, (key, value) in enumerate(dict_cost):
            Goal_edge[img_label == key + 1] = 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_area_list.append(value)
            if i == 3:
                break
        # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if frontiers is not None:
        #     for frontier in frontiers:
        #         x1, y1, x2, y2 = frontier[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return Wall_lines, Goal_area_list, Goal_edge, Goal_point

def Objects_Extract(full_map_pred):

    semantic_map = full_map_pred[4:]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    for i in range(len(semantic_map)):
        if semantic_map[i, :, :].sum() != 0:
            Single_object_list = []
            se_object_map = semantic_map[i, :, :].cpu().numpy()
            se_object_map[se_object_map>0.1] = 1
            se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if len(cnt) > 30:
                    epsilon = 0.05 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    Single_object_list.append(approx)
                    cv2.polylines(dst, [approx], True, 1)
            if len(Single_object_list) > 0:
                Object_list[hm3d_category[i]] = Single_object_list

    return Object_list

def form_prompt_for_chatgpt(goal_name, pose_pred, object_list, Wall_list, Frontier_list, last_decision, Frontier_points):

    Robot_Position = "\n ".join([f"robot_{i}: {pose_pred[i][0], pose_pred[i][1]}" 
                        for i in range(len(pose_pred))])

    for key, value in object_list.items():
        value = ", ".join([f"{value[i]}" 
                        for i in range(len(value))])
    Objects_Position = "\n ".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in object_list.items()]) + "\n"

    if Wall_list is not None:
        Walls_Position = "\n ".join([f"wall_{i}: <" +  ", ".join([f"{Wall_list[i][0][j]}"  
                            for j in range(len(Wall_list[i][0]))]) + f">"
                            for i in range(len(Wall_list))]) + "\n" 
    else: 
        Walls_Position = None

    Frontiers = "\n ".join([f"frontier_{i}: <centroid: {Frontier_points[i][0], Frontier_points[i][1]}, number: {Frontier_list[i]}>"  
                        for i in range(len(Frontier_points))])

    if len(last_decision) > 0:
        Last_Decision = "\n ".join([f"robot_{i}: {last_decision[i]}" 
                        for i in range(len(last_decision))])
    else:
        Last_Decision = "No frontiers"

    prompt_template = """
    [input:]
    Task: Locate the {GOAL_NAME}

    Position: 
    {ROBOT_POSITION}

    Scene Objects:
    {OBJECTS_POSITION}

    Walls:
    {WALLS_POSITION}

    Previous Movements:
    {LAST_DECISION}

    Unexplored Frontier: 
    {FRONTIERS}
                
    [output:] """

    User_prompt = prompt_template.format(
                    GOAL_NAME = str(goal_name),
                    ROBOT_POSITION = Robot_Position,
                    OBJECTS_POSITION = Objects_Position,
                    WALLS_POSITION = Walls_Position,
                    FRONTIERS = Frontiers,
                    LAST_DECISION = Last_Decision
                )

    Frontiers_dict = {}
    for i in range(len(Frontier_points)):
        Frontiers_dict['frontier_' + str(i)] = f"<centroid: {Frontier_points[i][0], Frontier_points[i][1]}, number: {Frontier_list[i]}>"

    return User_prompt, Frontiers_dict

def parse_answer(response_message):
    lines = response_message.split('\n')

    parsed_dict_num = {}

    for line in lines:
        key, value = line.split(': ')
        parsed_dict_num[key.strip()] = int(value.split('_')[1])

    return parsed_dict_num

@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )

        return config

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")

    config_env = habitat.get_config(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    config_env.defrost()

    config_env.TASK.POSSIBLE_ACTIONS = config_env.TASK.POSSIBLE_ACTIONS + [
        "TURN_LEFT_S",
        "TURN_RIGHT_S",
    ]
    config_env.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config_env.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"
    config_env.freeze()


    env = Multi_Agent_Env(config_env=config_env)

    num_episodes = env.number_of_episodes

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config_env.SIMULATOR.NUM_AGENTS
    agent = []
    for i in range(num_agents):
        agent.append(LLM_Agent(args, i))


    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'output.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    # print(args)
    logging.info(args)
    # ------------------------------------------------------------------


    device = torch.device("cuda:0" if args.cuda else "cpu")

    agg_metrics: Dict = defaultdict(float)

    count_episodes = 0
    count_step = 0
    goal_points = []
    log_start = time.time()
    last_decision = []
    total_usage = []

    while count_episodes < num_episodes:
        observations = env.reset()
        for i in range(num_agents):
            agent[i].reset()

        while not env.episode_over:
            action = [0, 0]
            full_map = []
            visited_vis = []
            pose_pred = []
            start = time.time()
            for i in range(num_agents):
                agent[i].mapping(observations[i])
                full_map.append(agent[i].local_map)
                visited_vis.append(agent[i].visited_vis)
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / agent[i].visited_vis.shape[0],
                    (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / agent[i].visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
                pose_pred.append(pos)
                
            full_map2 = torch.cat((full_map[0].unsqueeze(0), full_map[1].unsqueeze(0)), 0)

            full_map_pred, _ = torch.max(full_map2, 0)

            # mapping_end = time.time()
            # mapping_time = mapping_end - start
            # print('mapping_time: %.3f秒'%mapping_time)

            if agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0:
                goal_points.clear()

                Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred)

                if len(target_point_map) > 0:
                    object_list = Objects_Extract(full_map_pred)

                    User_prompt, Frontiers_dict = form_prompt_for_chatgpt(agent[0].goal_name, 
                                                            pose_pred, 
                                                            object_list,
                                                            Wall_list, 
                                                            Frontier_list, 
                                                            last_decision,
                                                            target_point_map)

                    message_list=[]
                    message_list.append({"role": "system", "content": system_prompt})
                    message_list.append({"role": "user", "content": User_prompt})

                    retries = 10    
                    while retries > 0:  
                        try: 
                            response = openai.ChatCompletion.create(
                                model=gpt_name[args.gpt_type], 
                                messages=message_list,
                                temperature=0,
                                request_timeout=15,
                            )

                            response_message = response["choices"][0]["message"]["content"]
                            usage = 0
                            print(gpt_name[args.gpt_type] + " response: ")
                            print(response_message)
                            if gpt_name[args.gpt_type] == 'gpt-4':
                                usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage']['completion_tokens'] * 0.06 / 1000
                            elif gpt_name[args.gpt_type] == 'gpt-3.5-turbo':
                                usage = response['usage']['total_tokens'] * 0.002 / 1000
                            total_usage.append(usage)
                            goal_frontiers = parse_answer(response_message)

                            last_decision.clear()
                            for i in range(num_agents):
                                goal_points.append(target_point_map[goal_frontiers["robot_"+ str(i)]])

                                last_decision.append(Frontiers_dict["frontier_"+str(goal_frontiers["robot_"+ str(i)])] ) 
                            
                            break
                        except OpenAIError as e:
                            if e:
                                print(e)
                                print('Timeout error, retrying...')    
                                retries -= 1
                                time.sleep(5)
                            else:
                                raise e
                else:
                    for i in range(num_agents):
                        actions = np.random.rand(1, 2).squeeze()*(target_edge_map.shape[0] - 1)

                        goal_points.append([int(actions[0]), int(actions[1])])

            # start_act = time.time()
            for i in range(num_agents):
                action[i] = agent[i].act(goal_points[i])
            # act_end = time.time()
            # act_time = act_end - start_act
            # print('act_time: %.3f秒'%act_time)


            observations = env.step(action)
            # step_end = time.time()
            # step_time = step_end - act_end
            # print('step_time: %.3f秒'%step_time)


            if args.visualize or args.print_images: 
                Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                            agent[0].goal_id, visited_vis, target_edge_map, goal_points)

        
        count_episodes += 1
        count_step += agent[0].l_step

        # ------------------------------------------------------------------
        ##### Logging
        # ------------------------------------------------------------------
        log_end = time.time()
        time_elapsed = time.gmtime(log_end - log_start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_step),
            "FPS {},".format(int(count_step / (log_end - log_start)))
        ]) + '\n'

        metrics = env.get_metrics()
        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v

        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

        log += "Total usage: " + str(sum(total_usage)) + ", average usage: " + str(np.mean(total_usage))
        print(log)
        logging.info(log)
        # ------------------------------------------------------------------


    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    return avg_metrics

if __name__ == "__main__":
    main()
