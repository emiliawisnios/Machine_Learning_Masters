#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import time
import random

green = [0, 230, 0, 1]
pink = [1, 0.5, 0.7, 1]


def sgn(number):
    if number >= 0:
        return 1
    else:
        return -1


def set_cube_color(cube, color=None):
    if color is None:
        color = pink
    p.changeVisualShape(cube, -1, rgbaColor=color)


def update_cubes(cubes, state_of_cubes, wanted_area):
    for i in range(len(cubes)):
        position, orientation = p.getBasePositionAndOrientation(cubes[i])
        if abs(position[0]) < wanted_area and abs(position[1]) < wanted_area:
            set_cube_color(cubes[i], green)
            # False == in the wanted area
            # True == not in the wanted area
            state_of_cubes[i] = False
        else:
            state_of_cubes[i] = True


def prepare_world(number_of_cubes, state_of_cubes, cubes):
    # start the simulation with a GUI (p.DIRECT is without GUI)
    p.connect(p.GUI)

    # we can load plane and cube from pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # load a plane
    p.loadURDF("plane.urdf", [0, 0, - 0.1], useFixedBase=True)

    # setup gravity (without it there is no gravity at all)
    p.setGravity(0, 0, -10)

    # load our robot definition
    robot = p.loadURDF("robot.urdf")

    # load a cube
    for i in range(number_of_cubes):
        cube = p.loadURDF("cube.urdf", [random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9), 0.1], globalScaling=0.05)
        p.changeVisualShape(cube, -1, rgbaColor=pink)
        cubes.append(cube)
        # if state is equal to True cube is not in wanted area
        state_of_cubes.append(True)

    # display info about robot joints
    numJoints = p.getNumJoints(robot)
    for joint in range(numJoints):
        print(p.getJointInfo(robot, joint))

    return robot


def main():
    wanted_area = 0.25
    number_of_cubes = 5
    cubes = []
    state_of_cubes = []
    robot = prepare_world(number_of_cubes, state_of_cubes, cubes)

    # robot states = {0: prepare to push horizontally,
    #                 1: go down
    #                 2: push horizontally,
    #                 3: go up,
    #                 4: prepare to push vertically,
    #                 5: push vertically}

    robot_state = 0
    freq = 0
    push_freq = 0
    i = 0
    while True:
        # step Simulation
        p.stepSimulation()
        update_cubes(cubes, state_of_cubes, wanted_area)
        if freq % 150 == 0:
            if state_of_cubes[i]:
                position, orientation = p.getBasePositionAndOrientation(cubes[i])
                if robot_state == 0:
                    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, 0.2)
                    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, position[1])
                    p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, position[0] + sgn(position[0]) * 0.1)
                    robot_state += 1
                elif robot_state == 1:
                    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, position[2])
                    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, position[1])
                    p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, position[0] + sgn(position[0]) * 0.1)
                    robot_state += 1
                elif robot_state == 2:
                    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, position[2])
                    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, position[1])
                    p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, 0, maxVelocity=1.5)
                    robot_state += 1
                elif robot_state == 3:
                    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, 0.2)
                    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, position[1])
                    p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, position[0])
                    robot_state += 1
                elif robot_state == 4:
                    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, position[2])
                    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, position[1] + sgn(position[1]) * 0.1)
                    p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, position[0])
                    robot_state += 1
                elif robot_state == 5:
                    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, position[2])
                    p.setJointMotorControl2(robot, 1, p.POSITION_CONTROL, 0, maxVelocity=1.5)
                    p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, position[0])
                    robot_state = 0
                    push_freq += 1
            else:
                if i == number_of_cubes - 1:
                    break
                else:
                    freq = -1
                    i += 1
        freq += 1
        time.sleep(0.01)  # sometimes pybullet crashes, this line helps a lot

    print('You won!')


main()
