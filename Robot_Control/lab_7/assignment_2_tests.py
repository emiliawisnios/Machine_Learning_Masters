from assignment_2_lib import *
from assignment_2_solution import *
import random
from scipy.spatial import distance

NUMBER_OF_TESTS = 100

def test_forward_distance():
  for seed in range(NUMBER_OF_TESTS):
    random.seed(seed)
    car = build_world_with_car()
    pos_1 = [3 * random.random() + 2, 0, 1]
    ball = p.loadURDF("sphere2red.urdf", pos_1, globalScaling=0.4)
    for i in range(200):
      p.stepSimulation()
    photo = take_a_photo(car)
    dist = forward_distance(photo)
    ball_start_pos = p.getBasePositionAndOrientation(ball)[0]
    simulate_car(car, 0, 2, dist)
    # should be less than 1
    car_ball = distance.euclidean(p.getBasePositionAndOrientation(car)[0], 
          p.getBasePositionAndOrientation(ball)[0])
    assert car_ball < 1, car_ball
    # should be less than 0.1
    ball_move = distance.euclidean(ball_start_pos, p.getBasePositionAndOrientation(ball)[0])
    assert ball_move < 0.1, ball_move

print("testing forward")
test_forward_distance()


def test_find_a_ball():
  for seed in range(NUMBER_OF_TESTS):
    random.seed(seed)
    print("seed", seed)
    car = build_world_with_car()
    pos_1 = [0, 0, 1]
    while distance.euclidean(pos_1, [0, 0, 1]) < 1:
      pos_1 = [6 * random.random() - 3, 6 * random.random() - 3, 1]
    ball = p.loadURDF("sphere2red.urdf", pos_1, globalScaling = 0.4)
    for i in range(200):
      p.stepSimulation()
    ball_start_pos = p.getBasePositionAndOrientation(ball)[0]
    find_a_ball(car)
    # should be less than 1
    car_ball = distance.euclidean(p.getBasePositionAndOrientation(car)[0], 
          p.getBasePositionAndOrientation(ball)[0])
    assert car_ball < 1, car_ball
    # should be less than 0.1
    ball_move = distance.euclidean(ball_start_pos, p.getBasePositionAndOrientation(ball)[0])
    assert ball_move < 0.1, ball_move

print("testing find a ball")
test_find_a_ball()

def test_move_a_ball():

  for seed in range(NUMBER_OF_TESTS):
    random.seed(seed)
    print("seed", seed)
    
    pos_1 = [1 + random.random(), 2 * random.random() - 1, 1]
    car = build_world_with_car()
    print(type(car))
    ball = p.loadURDF("sphere2red.urdf", pos_1, globalScaling = 0.4)
    s = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
     visualFramePosition=[-2,-1,0], rgbaColor=[0, 0, 1, 1],
     radius=0.1, length=10)
    p.createMultiBody(baseVisualShapeIndex=s)
    s = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
     visualFramePosition=[-2,1,0], rgbaColor=[0, 0, 1, 1],
     radius=0.1, length=10)
    p.createMultiBody(baseVisualShapeIndex=s)

    for i in range(200):
      p.stepSimulation()
    
    move_a_ball(car)
    pos = p.getBasePositionAndOrientation(ball)[0]
    assert -3 < pos[0] < -2 and -1 < pos[1] < 1


test_move_a_ball()
