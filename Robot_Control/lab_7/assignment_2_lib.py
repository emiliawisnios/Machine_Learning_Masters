import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
# p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def build_world_with_car():
  pos = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
  p.resetSimulation()
  p.setGravity(0, 0, -10)
  p.loadURDF("plane.urdf")
  car = p.loadURDF("racecar/racecar.urdf")
  p.resetBasePositionAndOrientation(car, pos[0], pos[1])
  return car

def simulate_car(car, steeringAngle = 0.2, targetVelocity = -2, steps=5000):
  wheels = [2, 3, 5, 7]
  steering = [4, 6]
  maxForce = 10
  for wheel in wheels:
    p.setJointMotorControl2(car,
                            wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=targetVelocity,
                            force=maxForce)
  for steer in steering:
    p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)
  for i in range(steps):
     p.stepSimulation() 
  return p.getBasePositionAndOrientation(car)

def drive(car, forward, direction):
  if forward:
    speed = 2
  else:
    speed = -2
  if direction < 0:
    steeringAngle = -0.45
  elif direction > 0:
    steeringAngle = 0.45
  else:
    steeringAngle = 0
  simulate_car(car, steeringAngle, speed, 250)

def take_a_photo(car, debug=False):
  pos = p.getBasePositionAndOrientation(car)
  orn = p.getQuaternionFromEuler([0, 0, 0])
  other_pos = [[20,0,0], orn]
  combined_pos = p.multiplyTransforms(pos[0], pos[1], other_pos[0], other_pos[1])
  pos = list(pos[0])
  pos[2] += 0.22
  _, _, rgb, _, _ = p.getCameraImage(640, 640,
    viewMatrix=p.computeViewMatrix(pos, combined_pos[0], [0,0,1]),
    projectionMatrix=p.computeProjectionMatrixFOV(45, 1, 0.1, 10))
  return rgb
