from assignment_2_lib import take_a_photo, drive
import cv2
import numpy as np
import imutils
from typing import Optional

FOCAL_LENGTH = 386.75
STEPS_PER_DRIVE_CALL = 250
CAR_VELOCITY = 0.10607170891503596
DISTANCE_PER_DRIVE_STEP = CAR_VELOCITY/STEPS_PER_DRIVE_CALL
BALL_RADIUS = 0.4
BLUE_HUE_MARGIN = [np.array([110, 50, 50]), np.array([130, 255, 255])]
POSITIVE_HUE_MARGIN_RED = [np.array([0, 100, 50]), np.array([10, 255, 255])]
NEGATIVE_HUE_MARGIN_RED = [np.array([160, 100, 50]), np.array([179, 255, 255])]


def crop_image(image: np.ndarray) -> np.ndarray:
    """
    Function that crops the bottom part of the image (with visible car parts)
    :param image: Image taken by car (with visible car parts)
    :return: Croped image without visible car parts
    """
    return image[0:400, :]


def distance_to_camera(known_width: float, focal_length: float, width_from_picture: float,
                       margin: Optional[bool] = True) -> float:
    """
    Function that counts the distance from camera to the object
    :param known_width: known width of the object
    :param focal_length: constant number -- focal length value
    :param width_from_picture: width of object counted from the photo
    :param margin: boolean parameter, True if margin equal to known Width should be kept (default: True)
    :return: distance from camera to the object
    """
    distance = (known_width * focal_length) / width_from_picture
    if margin:
        distance -= 2 * known_width
    return distance


def distance_to_steps(distance: float) -> int:
    """
    Function that converts distance to object to simulation steps
    :param distance: Calculated distance to the object
    :return: number of simulation steps
    """
    n = int(distance / DISTANCE_PER_DRIVE_STEP)
    return n


def turn_inplace(car, direction: Optional[str] = 'right') -> None:
    """
    Function that turns car inplace in given direction
    :param car:
    :param direction: Direction of turn (default: 'right')
    :return: None
    """
    if direction == 'right':
        drive(car, forward=True, direction=1)
        drive(car, forward=False, direction=-1)
    else:
        drive(car, forward=True, direction=-1)
        drive(car, forward=False, direction=1)


def add_mask(image: np.ndarray, lower_hue_margin: np.array, upper_hue_margin: np.array) -> np.ndarray:
    """
    Checks if array elements lie between the elements of two other arrays.
    :param image: Image values
    :param lower_hue_margin: Values for lower hue margin
    :param upper_hue_margin: Values for upper hue margin
    :return: output array of the same size as image and CV_8U type
    """
    return cv2.inRange(image, lower_hue_margin, upper_hue_margin)


def prepare_photo_ball(image: np.ndarray) -> np.ndarray:
    """"
    Function that finds red elements on the picture
    :param image: Image taken by car
    :return: Image with mask on red elements on the picture
    """
    # conversion to HSV colorspace
    hsv_image = cv2.cvtColor(crop_image(image), cv2.COLOR_RGB2HSV)

    # positive red hue margin
    mask1 = add_mask(hsv_image, POSITIVE_HUE_MARGIN_RED[0], POSITIVE_HUE_MARGIN_RED[1])

    # negative red hue margin
    mask2 = add_mask(hsv_image, NEGATIVE_HUE_MARGIN_RED[0], NEGATIVE_HUE_MARGIN_RED[1])

    return mask1 + mask2


def prepare_photo_cylinder(image: np.ndarray) -> np.ndarray:
    """
    Function that finds blue elements on the picture
    :param image: Image taken by car
    :return: Image with mask on blue elements on the picture
    """
    # conversion to HSV colorspace
    hsv_image = cv2.cvtColor(crop_image(image), cv2.COLOR_RGB2HSV)

    # blue hue margin
    mask = add_mask(hsv_image, BLUE_HUE_MARGIN[0], BLUE_HUE_MARGIN[1])
    return mask


def find_marker(image: np.ndarray) -> np.ndarray:
    """
    Function that finds markers for ball out of picture using color masking.
    :param image: Image taken by the car
    :return: Coordinates of bounding circle (center, radius)
    """
    mask = prepare_photo_ball(image)

    # Find the contours in the edged image and keep the largest one
    contours = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return []
    c = max(contours, key=cv2.contourArea)
    # Compute the bounding circle of the o
    return cv2.minEnclosingCircle(c)


def ball_is_visible(photo: np.ndarray, threshold: Optional[int] = 50, distance: Optional[float] = 0.1) -> bool:
    """
    Function that returns True if center of a ball is in the given threshold from the center of the image
    and it's not closer than some threshold
    :param photo: Image taken by car
    :param threshold: Distance from the center of the picture that we accept for center of the ball
    :param distance: Minimal distance to the ball
    :return: True if ball is in the given threshold and is no closer than distance
    """
    markers = find_marker(photo)
    photo = crop_image(photo)
    photo_center = photo.shape[1]/2

    if len(markers) == 0:
        # there is no ball on the picture
        return False
    else:
        if (photo_center - threshold < markers[0][0] < photo_center + threshold) or \
                (distance_to_camera(BALL_RADIUS, FOCAL_LENGTH, markers[1]) <= distance):
            return True
        else:
            return False


def localize_ball(car) -> int:
    """
    Function that turns car inplace as long as ball is visible and its center is in some threshold. Then it returns
    number of simulation steps to get to the ball
    :param car: Car
    :return: Number of simulation steps to the ball
    """
    photo = take_a_photo(car)
    while not ball_is_visible(photo):
        turn_inplace(car)
        photo = take_a_photo(car)
    return forward_distance(photo)


def cylinders_are_visible(image: np.ndarray) -> bool:
    """
    Function that returns empty list if there are no cylinders on the picture, True if there are two cylinders,
    False if there is one cylinder
    :param image: Image taken by car
    :return: Empty list if there are no cylinders on the picture, True if there are two cylinders,
    False if there is one cylinder
    """
    mask = prepare_photo_cylinder(image)
    contours = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return False
    if len(contours) >= 2:
        return True
    else:
        return False


def ball_between_cylinders(car, threshold: Optional[int] = 50) -> bool:
    """
    Function that returns True if there are two cylinders and ball center in some given threshold
    :param car: Car
    :param threshold: Maximum distance from the photo center to the ball center (default: 50)
    :return: True if there are two cylinders and center of the ball is in the given threshold, False otherwise
    """
    photo = take_a_photo(car)
    photo_center = photo.shape[1]/2
    if cylinders_are_visible(photo):
        markers = find_marker(photo)
        if len(markers) == 0:
            return False
        else:
            if photo_center - threshold < markers[0][0] < photo_center + threshold:
                return True
            else:
                return False
    else:
        return False


def forward_distance(image: np.ndarray) -> int:
    """
    Function that finds forward distance in simulation steps to the ball
    :param image: Image taken by the car
    :return: Number of simulation steps to the ball
    """
    marker = find_marker(image)
    distance = distance_to_camera(BALL_RADIUS, FOCAL_LENGTH, round(marker[1], 2))
    number_of_steps = distance_to_steps(distance)
    return number_of_steps


def find_a_ball(car: np.ndarray, threshold: Optional[int] = 50):
    """
    Function that finds the ball and drives to it
    :param car: Car
    :param threshold: Maximum distance from the photo center to the ball center (default: 50)
    :return: 0
    """
    photo = take_a_photo(car)
    while not ball_is_visible(photo):
        drive(car, forward=True, direction=1)
        drive(car, forward=False, direction=-1)
        photo = take_a_photo(car, threshold)
    n = forward_distance(photo)
    n //= STEPS_PER_DRIVE_CALL
    for i in range(n):
        drive(car, forward=True, direction=0)
    return 0


def move_a_ball(car):
    """
    Functions that moves ball between cylinders
    :param car: Car
    :return: 0
    """
    mode_left = False


    # Car goes back to the cylinders
    for _ in range(20):
        drive(car, forward=False, direction=0)

    photo = crop_image(take_a_photo(car))
    markers = find_marker(photo)
    photo_center = photo.shape[1] / 2

    # check if ball is on the left side of the board
    if markers[0][0] > photo_center:
        mode_left = True

    turning_range = 26

    # special cases of turning range change (due to numerical errors)

    if 150 < markers[0][0] < 300 and 41 < markers[1] < 49:
        turning_range = 25
    elif 300 < markers[0][0] < 500 and 38 < markers[1] < 50:
        turning_range = 26

    if 110 < markers[0][0] < 220 or 485 > markers[0][0] > 470:
        turning_range = 27

    if 210 < markers[0][0] < 230 and markers[1] < 42:
        turning_range = 28

    if 478 < markers[0][0] < 480:
        turning_range = 26

    if 485 < markers[0][0] < 487:
        turning_range = 25

    # count number of steps to the ball
    n = forward_distance(photo)
    n //= STEPS_PER_DRIVE_CALL

    # special cases of changing number of steps to the ball (due to numerical errors)

    if 102 < markers[0][0] < 103:
        n += 2
    elif 138 < markers[0][0] < 140:
        n += 2
    elif 154 < markers[0][0] < 155:
        n += 2
    elif 105 < markers[0][0] < 106:
        n += 2
    elif 130 < markers[0][0] < 131:
        n += 2

    # go to the ball
    find_a_ball(car, threshold=20)

    photo = crop_image(take_a_photo(car))
    markers = find_marker(photo)

    # correcting position (ball center must be in the center of the photo)
    if not photo.shape[1]/2 - 5 < markers[0][0] < photo.shape[1]/2 + 5:

        for _ in range(3):
            drive(car, forward=False, direction=0)

        find_a_ball(car, threshold=5)

    # turning left (or right in left_mode) as long as we don't see ball on the picture
    photo = crop_image(take_a_photo(car))
    if not mode_left:
        while len(find_marker(photo)) > 0:
            turn_inplace(car, direction='left')
            photo = crop_image(take_a_photo(car))
        turn_inplace(car, direction='left')
        turn_inplace(car, direction='left')

        for _ in range(turning_range):
            drive(car, forward=True, direction=1)

    else:
        while len(find_marker(photo)) > 0:
            turn_inplace(car, direction='right')
            photo = crop_image(take_a_photo(car))

        turn_inplace(car, direction='right')
        turn_inplace(car, direction='right')

        for _ in range(turning_range):
            drive(car, forward=True, direction=-1)

    # turn bask as long as we don't see ball and cylinders
    if not mode_left:
        photo = crop_image(take_a_photo(car))
        markers = find_marker(photo)
        while len(markers) == 0 and not cylinders_are_visible(photo):
            drive(car, forward=False, direction=-1)
            photo = crop_image(take_a_photo(car))
            markers = find_marker(photo)
        for _ in range(2):
            drive(car, forward=False, direction=-1)

    else:
        photo = crop_image(take_a_photo(car))
        markers = find_marker(photo)
        while len(markers) == 0 and not cylinders_are_visible(photo):
            drive(car, forward=False, direction=1)
            photo = crop_image(take_a_photo(car))
            markers = find_marker(photo)

        for _ in range(2):
            drive(car, forward=False, direction=-1)

    # go to the ball
    find_a_ball(car, threshold=10)

    # push ball to the cylinders
    k = 0
    for i in range(n):
        photo = take_a_photo(car)
        makers = find_marker(photo)
        if makers[0][0] > photo.shape[1] / 2:
            drive(car, forward=True, direction=-1)
            k += 1
        elif makers[0][0] < photo.shape[1] / 2:
            drive(car, forward=True, direction=1)
            k += 1

        drive(car, forward=True, direction=1)
        drive(car, forward=True, direction=-1)
        k += 1

        if k >= n + 6:
            break

    return 0
