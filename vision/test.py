import numpy as np
arr = np.random.randint(1, 20, size=(3, 3))
print("\norignal array:")
print(arr)
arr[0, 2] = 6
arr[1, 1] = 8
arr[2, 0] = 4
print("\nmodifies array:")
print(arr)

""""
print(cv2.__version__)
print(hasattr(cv2, "legacy"))

print(cv2.legacy.TrackerCSRT_create())
"""