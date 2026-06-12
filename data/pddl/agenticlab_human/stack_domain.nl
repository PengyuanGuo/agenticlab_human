The robot has two actions: pick and place. The domain assumes a world where objects (cubes, fruits, etc.) are always resting on top of other objects (like mats, plates, or other cubes). There is no explicit "table" surface; instead, base objects like mats or plates serve as the foundation.

The actions defined in this domain include:
pick: allows the arm to pick up an object from its current support (e.g., a mat or another cube). The robot must be empty-handed, and the object must be clear (nothing on top of it). After picking, the robot holds the object, the object is no longer on its support, and the support becomes clear.

place: allows the arm to place a held object at a target. The target must be clear. After placing, the robot is empty-handed, the object is on top of the target, the object becomes clear, and the target is no longer clear.
