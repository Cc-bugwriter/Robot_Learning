from jointCtlComp import *
from taskCtlComp import *

# Controller in the joint space. The robot has to reach a fixed position.
# jointCtlComp(['P'], True)

# Same controller, but this time the robot has to follow a fixed trajectory.
# jointCtlComp(['P'], False)

# Controller in the task space.
taskCtlComp(['JacNullSpace'], pauseTime=True, resting_pos=np.mat([0, -pi]).T)

input('Press Enter to close')