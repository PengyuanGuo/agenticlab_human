"""Save Flexiv robot EEF poses for hand-eye calibration.

Output CSV format (same as save_poses_ur5.py):
    pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w

This is directly consumed by compute_in_hand.py / compute_to_hand.py
which call  R.from_quat(pose[3:7])  (scipy xyzw order).
"""

import time
import csv
import sys
import os
from datetime import datetime

import flexivrdk
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Configuration ---
ROBOT_SN = "Rizon4s-063239"
# ROBOT_SN = "Rizon4s-063215"
LOCAL_IP_WHITELIST = ["192.168.100.2"]


def save_pose_to_csv(pose_data, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["pos_x", "pos_y", "pos_z",
                             "quat_x", "quat_y", "quat_z", "quat_w"])
        writer.writerow(pose_data)
    print(f"Pose saved to {filename}: {[f'{v:.6f}' for v in pose_data]}")


def main():
    robot = None
    pose_count = 0
    csv_filename = None

    try:
        print(f"Connecting to Flexiv robot [{ROBOT_SN}] …")
        robot = flexivrdk.Robot(ROBOT_SN, LOCAL_IP_WHITELIST)

        if robot.fault():
            print("Clearing fault …")
            robot.ClearFault()
            time.sleep(1)

        print("Enabling robot …")
        robot.Enable()
        while not robot.operational():
            time.sleep(0.5)
        print("Robot is operational.")

        # Configure tool parameters so floatingSoft uses the right payload model.
        robot.SwitchMode(flexivrdk.Mode.IDLE)
        tool = flexivrdk.Tool(robot)

        # print("All configured tools:")
        # tool_list = tool.list()
        # for i in range(len(tool_list)):
        #     print(f"[{i}] {tool_list[i]}")
        # print()
        # print(f"Current active tool: [{tool.name()}]")

        # new_tool_name = "CheckerBoard"
        # new_tool_params = flexivrdk.ToolParams()
        # new_tool_params.mass = 1.0
        # new_tool_params.CoM = [0.0, 0.0, 0.01]
        # new_tool_params.inertia = [4.43e-04, 8.19e-04, 1.262e-03, 0.0, 0.0, 0.0]
        # # flexiv ToolParams tcp_location expects 7 floats: [x, y, z, qw, qx, qy, qz]
        # new_tool_params.tcp_location = [0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]

        # if tool.exist(new_tool_name):
        #     print(f"Tool with the same name [{new_tool_name}] already exists, removing it now")
        #     tool.Switch("Flange")  # must not remove the currently active tool
        #     tool.Remove(new_tool_name)

        # print(f"Adding new tool [{new_tool_name}] to the robot")
        # tool.Add(new_tool_name, new_tool_params)

        # print("All configured tools:")
        # tool_list = tool.list()
        # for i in range(len(tool_list)):
        #     print(f"[{i}] {tool_list[i]}")
        # print()

        # print(f"Switching to tool [{new_tool_name}]")
        # tool.Switch(new_tool_name)
        print(f"Current active tool: [{tool.name()}]")

        print("Entering free-drive mode (floatingSoft) …")
        robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("floatingSoft", {})

        print("\n--- Controls ---")
        print("Press Enter to save the current EEF pose.")
        print("Type 'q' + Enter to quit and exit free-drive mode.")
        print("----------------")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"RobotToolPose_{timestamp}.csv"
        print(f"Pose data will be saved to: {csv_filename}\n")

        while True:
            try:
                cmd = input(f"[{pose_count} saved] Enter=save  q=quit > ").strip().lower()
            except EOFError:
                break

            if cmd == "q":
                print("Exiting …")
                break

            # Any input other than 'q' (including just Enter) saves a pose.
            print(f"Saving pose #{pose_count + 1} …")
            robot.Stop()  # hold current pose for a stable reading
            # time.sleep(0.5)
            time.sleep(5.0)
            # flexivrdk native: [x, y, z, qw, qx, qy, qz]
            tcp = robot.states().tcp_pose
            pos = tcp[:3]
            qw, qx, qy, qz = tcp[3], tcp[4], tcp[5], tcp[6]

            current_pose_data = [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(qx),
                float(qy),
                float(qz),
                float(qw),
            ]
            save_pose_to_csv(current_pose_data, csv_filename)
            pose_count += 1

            # Re-enter free-drive for the next sample.
            print("Re-entering free-drive mode …\n")
            robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
            robot.ExecutePrimitive("floatingSoft", {})
            time.sleep(0.3)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if robot is not None:
            print("Stopping robot and cleaning up …")
            try:
                robot.Stop()
            except Exception:
                pass

        if csv_filename:
            print(f"Done. {pose_count} pose(s) saved. ({csv_filename})")
        else:
            print(f"Done. {pose_count} pose(s) saved.")
        sys.exit()


if __name__ == "__main__":
    main()
