1. 恢复代码掌控力，压回主干。
    1.1 必要时，自己画模块调用图，不让ai画，或者ai只提意见。
2. 把冗长的guidebook拆成小文档 (历史工作放入)
3. 删除代码的顺序
    3.1 先删完全没有 import 的文件； 3.2 重复 adapter 3. historical scripts

### cleanup log
1. action sequence.py 
冗余主要在 ActionSequence.load() 同时兼容目录、task_plan.json、TXT/PDDL fallback。现在 executor 的唯一入口是 action_sequence.json
2. action.py
remove _inject_test_grasp_camera_pose 冗余的原来的测试代码
3. clean up unused code
4. x5_remote_backend.py, x5_controller.py 删除冗余代码（gripper 逻辑拆分出来， 坐标转换 拆分）
5. 