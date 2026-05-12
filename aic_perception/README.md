# aic-perception

## Usage

```
/entrypoint.sh spawn_task_board:=true     task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2     task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.785     sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.08     sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09     nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005     sc_port_0_present:=true sc_port_0_translation:=-0.04     spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true     ground_truth:=true start_aic_engine:=false
```


sample config task 1
```
/entrypoint.sh spawn_task_board:=true     task_board_x:=0.15 task_board_y:=-0.2 task_board_z:=1.14     task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=3.1415     sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=0.03     sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09     nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.036     sc_port_0_present:=true sc_port_0_translation:=0.042     spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true     ground_truth:=true start_aic_engine:=false
```


```
pixi run ros2 run aic_perception pose_estimator_node --ros-args -p yolo_checkpoint_path:=/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data/weights_istances/yolo26_segment.pt -p templates_dir:=/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data/templates -p models_dir:=/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data/ic/models -p image_topic:=/center_camera/image -p camera_info_topic:=/center_camera/camera_info -p use_sim_time:=true
```


iaslab@rvlab-04:~$ ros2 run tf2_ros tf2_echo gripper/tcp cable_0/sfp_tip_link
[INFO] [1778580776.781914610] [tf2_echo]: Waiting for transform gripper/tcp ->  cable_0/sfp_tip_link: Invalid frame ID "gripper/tcp" passed to canTransform argument target_frame - frame does not exist
At time 243.492000000
- Translation: [-0.000, -0.018, 0.048]
- Rotation: in Quaternion (xyzw) [0.180, 0.006, -0.027, 0.983]
- Rotation: in RPY (radian) [0.361, 0.021, -0.052]
- Rotation: in RPY (degree) [20.671, 1.192, -2.982]
- Matrix:
  0.998  0.056  0.001 -0.000
 -0.052  0.934 -0.354 -0.018
 -0.021  0.353  0.935  0.048
  0.000  0.000  0.000  1.000
At time 244.80000000
- Translation: [-0.000, -0.018, 0.048]
- Rotation: in Quaternion (xyzw) [0.180, 0.006, -0.027, 0.983]
- Rotation: in RPY (radian) [0.361, 0.021, -0.052]
- Rotation: in RPY (degree) [20.671, 1.192, -2.982]
- Matrix:
  0.998  0.056  0.001 -0.000
 -0.052  0.934 -0.354 -0.018
 -0.021  0.353  0.935  0.048
  0.000  0.000  0.000  1.000


gripper/tcp -> sfp_tip_link transform:
[[    0.99845    0.055723  0.00031675   2.777e-05]
 [  -0.052085     0.93525    -0.35012   -0.020671]
 [  -0.019806     0.34956      0.9367    0.054261]
 [          0           0           0           1]]
gripper/tcp -> sfp_tip_link transform:
[[    0.99844    0.055793  0.00050377  1.7201e-06]
 [  -0.052086     0.93527    -0.35009    -0.02067]
 [  -0.020003     0.34951     0.93672    0.051115]
 [          0           0           0           1]]
gripper/tcp -> sfp_tip_link transform:
[[    0.99845    0.055665  0.00016862 -8.8648e-07]
 [  -0.052081     0.93523    -0.35018   -0.020687]
 [  -0.019651     0.34963     0.93668    0.054119]
 [          0           0           0           1]]
t: geometry_msgs.msg.Vector3(x=-8.86478297101867e-07, y=-0.020687488511920038, z=0.054118867685087224)
q: geometry_msgs.msg.Quaternion(x=-0.17785966749625665, y=-0.00503708733179058, z=0.027383843138112103, w=-0.983661891514159)

iaslab@rvlab-04:~$ ros2 run tf2_ros tf2_echo gripper/tcp cable_1/sc_tip_link
[INFO] [1778581161.051660237] [tf2_echo]: Waiting for transform gripper/tcp ->  cable_1/sc_tip_link: Invalid frame ID "gripper/tcp" passed to canTransform argument target_frame - frame does not exist
At time 118.42000000
- Translation: [-0.000, -0.013, 0.017]
- Rotation: in Quaternion (xyzw) [0.159, -0.166, 0.694, 0.682]
- Rotation: in RPY (radian) [-0.015, -0.464, 1.592]
- Rotation: in RPY (degree) [-0.843, -26.599, 91.223]
- Matrix:
 -0.019 -1.000 -0.005 -0.000
  0.894 -0.015 -0.448 -0.013
  0.448 -0.013  0.894  0.017
  0.000  0.000  0.000  1.000
At time 118.360000000
- Translation: [-0.000, -0.013, 0.017]
- Rotation: in Quaternion (xyzw) [0.159, -0.166, 0.694, 0.682]
- Rotation: in RPY (radian) [-0.015, -0.464, 1.592]
- Rotation: in RPY (degree) [-0.843, -26.603, 91.223]
- Matrix:
 -0.019 -1.000 -0.005 -0.000
  0.894 -0.015 -0.448 -0.013
  0.448 -0.013  0.894  0.017
  0.000  0.000  0.000  1.000


gripper/tcp -> sc_tip_link transform:
[[  -0.019921    -0.99979  -0.0051387 -0.00038171]
 [    0.89395   -0.015511    -0.44789   -0.013111]
 [    0.44772   -0.013516     0.89407    0.016467]
 [          0           0           0           1]]

gripper/tcp -> sc_tip_link transform:
[[   0.018881    -0.99982   -0.001564  -0.0005699]
 [    0.79156    0.015904    -0.61088   -0.010964]
 [     0.6108    0.010296     0.79172   0.0096407]
 [          0           0           0           1]]
t: geometry_msgs.msg.Vector3(x=-0.0005699007703731662, y=-0.01096440443128599, z=0.009640708469970782)
q: geometry_msgs.msg.Quaternion(x=-0.2298133984379278, y=0.22655156773159751, z=-0.6627472977643364, w=-0.6757412205316223)

 gripper/tcp -> sc_tip_link transform:
[[   0.018877    -0.99982  -0.0015644 -0.00056606]
 [    0.79156    0.015901    -0.61088   -0.010964]
 [     0.6108    0.010293     0.79172   0.0096344]
 [          0           0           0           1]]
t: geometry_msgs.msg.Vector3(x=-0.0005660645834346112, y=-0.010964335024747696, z=0.009634385494507569)
q: geometry_msgs.msg.Quaternion(x=-0.22981359577438196, y=0.22655281991311554, z=-0.6627482173286723, w=-0.675739871567159)