# aic-perception

## Usage

```
/entrypoint.sh spawn_task_board:=true     task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2     task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.785     sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.08     sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09     nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005     sc_port_0_present:=true sc_port_0_translation:=-0.04     spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true     ground_truth:=true start_aic_engine:=false
```

```
pixi run ros2 run aic_perception pose_estimator_node --ros-args -p yolo_checkpoint_path:=/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data/weights_istances/yolo26_segment.pt -p templates_dir:=/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data/templates -p models_dir:=/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data/ic/models -p image_topic:=/center_camera/image -p camera_info_topic:=/center_camera/camera_info -p use_sim_time:=true
```