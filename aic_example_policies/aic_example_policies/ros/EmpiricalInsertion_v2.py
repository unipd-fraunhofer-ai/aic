#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import numpy as np
from enum import Enum, auto

from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from transforms3d.quaternions import quat2mat, mat2quat
from geometry_msgs.msg import Pose, PoseStamped, Vector3, Wrench, WrenchStamped
from std_msgs.msg import Header

from .utils import (
    transform_to_matrix,
    matrix_to_pose,
    xyz_rpy_to_matrix,
    quat_multiply,
    quat_normalize,
    euler_xyz_to_quat,
    euler_xyz_to_quat_wxyz,
    quat2euler,
    add_noise_to_transform,
    wrench_at_tip_from_wrist,
    subtract_wrench_offset,
)

QuaternionTuple = tuple[float, float, float, float]

class InsertCableState(Enum):
    INIT = auto()
    WAIT_FOR_TFS = auto()
    LOOKUP_TFS = auto()
    MOVE_ABOVE_PORT = auto()
    DESCEND_AND_INSERT = auto()
    STABILIZE_XY = auto()
    UNTILT_INSERTED_CABLE = auto()
    REDESCEND = auto()
    FIX_YAW = auto()
    STABILIZE = auto()
    DONE = auto()
    FAILED = auto()


class EmpiricalInsertion_v2(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        self.pos_std = [0.003]*3   # Scale of the noise to add to the target position
        self.rot_std = [np.deg2rad(7)]*3    # Scale of the noise to add to the target rotation
        super().__init__(parent_node)
        self._plug_wrench_pub = parent_node.create_publisher(
            WrenchStamped,
            "/plug_wrench",
            10,
        )
        self._plug_pose_pub = parent_node.create_publisher(
            PoseStamped,
            "/plug_pose",
            10,
        )
        self._plug_reference_pose_pub = parent_node.create_publisher(
            PoseStamped,
            "/plug_reference_pose",
            10,
        )

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False
    
    def publish_tip_pose(self) -> PoseStamped:
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )

        plug_pose = PoseStamped(
            header=Header(
                frame_id="base_link",
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=Pose(
                position=Point(
                    x=plug_tf_stamped.transform.translation.x,
                    y=plug_tf_stamped.transform.translation.y,
                    z=plug_tf_stamped.transform.translation.z,
                ),
                orientation=Quaternion(
                    x=plug_tf_stamped.transform.rotation.x,
                    y=plug_tf_stamped.transform.rotation.y,
                    z=plug_tf_stamped.transform.rotation.z,
                    w=plug_tf_stamped.transform.rotation.w,
                ),
            ),
        )

        self._plug_pose_pub.publish(plug_pose)
        return plug_pose

    def publish_tip_wrench(self, observation:Observation):
        wrist_wrench_corrected = subtract_wrench_offset(
            observation.wrist_wrench,
            observation.controller_state.fts_tare_offset,
        )

        plug_wrench = wrench_at_tip_from_wrist(
            wrist_wrench_corrected,
            self.T_wrist_tip,
        )

        self._plug_wrench_pub.publish(
            WrenchStamped(
                header=Header(
                    frame_id=f"{self._task.cable_name}/{self._task.plug_name}_link",
                    stamp=self._parent_node.get_clock().now().to_msg(),
                ),
                wrench=plug_wrench,
            )
        )
        return plug_wrench

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        gripper_tip_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        x_offset: float = 0.0,
        reset_xy_integrator: bool = False,
        tilt_roll: float = 0.0,
        tilt_pitch: float = 0.0,
        tilt_yaw: float = 0.0,
        rotate_tcp_in_place: bool = False,
        controlled_frame_offset_tip: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> Pose:
        # Fixed transform: gripper/tcp -> plug tip
        T_gripper_tip = transform_to_matrix(gripper_tip_transform)

        # Fixed transform: plug tip -> controlled virtual frame
        # Translation is expressed in plug-tip coordinates.
        T_tip_controlled = np.eye(4)
        T_tip_controlled[:3, 3] = controlled_frame_offset_tip

        # Current gripper pose
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        T_base_gripper_current = transform_to_matrix(gripper_tf_stamped.transform)

        # Current plug tip pose
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        T_base_tip_current = transform_to_matrix(plug_tf_stamped.transform)

        # Current controlled-frame pose
        T_base_controlled_current = T_base_tip_current @ T_tip_controlled
        p_base_controlled_current = T_base_controlled_current[:3, 3]

        # Port pose
        T_base_port = transform_to_matrix(port_transform)
        R_base_port = T_base_port[:3, :3]
        p_base_port = T_base_port[:3, 3]

        # Additional desired tilt with respect to the port frame
        T_tilt = xyz_rpy_to_matrix(
            0.0, 0.0, 0.0,
            tilt_roll, tilt_pitch, tilt_yaw,
        )
        R_tilt = T_tilt[:3, :3]

        # Desired orientation of the controlled frame.
        # Since controlled frame has same orientation as tip,
        # this is also the desired tip orientation.
        R_base_controlled_desired = R_base_port @ R_tilt

        # XY feedback now uses the controlled frame, not the plug-tip center
        tip_x_error = p_base_port[0] - p_base_controlled_current[0]
        tip_y_error = p_base_port[1] - p_base_controlled_current[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        i_gain = 0.15

        # Desired controlled-frame position
        p_base_controlled_desired = (
            p_base_port
            + np.array([x_offset, 0.0, z_offset])
        )

        p_base_controlled_desired[0] += i_gain * self._tip_x_error_integrator
        p_base_controlled_desired[1] += i_gain * self._tip_y_error_integrator

        # Desired controlled-frame pose
        T_base_controlled_desired = np.eye(4)
        T_base_controlled_desired[:3, :3] = R_base_controlled_desired
        T_base_controlled_desired[:3, 3] = p_base_controlled_desired

        # Convert desired controlled-frame pose back to desired plug-tip pose:
        #
        # T_base_controlled = T_base_tip * T_tip_controlled
        # therefore:
        # T_base_tip = T_base_controlled * inv(T_tip_controlled)
        T_base_tip_desired = T_base_controlled_desired @ np.linalg.inv(T_tip_controlled)

        # Publish controlled-frame reference pose
        controlled_reference_pose = matrix_to_pose(T_base_controlled_desired)

        self._plug_reference_pose_pub.publish(
            PoseStamped(
                header=Header(
                    frame_id="base_link",
                    stamp=self._parent_node.get_clock().now().to_msg(),
                ),
                pose=controlled_reference_pose,
            )
        )

        # Convert desired plug-tip pose into desired TCP/gripper pose:
        #
        # T_base_tip = T_base_gripper * T_gripper_tip
        # therefore:
        # T_base_gripper = T_base_tip * inv(T_gripper_tip)
        T_base_gripper_target = T_base_tip_desired @ np.linalg.inv(T_gripper_tip)

        # Interpolate gripper position
        current_xyz = T_base_gripper_current[:3, 3]
        target_xyz = T_base_gripper_target[:3, 3]

        if rotate_tcp_in_place:
            blend_xyz = current_xyz
        else:
            blend_xyz = (
                position_fraction * target_xyz
                + (1.0 - position_fraction) * current_xyz
            )

        # Interpolate gripper orientation
        q_current = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )

        q_target = mat2quat(T_base_gripper_target[:3, :3])
        q_slerp = quaternion_slerp(q_current, q_target, slerp_fraction)

        roll, pitch, yaw = quat2euler(q_slerp)

        self.get_logger().info(
            f"[CMD POSE] "
            f"x={blend_xyz[0]:.4f}, y={blend_xyz[1]:.4f}, z={blend_xyz[2]:.4f} | "
            f"roll={np.rad2deg(roll):.2f}°, "
            f"pitch={np.rad2deg(pitch):.2f}°, "
            f"yaw={np.rad2deg(yaw):.2f}°"
        )

        return Pose(
            position=Point(
                x=float(blend_xyz[0]),
                y=float(blend_xyz[1]),
                z=float(blend_xyz[2]),
            ),
            orientation=Quaternion(
                w=float(q_slerp[0]),
                x=float(q_slerp[1]),
                y=float(q_slerp[2]),
                z=float(q_slerp[3]),
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"Entering policy. \nTask: {task}")
        self._task = task

        state = InsertCableState.INIT

        port_frame = None
        cable_tip_frame = None

        port_transform = None
        gripper_tip_transform = None

        z_offset = 0.10
        z_force_threshold = -2.0  # [N], threshold for detecting contact in Z during insertion
        stiffness = [200.0, 200.0, 200.0, 50, 50, 50]
        wrench_feedback_gains_at_tip = [0.] * 3 + [0] * 3

        roll_angle = np.deg2rad(15)
        pitch_angle = np.deg2rad(15)
        yaw_angle = np.deg2rad(0)
        untilt_start_time = None
        untilt_duration = Duration(seconds=30.0)
        untilt_step = np.deg2rad(0.5)

        move_above_step = 0
        insertion_start_time = None
        stabilize_step = 0
        target_stabilize_steps = 50        
        tf_initialized = False
        
        # fix yaw parameters
        fix_yaw_direction = 1.0
        fix_yaw_attempt = 0
        fix_yaw_max_attempts = 2
        fix_yaw_start_z = None
        fix_yaw_best_z = None
        fix_yaw_no_improve_steps = 0
        fix_yaw_max_no_improve_steps = 100
        fix_yaw_step = np.deg2rad(0.2)
        fix_yaw_limit = np.deg2rad(10.0)
        fix_yaw_min_z_improvement = 0.001

        while state not in [InsertCableState.DONE, InsertCableState.FAILED]:
            observation = get_observation()

            should_send_command = False
            command_kwargs = dict(
                stiffness=stiffness,
                wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                z_offset=z_offset,
                reset_xy_integrator=True,
                tilt_roll=roll_angle,
                tilt_pitch=pitch_angle,
                tilt_yaw=yaw_angle,
            )

            if tf_initialized:
                plug_wrench = self.publish_tip_wrench(observation)
                plug_pose = self.publish_tip_pose()

            if state == InsertCableState.INIT:
                self.sleep_for(1)

                port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
                cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

                state = InsertCableState.WAIT_FOR_TFS

            elif state == InsertCableState.WAIT_FOR_TFS:
                for frame in [port_frame, cable_tip_frame]:
                    if not self._wait_for_tf("base_link", frame):
                        state = InsertCableState.FAILED
                        break
                else:
                    state = InsertCableState.LOOKUP_TFS

            elif state == InsertCableState.LOOKUP_TFS:
                try:
                    port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                        "base_link", port_frame, Time()
                    )
                    gripper_tip_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                        "gripper/tcp", cable_tip_frame, Time()
                    )
                    wrist_tip_tf = self._parent_node._tf_buffer.lookup_transform(
                        "ati/tool_link", cable_tip_frame, Time()
                    )

                except TransformException as ex:
                    self.get_logger().error(f"TF lookup failed: {ex}")
                    state = InsertCableState.FAILED
                    continue

                gripper_tip_transform = gripper_tip_tf_stamped.transform
                self.T_wrist_tip = transform_to_matrix(wrist_tip_tf.transform)

                port_transform = add_noise_to_transform(
                    port_tf_stamped.transform,
                    pos_std=self.pos_std,
                    rot_std=self.rot_std,
                )

                move_above_step = 0
                tf_initialized = True
                state = InsertCableState.MOVE_ABOVE_PORT

            elif state == InsertCableState.MOVE_ABOVE_PORT:
                interp_fraction = move_above_step / 100.0

                stiffness = [100.0, 100.0, 100.0, 60.0, 60.0, 60.0]
                wrench_feedback_gains_at_tip = [0.0] * 6

                command_kwargs.update(
                    stiffness=stiffness,
                    wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )

                should_send_command = True
                move_above_step += 1

                if move_above_step >= 100:
                    insertion_start_time = self.time_now()
                    state = InsertCableState.DESCEND_AND_INSERT

            elif state == InsertCableState.DESCEND_AND_INSERT:
                contact_detected = plug_wrench.force.z < z_force_threshold

                if contact_detected:
                    self.get_logger().info(
                        f"Z contact detected: Fz={plug_wrench.force.z:.3f} N"
                    )
                    untilt_start_time = self.time_now()
                    state = InsertCableState.STABILIZE_XY
                    continue

                if z_offset < -0.015:
                    self.get_logger().warn(
                        "Reached max insertion depth without detecting Z contact."
                    )
                    untilt_start_time = self.time_now()
                    state = InsertCableState.STABILIZE
                    continue

                z_offset -= 0.001

                command_kwargs.update(
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                )

                should_send_command = True

            elif state == InsertCableState.STABILIZE_XY:
                if z_offset < -0.015:
                    self.get_logger().info("Insertion completed.")

                    port_transform.translation.x = plug_pose.pose.position.x
                    port_transform.translation.y = plug_pose.pose.position.y

                    if plug_pose.pose.position.z >= 0.1765:
                        self.get_logger().warn(
                            "Plug tip is not aligned with port. Trying to fix yaw."
                        )
                        state = InsertCableState.FIX_YAW
                    else:
                        self.get_logger().warn(
                            "Plug tip is aligned with port. Trying to reorient."
                        )
                        state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue

                z_offset -= 0.001

                stiffness = [10, 10, 90, 5, 5, 5]
                wrench_feedback_gains_at_tip = [0.0] * 6

                command_kwargs.update(
                    stiffness=stiffness,
                    wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                )

                should_send_command = True

            elif state == InsertCableState.FIX_YAW:
                current_z = plug_pose.pose.position.z
                plug_inserted = current_z < 0.1765

                if fix_yaw_start_z is None:
                    fix_yaw_start_z = current_z
                    fix_yaw_best_z = current_z
                    fix_yaw_no_improve_steps = 0

                if plug_inserted:
                    self.get_logger().info("Yaw fixed.")
                    fix_yaw_start_z = None
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue

                if current_z < fix_yaw_best_z - fix_yaw_min_z_improvement:
                    fix_yaw_best_z = current_z
                    fix_yaw_no_improve_steps = 0
                else:
                    fix_yaw_no_improve_steps += 1

                yaw_angle += fix_yaw_direction * fix_yaw_step

                yaw_limit_reached = abs(yaw_angle) >= fix_yaw_limit
                no_improvement = fix_yaw_no_improve_steps >= fix_yaw_max_no_improve_steps

                if yaw_limit_reached or no_improvement:
                    fix_yaw_attempt += 1

                    if fix_yaw_attempt < fix_yaw_max_attempts:
                        fix_yaw_direction *= -1.0
                        fix_yaw_start_z = None
                        fix_yaw_best_z = None
                        fix_yaw_no_improve_steps = 0
                        continue

                    self.get_logger().error("Yaw adjustment failed in both directions.")
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue

                stiffness = [90, 90, 90, 50, 50, 200]
                wrench_feedback_gains_at_tip = [0.0] * 6

                command_kwargs.update(
                    stiffness=stiffness,
                    wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                    tilt_yaw=yaw_angle,
                )

                should_send_command = True

            elif state == InsertCableState.UNTILT_INSERTED_CABLE:
                roll_done = abs(roll_angle) <= untilt_step
                pitch_done = abs(pitch_angle) <= untilt_step

                if not roll_done:
                    roll_angle -= np.sign(roll_angle) * untilt_step

                if not pitch_done:
                    pitch_angle -= np.sign(pitch_angle) * untilt_step

                if roll_done and pitch_done and z_offset >= 0.04:
                    state = InsertCableState.REDESCEND
                    continue

                z_offset += 0.001

                stiffness = [90, 90, 200, 50, 50, 200]
                wrench_feedback_gains_at_tip = [0.0] * 6

                command_kwargs.update(
                    stiffness=stiffness,
                    wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                    tilt_yaw=yaw_angle,
                )

                should_send_command = True

            elif state == InsertCableState.REDESCEND:
                z_offset -= 0.001

                stiffness = [90, 90, 90, 200, 200, 200]
                wrench_feedback_gains_at_tip = [0.0] * 6

                command_kwargs.update(
                    stiffness=stiffness,
                    wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                    tilt_yaw=yaw_angle,
                )

                should_send_command = True

                if z_offset < -0.015:
                    self.get_logger().info("Redescend completed.")
                    state = InsertCableState.STABILIZE
                    continue

            elif state == InsertCableState.STABILIZE:
                self.get_logger().info("Waiting for connector to stabilize...")
                stabilize_step += 1

                if stabilize_step > target_stabilize_steps:
                    state = InsertCableState.DONE

            # ---------------------------------------------------------
            # Single command sending point
            # ---------------------------------------------------------
            if should_send_command:
                try:
                    pose=self.calc_gripper_pose(
                        port_transform,
                        gripper_tip_transform,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                        tilt_roll=roll_angle,
                        tilt_pitch=pitch_angle,
                        tilt_yaw=yaw_angle,
                        controlled_frame_offset_tip=np.array([-0.003, 0.003, 0.0]),
                    )

                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=pose,
                        stiffness=command_kwargs["stiffness"],
                        wrench_feedback_gains_at_tip=command_kwargs["wrench_feedback_gains_at_tip"],
                    )

                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed in state {state.name}: {ex}"
                    )

            self.sleep_for(0.05)
        
        self.get_logger().info("CheatCode.insert_cable() exiting...")
        return True
    
    def set_pose_target(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        frame_id: str = "base_link",
        stiffness: list = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
        damping: list = [100.0, 100.0, 100.0, 40.0, 40.0, 40.0],
        wrench_feedback_gains_at_tip: list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ) -> None:
        """Invoke the move_robot callback to request the supplied Pose.

        This is a convenience function which populates a MotionUpdate message
        with a reasonable set of default parameters, and invokes the move_robot
        callback to request motion to the supplied pose.

        The robot can be controlled in several different ways. This function
        is intended to be the simplest way to move the arm around, by sending
        a desired pose (position and orientation) for the gripper's
        "tool control point" (TCP), which is the "pinch point" between the very
        end of the gripper fingers. The rest of the control stack will take care
        of moving all the arm's joints to so that the gripper TCP ends up in
        the desired position and orientation.

        The constants defined in this function are intended to provide
        reasonable default behavior if the arm is unable to achieve the
        requested pose. Different values for stiffness, damping, wrenches, and
        so on can be used for different types of arm behavior. These values
        are only intended to provide a starting point, and can be adjusted as
        desired.
        """
        motion_update = MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")
