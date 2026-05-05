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


# Transform utilities
def transform_to_matrix(t: Transform) -> np.ndarray:
    q_wxyz = np.array([
        t.rotation.w,
        t.rotation.x,
        t.rotation.y,
        t.rotation.z,
    ])
    R = quat2mat(q_wxyz)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [
        t.translation.x,
        t.translation.y,
        t.translation.z,
    ]
    return T

def matrix_to_pose(T: np.ndarray) -> Pose:
    q_wxyz = mat2quat(T[:3, :3])
    return Pose(
        position=Point(
            x=float(T[0, 3]),
            y=float(T[1, 3]),
            z=float(T[2, 3]),
        ),
        orientation=Quaternion(
            w=float(q_wxyz[0]),
            x=float(q_wxyz[1]),
            y=float(q_wxyz[2]),
            z=float(q_wxyz[3]),
        ),
    )

def xyz_rpy_to_matrix(x, y, z, roll, pitch, yaw) -> np.ndarray:
    q_xyzw = euler_xyz_to_quat(roll, pitch, yaw)
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    T = np.eye(4)
    T[:3, :3] = quat2mat(q_wxyz)
    T[:3, 3] = [x, y, z]
    return T

def quat_multiply(q1, q2):
    """
    Quaternion multiplication.
    Quaternions are in [x, y, z, w] format.
    Returns q = q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w])

def quat_normalize(q):
    return q / np.linalg.norm(q)

def euler_xyz_to_quat(roll, pitch, yaw):
    """
    Convert XYZ Euler angles to quaternion [x, y, z, w]
    """
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.array([x, y, z, w])

def euler_xyz_to_quat_wxyz(roll, pitch, yaw):
    q_xyzw = euler_xyz_to_quat(roll, pitch, yaw)
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

def quat2euler(q):
    w, x, y, z = q

    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw

# Noise addition utility
def add_noise_to_transform(transform, pos_std=[0.002]*3, rot_std=[0.01]):
    """
    Add Gaussian noise to a geometry_msgs Transform.
    
    pos_std: translation std in meters
    rot_std: rotation std in radians
    """
    # Translation noise
    # transform.translation.x += np.random.normal(0.0, pos_std[0])
    # transform.translation.y += np.random.normal(0.0, pos_std[1])
    # transform.translation.z += np.random.normal(0.0, pos_std[2])
    transform.translation.x += 0.005
    transform.translation.y += 0.005
    transform.translation.z -= 0.00

    # Current quaternion [x, y, z, w]
    q_current = np.array([
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w
    ])

    # Small rotation noise in roll-pitch-yaw
    # roll_noise = np.random.normal(0.0, rot_std[0])
    # pitch_noise = np.random.normal(0.0, rot_std[1])
    # yaw_noise = np.random.normal(0.0, rot_std[2])
    roll_noise = np.deg2rad(0)
    pitch_noise = np.deg2rad(0)
    yaw_noise = np.deg2rad(5)

    q_noise = euler_xyz_to_quat(roll_noise, pitch_noise, yaw_noise)

    # Apply noise rotation
    q_new = quat_multiply(q_noise, q_current)
    q_new = quat_normalize(q_new)

    transform.rotation.x = q_new[0]
    transform.rotation.y = q_new[1]
    transform.rotation.z = q_new[2]
    transform.rotation.w = q_new[3]

    return transform

# Wrench transformation utility
def wrench_at_tip_from_wrist(wrist_wrench: WrenchStamped, T_wrist_tip: np.ndarray,) -> Wrench:
    R = T_wrist_tip[:3, :3]
    r = T_wrist_tip[:3, 3]

    F_w = np.array([
        wrist_wrench.wrench.force.x,
        wrist_wrench.wrench.force.y,
        wrist_wrench.wrench.force.z,
    ])

    tau_w = np.array([
        wrist_wrench.wrench.torque.x,
        wrist_wrench.wrench.torque.y,
        wrist_wrench.wrench.torque.z,
    ])

    # Transport moment from wrist to tip, then express in tip frame
    F_tip = R.T @ F_w
    tau_tip = R.T @ (tau_w - np.cross(r, F_w))

    return Wrench(
        force=Vector3(
            x=float(F_tip[0]),
            y=float(F_tip[1]),
            z=float(F_tip[2]),
        ),
        torque=Vector3(
            x=float(tau_tip[0]),
            y=float(tau_tip[1]),
            z=float(tau_tip[2]),
        ),
    )
    
def subtract_wrench_offset(wrench: WrenchStamped, offset: WrenchStamped) -> WrenchStamped:
    return WrenchStamped(
        header=wrench.header,
        wrench=Wrench(
        force=Vector3(
            x=wrench.wrench.force.x - offset.wrench.force.x,
            y=wrench.wrench.force.y - offset.wrench.force.y,
            z=wrench.wrench.force.z - offset.wrench.force.z,
        ),
        torque=Vector3(
            x=wrench.wrench.torque.x - offset.wrench.torque.x,
            y=wrench.wrench.torque.y - offset.wrench.torque.y,
            z=wrench.wrench.torque.z - offset.wrench.torque.z,
        ),
        ))


class EmpiricalInsertion(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        self.pos_std = [0.001]*3   # Scale of the noise to add to the target position
        self.rot_std = [0.01]*3    # Scale of the noise to add to the target rotation
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
    ) -> Pose:
        # Fixed transform: gripper/tcp -> plug tip
        T_gripper_tip = transform_to_matrix(gripper_tip_transform)

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

        # Port pose
        T_base_port = transform_to_matrix(port_transform)
        R_base_port = T_base_port[:3, :3]
        p_base_port = T_base_port[:3, 3]

        # Additional desired tilt of the plug tip w.r.t. the port frame
        T_tilt = xyz_rpy_to_matrix(
            0.0, 0.0, 0.0,
            tilt_roll, tilt_pitch, tilt_yaw,
        )
        R_tilt = T_tilt[:3, :3]

        # Desired plug-tip orientation.
        # This is the matrix equivalent of the original q_port * q_plug_inv mechanism,
        # but with an extra relative tilt.
        R_base_tip_desired = R_base_port @ R_tilt

        # Current tip position for XY feedback
        p_base_tip_current = np.array([
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        ])

        tip_x_error = p_base_port[0] - p_base_tip_current[0]
        tip_y_error = p_base_port[1] - p_base_tip_current[1]

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

        # Desired tip position.
        # This makes the TIP move along its tilted local z-axis.
        # p_base_tip_desired = (
        #     p_base_port
        #     + R_base_tip_desired @ np.array([0.0, 0.0, -z_offset])
        # )
        p_base_tip_desired = (
            p_base_port
            + np.array([x_offset, 0.0, z_offset])
        )

        # Keep the original XY integral correction behavior
        p_base_tip_desired[0] += i_gain * self._tip_x_error_integrator
        p_base_tip_desired[1] += i_gain * self._tip_y_error_integrator

        T_base_tip_desired = np.eye(4)
        T_base_tip_desired[:3, :3] = R_base_tip_desired
        T_base_tip_desired[:3, 3] = p_base_tip_desired
        
        # Convert desired TIP pose to Pose
        tip_reference_pose = matrix_to_pose(T_base_tip_desired)

        # Publish it
        self._plug_reference_pose_pub.publish(
            PoseStamped(
                header=Header(
                    frame_id="base_link",
                    stamp=self._parent_node.get_clock().now().to_msg(),
                ),
                pose=tip_reference_pose,
            )
        )

        # Convert desired TIP pose into desired TCP/gripper pose:
        # T_base_gripper * T_gripper_tip = T_base_tip_desired
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
        
        roll, pitch, yaw = quat2euler(q_slerp)  # assumes (w, x, y, z)

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

        z_offset = 0.15
        z_force_threshold = -2.0  # [N], threshold for detecting contact in Z during insertion
        stiffness = [200.0, 200.0, 200.0, 50, 50, 50]
        wrench_feedback_gains_at_tip = [0.] * 3 + [0] * 3

        roll_angle = np.deg2rad(10)
        pitch_angle = np.deg2rad(10)
        yaw_angle = np.deg2rad(0)
        untilt_start_time = None
        untilt_duration = Duration(seconds=30.0)
        untilt_step = np.deg2rad(0.5)

        move_above_step = 0
        insertion_start_time = None
        stabilize_steps = 0
        tf_initialized = False
        
        # fix yaw parameters
        fix_yaw_direction = 1.0
        fix_yaw_attempt = 0
        fix_yaw_max_attempts = 2
        fix_yaw_start_z = None
        fix_yaw_best_z = None
        fix_yaw_no_improve_steps = 0
        fix_yaw_max_no_improve_steps = 100
        fix_yaw_step = np.deg2rad(0.02)
        fix_yaw_limit = np.deg2rad(10.0)
        fix_yaw_min_z_improvement = 0.001

        while state not in [InsertCableState.DONE, InsertCableState.FAILED]:
            observation = get_observation()
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
                        "base_link",
                        port_frame,
                        Time(),
                    )

                    gripper_tip_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                        "gripper/tcp",
                        cable_tip_frame,
                        Time(),
                    )

                    wrist_tip_tf = self._parent_node._tf_buffer.lookup_transform(
                        "ati/tool_link",
                        cable_tip_frame,
                        Time(),
                    )

                except TransformException as ex:
                    self.get_logger().error(f"TF lookup failed: {ex}")
                    state = InsertCableState.FAILED
                    continue

                gripper_tip_transform = gripper_tip_tf_stamped.transform
                self.T_wrist_tip = transform_to_matrix(wrist_tip_tf.transform)

                port_transform = port_tf_stamped.transform
                self.get_logger().info(f"port_transform: {port_transform}")

                port_transform = add_noise_to_transform(
                    port_transform,
                    pos_std=self.pos_std,
                    rot_std=self.rot_std,
                )

                self.get_logger().info(f"port_transform with noise: {port_transform}")

                move_above_step = 0
                tf_initialized = True
                state = InsertCableState.MOVE_ABOVE_PORT

            elif state == InsertCableState.MOVE_ABOVE_PORT:

                interp_fraction = move_above_step / 100.0
                stiffness = [100.0, 100.0, 100.0, 60., 60., 60.]
                wrench_feedback_gains_at_tip = [0.] * 3 + [0.] * 3
                try:
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_transform,
                            gripper_tip_transform,
                            slerp_fraction=interp_fraction,
                            position_fraction=interp_fraction,
                            z_offset=z_offset,
                            reset_xy_integrator=True,
                            tilt_roll=roll_angle,
                            tilt_pitch=pitch_angle,
                        ),
                        stiffness=stiffness,
                        wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed during interpolation: {ex}"
                    )

                move_above_step += 1

                if move_above_step >= 100:
                    insertion_start_time = self.time_now()
                    state = InsertCableState.DESCEND_AND_INSERT

                self.sleep_for(0.05)

            elif state == InsertCableState.DESCEND_AND_INSERT:
                contact_detected = plug_wrench.force.z < z_force_threshold

                if contact_detected:
                    self.get_logger().info(
                        f"Z contact detected: Fz={plug_wrench.force.z:.3f} N"
                    )
                    untilt_start_time = self.time_now()
                    state = InsertCableState.STABILIZE_XY
                    # state = InsertCableState.STABILIZE
                    continue

                if z_offset < -0.015:
                    self.get_logger().warn(
                        "Reached max insertion depth without detecting Z contact."
                    )
                    untilt_start_time = self.time_now()
                    state = InsertCableState.STABILIZE
                    continue
                else:
                    z_offset -= 0.0005
                
                try:
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_transform,
                            gripper_tip_transform,
                            z_offset=z_offset,
                            reset_xy_integrator=True,
                            tilt_roll=roll_angle,
                            tilt_pitch=pitch_angle,
                        ),
                        stiffness=stiffness,
                        wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    )
                    
                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed during interpolation: {ex}"
                    )
                self.sleep_for(0.05)
                    
                
            elif state == InsertCableState.STABILIZE_XY:                
                # roll_done = abs(roll_angle) <= untilt_step
                # pitch_done = abs(pitch_angle) <= untilt_step
                # roll_done = True
                # pitch_done = True
                
                # if roll_done and pitch_done:
                #     if self.time_now() - untilt_start_time > untilt_duration:
                #         state = InsertCableState.STABILIZE
                #         continue
                # else:
                #     if not roll_done:
                #         roll_angle -= np.sign(roll_angle) * untilt_step
                #         pass

                #     if not pitch_done:
                #         pitch_angle -= np.sign(pitch_angle) * untilt_step
                #         pass
                
                
                if z_offset < -0.015:
                    self.get_logger().info(
                        "Insertion completed."
                    )
                    if plug_pose.pose.position.z >= 0.1765:
                        self.get_logger().warn(
                            "Plug tip is not aligned with port. Trying to fix yaw before redescending."
                        )
                        state = InsertCableState.FIX_YAW
                    else:
                        self.get_logger().warn(
                            "Plug tip is aligned with port. Trying to reorient."
                        )
                        state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue
                else:
                    z_offset -= 0.0005

                self.get_logger().info(
                    f"untilting: roll={np.rad2deg(roll_angle):0.3f} deg, "
                    f"pitch={np.rad2deg(pitch_angle):0.3f} deg"
                )

                try:
                    wrench_feedback_gains_at_tip = [-0.] * 3 + [-.0] * 3
                    stiffness= [10,10,90,5,5,5]
                    # damping= [200,200,200,80,80,80]
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_transform,
                            gripper_tip_transform,
                            z_offset=z_offset,
                            reset_xy_integrator=True,
                            tilt_roll=roll_angle,
                            tilt_pitch=pitch_angle,
                            # rotate_tcp_in_place=True,
                        ),
                        stiffness=stiffness,
                        wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed during untilting: {ex}"
                    )

                self.sleep_for(0.05)
                
            elif state == InsertCableState.UNTILT_INSERTED_CABLE:
                observation = get_observation()
                roll_done = abs(roll_angle) <= untilt_step 
                pitch_done = abs(pitch_angle) <= untilt_step
                # roll_done = True
                # pitch_done = True
                
                # if roll_done and pitch_done:
                #     if self.time_now() - untilt_start_time > untilt_duration:
                #         state = InsertCableState.STABILIZE
                #         continue
                # else:
                if not roll_done:
                    roll_angle -= np.sign(roll_angle) * untilt_step
                    pass

                if not pitch_done:
                    pitch_angle -= np.sign(pitch_angle) * untilt_step
                    pass
                
                if roll_done and pitch_done and z_offset >= 0.04:
                    self.get_logger().info(
                        "Insertion completed."
                    )
                    state = InsertCableState.REDESCEND
                    continue
                
                z_offset += 0.001

                self.get_logger().info(
                    f"untilting: roll={np.rad2deg(roll_angle):0.3f} deg, "
                    f"pitch={np.rad2deg(pitch_angle):0.3f} deg, "
                    f"z_offset={z_offset:.4f} m"
                )

                try:
                    wrench_feedback_gains_at_tip = [0.] * 3 + [-.0] * 3
                    stiffness= [1,1,200,50,50,200]
                    # damping= [200,200,200,80,80,80]
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_transform,
                            gripper_tip_transform,
                            z_offset=z_offset,
                            reset_xy_integrator=True,
                            tilt_roll=roll_angle,
                            tilt_pitch=pitch_angle,
                            tilt_yaw=yaw_angle,
                            # rotate_tcp_in_place=True,
                        ),
                        stiffness=stiffness,
                        wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed during untilting: {ex}"
                    )

                self.sleep_for(0.05)
                
            elif state == InsertCableState.REDESCEND:
                z_offset -= 0.001

                self.get_logger().info(
                    f"redescending: z_offset={z_offset:.4f} m"
                )

                try:
                    wrench_feedback_gains_at_tip = [0.] * 3 + [-.0] * 2 + [-.0]
                    stiffness= [1,1,90,200,200,200]
                    # damping= [200,200,200,80,80,80]
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_transform,
                            gripper_tip_transform,
                            z_offset=z_offset,
                            reset_xy_integrator=True,
                            tilt_roll=roll_angle,
                            tilt_pitch=pitch_angle,
                            tilt_yaw=yaw_angle,
                            # rotate_tcp_in_place=True,
                        ),
                        stiffness=stiffness,
                        wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed during redescending: {ex}"
                    )

                if z_offset < -0.015:
                    self.get_logger().info(
                        "Redescend completed."
                    )
                    state = InsertCableState.STABILIZE
                    continue

                self.sleep_for(0.05)
                
            elif state == InsertCableState.FIX_YAW:
                self.get_logger().info(f"Fixing yaw, yaw angle={np.rad2deg(yaw_angle):.2f} deg")
                yaw_done = yaw_angle >= np.deg2rad(5) 
                plug_inserted = plug_pose.pose.position.z < 0.1765
                
                if plug_inserted:
                    self.get_logger().info(
                        "Yaw fixed."
                    )
                    # self.sleep_for(2.0)
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue
                elif not plug_inserted and not yaw_done: 
                    yaw_angle += np.deg2rad(0.01)
                
                elif not plug_inserted and yaw_done:
                    self.get_logger().warn(
                        "Yaw adjustment did not work. Proceeding with untilting."
                    )
                    # self.sleep_for(2.0)
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue
                    
                try:
                    wrench_feedback_gains_at_tip = [0.] * 3 + [-.0] * 3 
                    stiffness= [10,10,90,50,50,200]
                    # damping= [200,200,200,80,80,80]
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_transform,
                            gripper_tip_transform,
                            z_offset=z_offset,
                            reset_xy_integrator=False,
                            tilt_roll=roll_angle,
                            tilt_pitch=pitch_angle,
                            tilt_yaw=yaw_angle,
                            # rotate_tcp_in_place=True,
                        ),
                        stiffness=stiffness,
                        wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed during redescending: {ex}"
                    )
                    
            # elif state == InsertCableState.FIX_YAW:
            #     current_z = plug_pose.pose.position.z
            #     plug_inserted = current_z < 0.1765

            #     if fix_yaw_start_z is None:
            #         fix_yaw_start_z = current_z
            #         fix_yaw_best_z = current_z
            #         fix_yaw_no_improve_steps = 0
            #         self.get_logger().info(
            #             f"Starting yaw fix attempt {fix_yaw_attempt + 1}/"
            #             f"{fix_yaw_max_attempts}, direction={fix_yaw_direction:+.0f}"
            #         )

            #     self.get_logger().info(
            #         f"Fixing yaw: yaw={np.rad2deg(yaw_angle):.2f} deg, "
            #         f"z={current_z:.4f}, best_z={fix_yaw_best_z:.4f}, "
            #         f"direction={fix_yaw_direction:+.0f}"
            #     )

            #     if plug_inserted:
            #         self.get_logger().info("Yaw fixed.")
            #         fix_yaw_start_z = None
            #         state = InsertCableState.UNTILT_INSERTED_CABLE
            #         continue

            #     # Lower z is better here.
            #     if current_z < fix_yaw_best_z - fix_yaw_min_z_improvement:
            #         fix_yaw_best_z = current_z
            #         fix_yaw_no_improve_steps = 0
            #     else:
            #         fix_yaw_no_improve_steps += 1

            #     yaw_angle += fix_yaw_direction * fix_yaw_step

            #     yaw_limit_reached = abs(yaw_angle) >= fix_yaw_limit
            #     no_improvement = fix_yaw_no_improve_steps >= fix_yaw_max_no_improve_steps

            #     if yaw_limit_reached or no_improvement:
            #         fix_yaw_attempt += 1

            #         if fix_yaw_attempt < fix_yaw_max_attempts:
            #             self.get_logger().warn(
            #                 "Yaw did not improve insertion. Trying opposite direction."
            #             )

            #             fix_yaw_direction *= -1.0
            #             yaw_angle = 0.0
            #             fix_yaw_start_z = None
            #             fix_yaw_best_z = None
            #             fix_yaw_no_improve_steps = 0
            #             continue

            #         else:
            #             self.get_logger().error(
            #                 "Yaw adjustment failed in both directions."
            #             )
            #             state = InsertCableState.FAILED
            #             continue

            #     try:
            #         wrench_feedback_gains_at_tip = [0.] * 3 + [-.0] * 3
            #         stiffness = [10, 10, 90, 50, 50, 200]

            #         self.set_pose_target(
            #             move_robot=move_robot,
            #             pose=self.calc_gripper_pose(
            #                 port_transform,
            #                 gripper_tip_transform,
            #                 z_offset=z_offset,
            #                 reset_xy_integrator=True,
            #                 tilt_roll=roll_angle,
            #                 tilt_pitch=pitch_angle,
            #                 tilt_yaw=yaw_angle,
            #             ),
            #             stiffness=stiffness,
            #             wrench_feedback_gains_at_tip=wrench_feedback_gains_at_tip,
            #         )
            #         self.sleep_for(0.05)

            #     except TransformException as ex:
            #         self.get_logger().warn(
            #             f"TF lookup failed during yaw fixing: {ex}"
            #         )

            elif state == InsertCableState.STABILIZE:
                self.get_logger().info("Waiting for connector to stabilize...")
                stabilize_steps +=1 
                if stabilize_steps > 100:
                    state = InsertCableState.DONE
                self.sleep_for(0.05)

        if state == InsertCableState.FAILED:
            self.get_logger().error("CheatCode.insert_cable() failed.")
            return False

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
