from __future__ import annotations

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms
import isaaclab.sim as sim_utils

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SfpPoseTargetCommand(CommandTerm):
    """Command generator that selects between the two pre-defined SFP ports poses in the NIC card."""

    cfg: SfpPoseTargetCommandCfg

    def __init__(self, cfg: SfpPoseTargetCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
                
        # The sensor that tracks the port frames relative to the asset
        self.sensor = env.scene.sensors[cfg.sensor_name]

        # Command buffer: (N, 7) -> [pos, quat] relative to asset
        self.poses_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.poses_w = torch.zeros_like(self.poses_b)
        self.targets_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.pending_targets_idx = None
        
        # Mapping from logical index (0: port_0, 1: port_1) to physical sensor index
        self._port_indices = torch.tensor([0, 1], device=self.device)
        self._port_indices_resolved = False

    def _resolve_port_indices(self):
        """Resolves the physical indices of 'port_0' and 'port_1' from the sensor data. Order may change wrt sensor creation."""
        if self._port_indices_resolved:
            return
        
        # Check if sensor data is available
        if not hasattr(self.sensor, "data") or self.sensor.data.target_frame_names is None:
            return

        frame_names = self.sensor.data.target_frame_names
        try:
            p0_idx = frame_names.index("port_0")
            p1_idx = frame_names.index("port_1")
            self._port_indices = torch.tensor([p0_idx, p1_idx], device=self.device)
            self._port_indices_resolved = True
        except ValueError:
            raise ValueError(f"port_0 or port_1 not found in sensor {self.cfg.sensor_name} frame_names")

    def _resample_command(self, env_ids: torch.Tensor):
        self._resolve_port_indices()
        
        if self.pending_targets_idx is not None:
            # Check for pending overrides (where value != -1)
            is_pending = self.pending_targets_idx[env_ids] != -1
            
            # Use pending target if available (interpreted as logical index)
            if is_pending.any():
                logical_idx = self.pending_targets_idx[env_ids[is_pending]]
                self.targets_idx[env_ids[is_pending]] = self._port_indices[logical_idx]
            
            # Randomly select for others
            if (~is_pending).any():
                rand_logical_idx = torch.randint(0, 2, ((~is_pending).sum(),), device=self.device)
                self.targets_idx[env_ids[~is_pending]] = self._port_indices[rand_logical_idx]
            
            # Clear pending overrides for the resampled envs
            self.pending_targets_idx[env_ids] = -1
        else:
            # Randomly select between logical port 0 and 1
            rand_logical_idx = torch.randint(0, 2, (len(env_ids),), device=self.device)
            self.targets_idx[env_ids] = self._port_indices[rand_logical_idx]

        # Immediate update of command buffers for fresh data
        self._update_command()
        self._debug_vis_callback(None)

    def _update_command(self):
        self._resolve_port_indices()
        
        # Fetch the current relative poses from the sensor
        self.poses_b[:, :3] = self.sensor.data.target_pos_source[torch.arange(self.num_envs), self.targets_idx]
        self.poses_b[:, 3:] = self.sensor.data.target_quat_source[torch.arange(self.num_envs), self.targets_idx]
       
        # Fetch world pose of active target for visualization
        self.poses_w[:, :3] = self.sensor.data.target_pos_w[torch.arange(self.num_envs), self.targets_idx]
        self.poses_w[:, 3:] = self.sensor.data.target_quat_w[torch.arange(self.num_envs), self.targets_idx]

    def _update_metrics(self):
        # No metrics to compute
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "visualizer"):
                self.visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.visualizer.set_visibility(True)
        else:
            if hasattr(self, "visualizer"):
                self.visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        if not hasattr(self.sensor, "data") or self.sensor.data.target_pos_w is None:
            return
        marker_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.visualizer.visualize(self.poses_w[:, :3], self.poses_w[:, 3:], marker_indices=marker_indices)

    @property
    def command(self) -> torch.Tensor:
        return self.poses_b
    
    @property
    def logical_targets_idx(self) -> torch.Tensor:
        """Returns the logical indices (0 for port_0, 1 for port_1) of the active targets."""
        self._resolve_port_indices()
        # Find which logical index each physical index corresponds to
        # targets_idx contains physical indices. _port_indices contains [p0_phys, p1_phys]
        return (self.targets_idx == self._port_indices[1]).long()

@configclass
class SfpPoseTargetCommandCfg(CommandTermCfg):
    """Configuration for a discrete pose target command (selecting between ports)."""
    class_type: type = SfpPoseTargetCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9) # never resample automatically
    debug_vis: bool = True
    sensor_name: str = "sfp_port_sensor" # sensor that tracks the ports
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CommandFrame",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.02, 0.02, 0.02),
            ),
        },
    )
