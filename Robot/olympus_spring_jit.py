from enum import IntEnum
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse
import torch
from torch import Tensor
from torch.nn.functional import normalize

from .olympus_view import OlympusView

    
class OlympusSpringJIT:
    def __init__(
        self,
        k: float,
        olympus_view: OlympusView,
        equality_dist: float,
        pulley_radius: float = 0,
    ):
        self.olympus_view = olympus_view
        self.k = torch.tensor(k, device=self.olympus_view._device)
        self.eq_dist = torch.tensor(equality_dist, device=self.olympus_view._device)
        self.r_pulley = torch.tensor(pulley_radius, device=self.olympus_view._device)

        self.front_motors_joint_indices = torch.tensor(
            [
                self.olympus_view.get_dof_index(f"FrontTransversalMotor_{pos}")
                for pos in [
                    "FL",
                    "FR",
                    "BL",
                    "BR",
                ]
            ],
            device=self.olympus_view._device,
        )
        self.back_motors_joint_indices = torch.tensor(
            [
                self.olympus_view.get_dof_index(f"BackTransversalMotor_{pos}")
                for pos in [
                    "FL",
                    "FR",
                    "BL",
                    "BR",
                ]
            ],
            device=self.olympus_view._device,
        )
        self._num_envs = self.olympus_view.count
        self.indicies = torch.tensor(
            [
                self.olympus_view.get_dof_index(f"{pos}TransversalMotor_{quad}")
                for quad in ["FL", "FR", "BL", "BR"]
                for pos in ["Front", "Back"]
            ],
            device=self.olympus_view._device,
        )
        self.batched_indicies = self.indicies.tile((self._num_envs, 1))
        self._compute_taus_jit = torch.jit.script(_compute_taus)

    def forward(self) -> ArticulationAction:
        """
        calculates the equivalent force/tourque from the sprig
        returns: a articulation action with the equivalent force/tourque
        """
        motor_housing_rot = self._get_motor_housings_rot()
        front_motor_pos = self._get_front_motors_pos()
        back_motor_pos = self._get_back_motors_pos()
        front_knee_pos = self._get_front_knees_pos()
        back_knee_pos = self._get_back_knees_pos()
        front_motor_joint_pos = self._get_front_motors_joint_pos()
        back_motor_joint_pos = self._get_back_motors_joint_pos()
        joint_efforts = self._compute_taus_jit(
            motor_housing_rot,
            front_motor_pos,
            front_knee_pos,
            back_motor_pos,
            back_knee_pos,
            front_motor_joint_pos,
            back_motor_joint_pos,
            self.eq_dist,
            self.k,
            self.r_pulley,
        )
                                           
        return ArticulationAction(
            joint_efforts=joint_efforts,
            joint_indices=self.batched_indicies,
        )
    
    def _get_motor_housings_rot(self) -> Tensor:
        _,fl= self.olympus_view.MotorHousing_FL.get_world_poses()
        _,fr= self.olympus_view.MotorHousing_FR.get_world_poses()
        _,bl= self.olympus_view.MotorHousing_BL.get_world_poses()
        _,br= self.olympus_view.MotorHousing_BR.get_world_poses()
        return torch.concatenate([fl, fr, bl, br], dim=0)
    
    def _get_front_motors_pos(self) -> Tensor:
        fl,_ = self.olympus_view.FrontMotor_FL.get_world_poses()
        fr,_ = self.olympus_view.FrontMotor_FR.get_world_poses()
        bl,_ = self.olympus_view.FrontMotor_BL.get_world_poses()
        br,_ = self.olympus_view.FrontMotor_BR.get_world_poses()
        return torch.concatenate([fl, fr, bl, br], dim=0)

    def _get_back_motors_pos(self) -> Tensor:
        fl,_ = self.olympus_view.BackMotor_FL.get_world_poses()
        fr,_ = self.olympus_view.BackMotor_FR.get_world_poses()
        bl,_ = self.olympus_view.BackMotor_BL.get_world_poses()
        br,_ = self.olympus_view.BackMotor_BR.get_world_poses()
        return torch.concatenate([fl, fr, bl, br], dim=0)

    def _get_front_knees_pos(self) -> Tensor:
        fl,_ = self.olympus_view.FrontKnee_FL.get_world_poses()
        fr,_ = self.olympus_view.FrontKnee_FR.get_world_poses()
        bl,_ = self.olympus_view.FrontKnee_BL.get_world_poses()
        br,_ = self.olympus_view.FrontKnee_BR.get_world_poses()
        return torch.concatenate([fl, fr, bl, br], dim=0)
    
    def _get_back_knees_pos(self) -> Tensor:
        fl,_ = self.olympus_view.BackKnee_FL.get_world_poses()
        fr,_ = self.olympus_view.BackKnee_FR.get_world_poses()
        bl,_ = self.olympus_view.BackKnee_BL.get_world_poses()
        br,_ = self.olympus_view.BackKnee_BR.get_world_poses()
        return torch.concatenate([fl, fr, bl, br], dim=0)
    
    def _get_front_motors_joint_pos(self) -> Tensor:
        joint_pos = self.olympus_view.get_joint_positions(joint_indices=self.front_motors_joint_indices, clone=True)
        return joint_pos.T.flatten()
    
    def _get_back_motors_joint_pos(self) -> Tensor:
        joint_pos = self.olympus_view.get_joint_positions(joint_indices=self.back_motors_joint_indices, clone=True)
        return joint_pos.T.flatten()
    

def _get_action_both_normal(
        front_motor_pos: Tensor,
        back_motor_pos: Tensor,
        front_knee_pos: Tensor,
        back_knee_pos: Tensor,
        eq_dist: Tensor,
        k: Tensor,
        ) -> Tensor:

        r_b_f = front_knee_pos - back_knee_pos
        dist = torch.norm(r_b_f, dim=1)
        s = dist - eq_dist
        actions = torch.zeros((s.shape[0], 2), device=s.device)
        mask = s > 0
        r_norm = normalize(r_b_f[mask])
        F = k * (s[mask]).unsqueeze(1) * r_norm
        tourqe_front = torch.cross((front_knee_pos[mask] - front_motor_pos[mask]), -F, dim=1)
        tourqe_back = torch.cross((back_knee_pos[mask] - back_motor_pos[mask]), F, dim=1)
        tau_front = -torch.norm(tourqe_front, dim=1)
        tau_back = -torch.norm(tourqe_back, dim=1)
        actions[mask] =torch.stack([tau_front, tau_back], dim=1)
        return actions

def _get_action_both_inverted(
    front_motor_pos: Tensor,
    back_motor_pos: Tensor,
    front_knee_pos: Tensor,
    front_motor_joint_pos: Tensor,
    back_motor_joint_pos: Tensor,
    eq_dist: Tensor,
    k: Tensor,
    r_pulley: Tensor,
    ) -> Tensor:

    l1 = torch.norm(front_knee_pos[0] - front_motor_pos[0])
    alpha = torch.acos(r_pulley / l1)
    gamma_front = front_motor_joint_pos - alpha
    gamma_back = back_motor_joint_pos - alpha
    l2 = torch.sqrt(l1**2 - r_pulley**2)
    d = torch.norm(front_motor_pos[0] - back_motor_pos[0])
    s = 2 * l2 + r_pulley * (gamma_back + gamma_front) + d - eq_dist
    s[s < 0] = 0
    tau = -k * s * r_pulley
    return torch.stack([tau, tau], dim=1)

def _get_action_front_normal_back_inverted(
    front_motor_pos: Tensor,
    back_motor_pos: Tensor,
    front_knee_pos: Tensor,
    front_motor_joint_pos: Tensor,
    back_motor_joint_pos: Tensor,
    eq_dist: Tensor,
    k: Tensor,
    r_pulley: Tensor,
    ) -> Tensor:

    r_fk_bm = back_motor_pos - front_knee_pos
    r_fk_fm = back_motor_pos - front_knee_pos
    d = torch.norm(front_motor_pos[0] - back_motor_pos[0])
    H = torch.norm(r_fk_bm, dim=1)
    m = torch.sqrt(H**2 -r_pulley**2)

    cos_angle_1 = (torch.bmm(normalize(r_fk_bm).unsqueeze(1), normalize(r_fk_fm).unsqueeze(2))).squeeze_().clamp(-1,1)
    angle_1 = torch.acos(cos_angle_1)
    angle_2 = torch.asin(r_pulley / H)
    beta_back = torch.pi / 2 - front_motor_joint_pos - angle_1 - angle_2
    l_thigh = torch.norm(r_fk_fm[0])
    l2 = torch.sqrt(l_thigh**2 - r_pulley**2)
    alpha = torch.acos(r_pulley / l_thigh)
    gamma_back = back_motor_joint_pos - alpha

    s = l2 + r_pulley * (gamma_back - beta_back) + m - eq_dist
    s[s < 0] = 0
    tau_back = -k * s * r_pulley
    tau_front = -k * s * torch.sin(angle_1 + angle_2)
    return torch.concatenate([tau_front.reshape(-1, 1), tau_back.reshape(-1, 1)], dim=1)

def _get_action_front_inverted_back_normal(
    front_motor_pos: Tensor,
    back_motor_pos: Tensor,
    back_knee_pos: Tensor,
    front_motor_joint_pos: Tensor,
    back_motor_joint_pos: Tensor,
    eq_dist: Tensor,
    k: Tensor,
    r_pulley: Tensor,
    ) -> Tensor:

    r_bk_fm = front_motor_pos - back_knee_pos
    r_bk_bm = back_motor_pos - back_knee_pos

    d = torch.norm(front_motor_pos[0] - back_motor_pos[0])
    H = torch.norm(front_motor_pos - back_knee_pos, dim=1)
    m = torch.sqrt(H**2 - r_pulley**2)

    cos_angle_1 = (torch.bmm(normalize(r_bk_fm).unsqueeze(1), normalize(r_bk_bm).unsqueeze(2))).squeeze().clamp(-1,1)
    angle_1 = torch.acos(cos_angle_1) 
    angle_2 = torch.asin(r_pulley / H)
    beta_front = torch.pi / 2 - back_motor_joint_pos - angle_1 - angle_2
    l_thigh = torch.norm(r_bk_bm[0])
    l2 = torch.sqrt(l_thigh**2 - r_pulley**2)
    alpha = torch.acos(r_pulley / l_thigh)
    gamma_front = front_motor_joint_pos - alpha

    s = l2 + r_pulley * (gamma_front - beta_front) + m - eq_dist
    s[s < 0] = 0
    tau_front = -k * s * r_pulley
    tau_back = -k * s * torch.sin(angle_1 + angle_2)

    return torch.concatenate([tau_front.reshape(-1, 1), tau_back.reshape(-1, 1)], dim=1)

def get_mode(
        motor_housing_rot: Tensor,
        front_motor_pos: Tensor,
        front_knee_pos: Tensor,
        back_motor_pos: Tensor,
        back_knee_pos: Tensor,
        r_pulley: Tensor,
) -> Tensor:
    # tranform everthing to the motor housing frame
    front_motor_pos = quat_rotate_inverse(motor_housing_rot, front_motor_pos)
    front_knee_pos  = quat_rotate_inverse(motor_housing_rot, front_knee_pos)
    back_motor_pos  = quat_rotate_inverse(motor_housing_rot, back_motor_pos)
    back_knee_pos   = quat_rotate_inverse(motor_housing_rot, back_knee_pos)
    # check if the knees are below the pulley
    r_bk_bm = back_motor_pos - back_knee_pos
    r_fk_fm = front_motor_pos - front_knee_pos
    front_knee_below = r_bk_bm[:,0] > r_pulley
    back_knee_below = r_fk_fm[:,0] > r_pulley
    modes = torch.zeros_like(back_knee_below).long()
    modes[torch.logical_and(~back_knee_below, ~front_knee_below)] = 1

    back_above_front_below = torch.logical_and(front_knee_below, ~back_knee_below)
    
    sin_tresh = r_pulley / torch.norm(r_bk_bm[0])
    r_bk_fk = front_knee_pos - back_knee_pos
    sin_angle = torch.cross(normalize(r_bk_bm),normalize(r_bk_fk),dim=1)[:,1]
    modes[torch.logical_and(back_above_front_below, sin_angle < sin_tresh)] = 2

    front_above_back_below = torch.logical_and(~front_knee_below, back_knee_below)
   
    sin_tresh = r_pulley / torch.norm(r_fk_fm[0])
    r_fk_bk = back_knee_pos - front_knee_pos
    sin_angle = -torch.cross(normalize(r_fk_fm),normalize(r_fk_bk),dim=1)[:,1]
    modes[torch.logical_and(front_above_back_below, sin_angle < sin_tresh)] = 3
    
    return modes.view(-1)



def _compute_taus(
    motor_housing_rot: Tensor,
    front_motor_pos: Tensor,
    front_knee_pos: Tensor,
    back_motor_pos: Tensor,
    back_knee_pos: Tensor,
    front_motor_joint_pos: Tensor,
    back_motor_joint_pos: Tensor,
    eq_dist: Tensor,
    k: Tensor,
    r_pulley: Tensor,
) -> Tensor:
    

    modes = get_mode(
        motor_housing_rot=motor_housing_rot,
        front_motor_pos=front_motor_pos,
        front_knee_pos=front_knee_pos,
        back_motor_pos=back_motor_pos,
        back_knee_pos=back_knee_pos,
        r_pulley=r_pulley,
    )
    actions = torch.zeros((modes.shape[0], 2), device=modes.device)


    #add dummy row to modes to deal with the case where some modes are not present
    pad_pos = torch.zeros(1,3,device=modes.device)
    pad_joint = torch.zeros(1,device=modes.device)
    mask = modes == 0
    
    front_motor_pos_0 = torch.cat([front_motor_pos[mask], pad_pos], dim=0)
    back_motor_pos_0 = torch.cat([back_motor_pos[mask], pad_pos], dim=0)
    front_knee_pos_0 = torch.cat([front_knee_pos[mask], pad_pos], dim=0)
    back_knee_pos_0 = torch.cat([back_knee_pos[mask],pad_pos], dim=0)
    actions[mask]= _get_action_both_normal(
        front_motor_pos_0,
        back_motor_pos_0,
        front_knee_pos_0,
        back_knee_pos_0,
        eq_dist,
        k,
    )[:-1,:]

    mask = modes == 1
    front_motor_pos_1 = torch.cat([front_motor_pos[mask], pad_pos], dim=0)
    back_motor_pos_1 = torch.cat([back_motor_pos[mask], pad_pos], dim=0)
    front_knee_pos_1 = torch.cat([front_knee_pos[mask], pad_pos], dim=0)
    front_motor_joint_pos_1 = torch.cat([front_motor_joint_pos[mask],pad_joint], dim=0)
    back_motor_joint_pos_1 = torch.cat([back_motor_joint_pos[mask],pad_joint], dim=0)

    actions[mask] = _get_action_both_inverted(
        front_motor_pos_1,
        back_motor_pos_1,
        front_knee_pos_1,
        front_motor_joint_pos_1,
        back_motor_joint_pos_1,
        eq_dist,
        k,
        r_pulley,
    )[:-1,:]

    mask = modes == 2
    front_motor_pos_2 = torch.cat([front_motor_pos[mask], pad_pos], dim=0)
    back_motor_pos_2 = torch.cat([back_motor_pos[mask], pad_pos], dim=0)
    front_knee_pos_2 = torch.cat([front_knee_pos[mask], pad_pos], dim=0)
    front_motor_joint_pos_2 = torch.cat([front_motor_joint_pos[mask],pad_joint], dim=0)
    back_motor_joint_pos_2 = torch.cat([back_motor_joint_pos[mask],pad_joint], dim=0)

    actions[mask] = _get_action_front_normal_back_inverted(
        front_motor_pos_2,
        back_motor_pos_2,
        front_knee_pos_2,
        front_motor_joint_pos_2,
        back_motor_joint_pos_2,
        eq_dist,
        k,
        r_pulley,
    )[:-1,:]

    mask = modes == 3
    front_motor_pos_3 = torch.cat([front_motor_pos[mask], pad_pos], dim=0)
    back_motor_pos_3 = torch.cat([back_motor_pos[mask], pad_pos], dim=0)
    back_knee_pos_3 = torch.cat([back_knee_pos[mask], pad_pos], dim=0)
    front_motor_joint_pos_3 = torch.cat([front_motor_joint_pos[mask],pad_joint], dim=0)
    back_motor_joint_pos_3 = torch.cat([back_motor_joint_pos[mask],pad_joint], dim=0)

    actions[mask] = _get_action_front_inverted_back_normal(
        front_motor_pos_3,
        back_motor_pos_3,
        back_knee_pos_3,
        front_motor_joint_pos_3,
        back_motor_joint_pos_3,
        eq_dist,
        k,
        r_pulley,
    )[:-1,:]

    num_envs = actions.shape[0] // 4
    joint_efforts = torch.concatenate(
        [actions[i * num_envs : (i + 1) * num_envs, :] for i in range(4)], dim=1
    )

    return joint_efforts