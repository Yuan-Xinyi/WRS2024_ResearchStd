import os
import numpy as np
from wrs import rm, mgm, mmc, mcm
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.robot_sim._kinematics.jlchain as rkjlc



class CobottaPipette(gpi.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='cobotta_pipette', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.lnk_list[-1].gl_pos
        cpl_end_rotmat = self.coupling.lnk_list[-1].gl_rotmat

        self.jlc = rkjlc.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, n_dof=6, name='base_jlc')
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "pipette_hand_body_v3.stl"))
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.dim_gray
        # joint 0
        self.jlc.jnts[0].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[0].lnk.name = "pipette_body"
        self.jlc.jnts[0].lnk.loc_pos = np.array([.008, .14275, .06075])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "pipette_body.stl"))
        self.jlc.jnts[0].lnk.rgba = [.3, .4, .6, 1]
        # joint 1
        self.jlc.jnts[1].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[1].lnk.name = "pipette_shaft"
        self.jlc.jnts[1].lnk.loc_pos = np.array([.008, .14275, .06075])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "pipette_shaft.stl"))
        self.jlc.jnts[1].lnk.rgba = rm.const.white
        # # joint 2
        self.jlc.jnts[2].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[2].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] ]))
        self.jlc.jnts[2].motion_range = [0, .015]
        self.jlc.jnts[2].loc_motionax = np.array([0, 1, 0])
        self.jlc.jnts[2].lnk.name = "plunge"
        self.jlc.jnts[2].lnk.loc_pos = np.array([0, 0, .0])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "plunge_presser.stl"))
        self.jlc.jnts[2].lnk.rgba = [.5, .5, .5, 1]
        # # joint 3
        self.jlc.jnts[3].loc_pos= np.array([0, -.007, .0])
        self.jlc.jnts[3].lnk.name = "plunge_button"
        self.jlc.jnts[3].lnk.loc_pos = np.array([.008, .14355, .06075])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "pipette_plunge.stl"))
        self.jlc.jnts[3].lnk.rgba = [1, 1, 1, 1]
        # # joint 4
        self.jlc.jnts[4].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[4].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] ]))
        self.jlc.jnts[4].loc_motionax = np.array([0, 1, 0])
        self.jlc.jnts[4].lnk.name = "ejection"
        self.jlc.jnts[4].lnk.loc_pos = np.array([0, 0, .0])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "ejection_presser.stl"))
        self.jlc.jnts[4].lnk.rgba = [.5, .5, .5, 1]
        # # joint 5
        self.jlc.jnts[5].loc_pos = np.array([0, .014, .0])
        self.jlc.jnts[5].lnk.name = "ejection_button"
        self.jlc.jnts[5].lnk.loc_pos = np.array([.008, .14355, .06075])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "pipette_ejection.stl"))
        self.jlc.jnts[5].lnk.rgba = [1, 1, 1, 1]

        # jaw range
        self.jaw_range = [0.0, .03]
        # jaw center
        self.jaw_center_pos = np.array([0.008, 0.14305, 0.06075])
        self.jaw_center_rotmat = rm.rotmat_from_axangle([1, 0, 0], -np.pi / 2)
        # reinitialize
        self.jlc.finalize()
        # collision detection
        self.all_cdelements = []


    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self.pos
        self.coupling.rotmat = self.rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.goto_given_conf(jnt_values=[0,0,side_jawwidth,0, -side_jawwidth,0])
        else:
            raise ValueError("The angle parameter is out of range!")

    def get_jaw_width(self):
        return self.jlc.jnts[1].motion_value

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='wg3_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame=False, toggle_jnt_frames=False,
                      toggle_cdprim=False, toggle_cdmesh=False, name='wg3_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_flange_frame=False,
                                    toggle_root_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self._gen_oiee_meshmodel(m_col=m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh)
        return m_col


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    gripper = CobottaPipette(enable_cc=True)
    # grpr.change_jaw_width(.0)
    # gripper.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    # grpr.gen_stickmodel().attach_to(base)
    # grpr.gen_stickmodel(toggle_jnt_frames=False).attach_to(base)
    # grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    # grpr.gen_meshmodel().attach_to(base)

    # animation test
    tgt_jawwidth = 0.03
    current_jawwidth = gripper.get_jaw_width()
    jawwidth_path = np.linspace(current_jawwidth, tgt_jawwidth, 11)
    print(jawwidth_path)
    counter = [0]
    gripper_mesh = [None]
    def gripper_test(gripper_s, jawwidth_path, counter, gripper_mesh, task):
        if counter[0] >= len(jawwidth_path):
            counter[0] = 0
        if gripper_mesh[0] is not None:
            # gripper_s.unshow_cdprimit()
            gripper_mesh[0].detach()
            print("detach")
        gripper_s.change_jaw_width(jawwidth_path[counter[0]])
        print(counter, jawwidth_path[counter[0]])
        # gripper_s.show_cdprimit()
        gripper_mesh[0] = gripper_s.gen_meshmodel()
        gripper_mesh[0].attach_to(base)
        counter[0] += 1
        return task.again
    taskMgr.doMethodLater(0.1, gripper_test, "gripper_test", extraArgs=[gripper, jawwidth_path, counter, gripper_mesh],
                          appendTask=True)

    base.run()
