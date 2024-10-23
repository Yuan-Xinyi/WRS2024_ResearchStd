import os
import numpy as np
import wrs.modeling.model_collection as mc
from wrs import rm, mgm
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gp
import wrs.robot_sim._kinematics.jlchain as rkjlc



class CobottaPipette(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='cobotta_pipette', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.lnk_list[-1].gl_pos
        cpl_end_rotmat = self.coupling.lnk_list[-1].gl_rotmat
        self.jlc = rkjlc.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, n_dof=9, name='base_jlc')
        self.jlc.jnts[0].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[1].loc_pos = np.array([0, .0, .0])
        # self.jlc.jnts[2]._type = 'fixed'
        self.jlc.jnts[2].loc_pos = np.array([0, .0, .0])
        # self.jlc.jnts[3]._type = 'fixed'
        self.jlc.jnts[3].loc_pos = np.array([0, .0, .0])
        # self.jlc.jnts[4]['end_type'] = 'fixed'
        self.jlc.jnts[4].loc_pos= np.array([0, -.007, .0])
        self.jlc.jnts[4].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] ]))
        self.jlc.jnts[4].motion_range = [0, .015]
        self.jlc.jnts[4].loc_motionax = np.array([0, 1, 0])
        self.jlc.jnts[5].loc_pos = np.array([0, .0, .0])
        # self.jlc.jnts[6]['end_type'] = 'fixed'
        self.jlc.jnts[6].loc_pos = np.array([0, .014, .0])
        self.jlc.jnts[6].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] ]))
        self.jlc.jnts[6].loc_motionax = np.array([0, 1, 0])
        self.jlc.jnts[7].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[7].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] ]))
        self.jlc.jnts[7].loc_motionax = np.array([0, 1, 0])
        self.jlc.jnts[0].lnk.name = "base"
        self.jlc.jnts[0].lnk.loc_pos = np.zeros(3)
        self.jlc.jnts[0].lnk.mesh_file = os.path.join(this_dir, "meshes", "pipette_hand_body.stl")
        self.jlc.jnts[0].lnk.rgba = [.35, .35, .35, 1]
        self.jlc.jnts[1].lnk.name = "cam_front"
        self.jlc.jnts[1].lnk.loc_pos = np.array([.008, .04, .08575])
        self.jlc.jnts[1].lnk.mesh_file = os.path.join(this_dir, "meshes", "camera.stl")
        self.jlc.jnts[1].lnk.rgba = [.2, .2, .2, 1]
        self.jlc.jnts[2].lnk.name = "cam_back"
        self.jlc.jnts[2].lnk.loc_pos = np.array([.008, .04, .03575])
        self.jlc.jnts[2].lnk.mesh_file = os.path.join(this_dir, "meshes", "camera.stl")
        self.jlc.jnts[2].lnk.rgba = [.2, .2, .2, 1]
        self.jlc.jnts[3].lnk.name = "pipette_body"
        self.jlc.jnts[3].lnk.loc_pos = np.array([.008, .14275, .06075])
        self.jlc.jnts[3].lnk.mesh_file = os.path.join(this_dir, "meshes", "pipette_body.stl")
        self.jlc.jnts[3].lnk.rgba = [.3, .4, .6, 1]
        self.jlc.jnts[4].lnk.name = "pipette_shaft"
        self.jlc.jnts[4].lnk.loc_pos = np.array([.008, .14275, .06075])
        self.jlc.jnts[4].lnk.mesh_file = os.path.join(this_dir, "meshes", "pipette_shaft.stl")
        self.jlc.jnts[4].lnk.rgba = [1, 1, 1, 1]
        self.jlc.jnts[5].lnk.name = "plunge"
        self.jlc.jnts[5].lnk.loc_pos = np.array([0, 0, .0])
        self.jlc.jnts[5].lnk.mesh_file = os.path.join(this_dir, "meshes", "plunge_presser.stl")
        self.jlc.jnts[5].lnk.rgba = [.5, .5, .5, 1]
        self.jlc.jnts[6].lnk.name = "plunge_button"
        self.jlc.jnts[6].lnk.loc_pos = np.array([.008, .14355, .06075])
        self.jlc.jnts[6].lnk.mesh_file = os.path.join(this_dir, "meshes", "pipette_plunge.stl")
        self.jlc.jnts[6].lnk.rgba = [1, 1, 1, 1]
        self.jlc.jnts[7].lnk.name = "ejection"
        self.jlc.jnts[7].lnk.loc_pos = np.array([0, 0, .0])
        self.jlc.jnts[7].lnk.mesh_file = os.path.join(this_dir, "meshes", "ejection_presser.stl")
        self.jlc.jnts[7].lnk.rgba = [.5, .5, .5, 1]
        self.jlc.jnts[8].lnk.name = "ejection_button"
        self.jlc.jnts[8].lnk.loc_pos = np.array([.008, .14355, .06075])
        self.jlc.jnts[8].lnk.mesh_file = os.path.join(this_dir, "meshes", "pipette_ejection.stl")
        self.jlc.jnts[8].lnk.rgba = [1, 1, 1, 1]
        # jaw range
        self.jaw_range = [0.0, .03]
        # jaw center
        self.jaw_center_pos = np.array([0.008, 0.14305, 0.06075])
        self.jaw_center_rotmat = rm.rotmat_from_axangle([1, 0, 0], -np.pi / 2)
        # reinitialize
        self.jlc.finalize()
        # collision detection
        self.all_cdelements = []
        # self.enable_cc(toggle_cdprimit=enable_cc)

    # def enable_cc(self, toggle_cdprimit):
    #     if toggle_cdprimit:
    #         super().enable_cc()
    #         # cdprimit
    #         self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 7])
    #         active_list = [self.jlc.lnk[0],
    #                        self.jlc.lnk[1],
    #                        self.jlc.lnk[2],
    #                        self.jlc.lnk[4],
    #                        self.jlc.lnk[5],
    #                        self.jlc.lnk[7]]
    #         self.cc.set_active_cdlnks(active_list)
    #         self.all_cdelements = self.cc.cce_dict
    #     # cdmesh
    #     for cdelement in self.all_cdelements:
    #         cdmesh = cdelement['collision_model'].copy()
    #         self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self._pos = pos
        self._rotmat = rotmat
        if jaw_width is not None:
            side_jawwidth = jaw_width / 2.0
            if self.jaw_range[1] < jaw_width or jaw_width < self.jaw_range[0]:
                self.jlc.jnts[5]['motion_value'] = side_jawwidth
                self.jlc.jnts[7]['motion_value'] = -jaw_width
                if side_jawwidth <= .007:
                    self.jlc.jnts[8]['motion_value'] = .0
                else:
                    self.jlc.jnts[8]['motion_value'] = (jaw_width - .014) / 2
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self._pos, self._rotmat)
        cpl_end_pos = self.coupling.jnts[-1].gl_pos_q
        cpl_end_rotmat = self.coupling.jnts[-1].gl_rotmat_q
        self.jlc.fix_to(cpl_end_pos, cpl_end_rotmat)

    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.goto_given_conf(jnt_values=[side_jawwidth, jaw_width])
        else:
            raise ValueError("The angle parameter is out of range!")

    def get_jaw_width(self):
        return self.jlc.jnts[1].motion_value

    # def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='ee_stickmodel'):
    #     stickmodel = mc.ModelCollection(name=name)
    #     self.coupling.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frames).attach_to(stickmodel)
    #     self.jlc.gen_stickmodel(toggle_tcpcs=False,
    #                             toggle_jntscs=toggle_jnt_frames,
    #                             toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
    #     if toggle_tcp_frame:
    #         jaw_center_gl_pos = self._rotmat.dot(self.jaw_center_pos) + self._pos
    #         jaw_center_gl_rotmat = self._rotmat.dot(self.jaw_center_rotmat)
    #         mgm.gen_dashed_stick(spos=self._pos,
    #                             epos=jaw_center_gl_pos,
    #                             radius=.0062,
    #                             rgba=[.5, 0, 1, 1],
    #                             type="round").attach_to(stickmodel)
    #         mgm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)
    #     return stickmodel

    def gen_meshmodel(self,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      rgba=None,
                      name='cbtp_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_mesh_model(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jnt_frames,
                                     rgba=rgba).attach_to(meshmodel)
        self.jlc.gen_mesh_model(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                rgba=rgba).attach_to(meshmodel)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self._rotmat.dot(self.jaw_center_pos) + self._pos
            jaw_center_gl_rotmat = self._rotmat.dot(self.jaw_center_rotmat)
            mgm.gen_dashed_stick(spos=self._pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(meshmodel)
            mgm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = CobottaPipette(enable_cc=True)
    grpr.change_jaw_width(.0)
    grpr.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    grpr.gen_stickmodel().attach_to(base)
    # grpr.gen_stickmodel(toggle_jnt_frames=False).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()
