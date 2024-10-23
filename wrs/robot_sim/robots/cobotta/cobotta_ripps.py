import os
import math
import numpy as np
from wrs import rm, mcm, mgm, mmc
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import wrs.robot_sim.end_effectors.grippers.cobotta_pipette.cobotta_pipette_v2 as cbtp


class CobottaRIPPS(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_plate = rkjlc.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     n_dof=6,
                                     name='base_plate_ripps')
        # self.base_plate.jnts[1]['loc_pos'] = np.array([0, 0, 0.01])
        # self.base_plate.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base_plate_ripps.stl")
        self.base_plate.jnts[1].loc_pos = np.array([0, 0, -0.035])
        self.base_plate.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "base_plate.stl"))
        self.base_plate.jnts[1].lnk.cmodel.rgba= [.55, .55, .55, 1]
        self.base_plate.finalize()
        # arm
        arm_homeconf = np.zeros(6)
        arm_homeconf[1] = -math.pi / 6
        arm_homeconf[2] = math.pi / 2
        arm_homeconf[4] = math.pi / 6
        self.manipulator = cbta.CobottaArm(pos=self.pos, rotmat=self.rotmat, name=name + "_arm", enable_cc=False)

        # grippers
        self.gripper_loc_rotmat = rm.rotmat_from_axangle([0,0,1], np.pi) # 20220607 rotate the pipetting end_type with 180^o.
        self.end_effector = cbtp.CobottaPipette(pos=self.manipulator.gl_flange_pos,
                                                rotmat=self.manipulator.gl_flange_rotmat, name=name + "_hnd")

        # tool center point
        self.manipulator.jlc.flange_jnt_id = -1
        self.manipulator.jlc._loc_flange_pos = self.gripper_loc_rotmat.dot(self.end_effector.jaw_center_pos)
        self.manipulator.jlc._loc_flange_rotmat = self.gripper_loc_rotmat.dot(self.end_effector.jaw_center_rotmat)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd

        if self.cc is not None:
            self.setup_cc()



    def setup_cc(self):
        # ee
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0], toggle_extcd=False)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [elb, el0, el1, ml3, ml4, ml5]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = [el0, el1]

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def change_jaw_width(self, jaw_width):
        return self.change_ee_values(ee_values=jaw_width)

    def get_jaw_width(self):
        return self.get_ee_values()



    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='cobotta_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.base_plate.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        if self._manipulator is not None:
            self._manipulator.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                             toggle_jnt_frames=toggle_jnt_frames,
                                             toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        if self.end_effector is not None:
            self.end_effector.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                             toggle_jnt_frames=toggle_jnt_frames).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='cobotta_largehand_rack_meshmodel'):
        """

        :param tcp_jnt_id:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :param toggle_tcpcs:
        :param toggle_jntscs:
        :param rgba:
        :param name:
        :return:
        """
        m_col = mmc.ModelCollection(name=name)
        self.base_plate.gen_meshmodel(rgb=rgb,
                                      alpha=alpha,
                                      toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_flange_frame=toggle_flange_frame,
                                      toggle_cdprim=toggle_cdprim,
                                      toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        if self._manipulator is not None:
            self.manipulator.gen_meshmodel(rgb=rgb,
                                           alpha=alpha,
                                           toggle_tcp_frame=False,
                                           toggle_jnt_frames=toggle_jnt_frames,
                                           toggle_flange_frame=toggle_flange_frame,
                                           toggle_cdprim=toggle_cdprim,
                                           toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        if self._end_effector is not None:
            self.end_effector.gen_meshmodel(rgb=rgb,
                                            alpha=alpha,
                                            toggle_tcp_frame=toggle_tcp_frame,
                                            toggle_jnt_frames=toggle_jnt_frames,
                                            toggle_cdprim=toggle_cdprim,
                                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col

if __name__ == '__main__':
    import time
    # from wrs import basis as rm, robot_sim as jl, robot_sim as cbta, robot_sim as cbtp, modeling as gm
    from wrs import mgm, rm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

    mgm.gen_frame().attach_to(base)
    robot_s = CobottaRIPPS(enable_cc=True)
    # robot_s.jaw_to(.02)
    robot_s.gen_meshmodel(toggle_flange_frame=False, toggle_jnt_frames=False).attach_to(base)
    # robot_s.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frame=False).attach_to(base)
    # robot_s.gen_stickmodel(toggle_flange_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot_s.show_cdprimit()
    # base.run()
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # base.run()
    component_name = 'arm'
    # jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    mgm.gen_frame(pos=robot_s.gl_tcp_pos, rotmat=robot_s.gl_tcp_rotmat).attach_to(base)
    # robot_s.fk(component_name, jnt_values=jnt_values)
    # robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcp_frame=True)
    # robot_s_meshmodel.attach_to(base)
    # robot_s.show_cdprimit()
    # robot_s.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = robot_s.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
