import copy

import numpy as np
import math
from wrs import mgm,mcm,rm,wd
from panda3d.core import CollisionNode, CollisionBox, Point3
import wrs.robot_sim.robots.cobotta.cobotta_ripps as cbtr
import utils_rack as utils
import config_file as conf


class Env(object):
    def __init__(self, rbt_s, armname="arm"):
        self.rbt_s = rbt_s
        self.armname = armname
        self.init_height = -conf.base_height
        self.pipette_pos = self.rbt_s.gl_tcp_pos - np.array([-0.008, -0.14085, 0.06075])
        self.tcp_loc_pos = self.rbt_s.gl_tcp_pos
        self.env_build()

        # self.load_pipette()

    def load_pipette(self):
        self.outside = mcm.CollisionModel("./meshes/model_base.stl", cdprimit_type="box")
        pipettemat4 = rm.homomat_from_posrot(self.pipette_pos, np.eye(3))
        eepos, eerot = self.rbt_s.get_gl_tcp(manipulator_name=self.armname)
        tcpmat4 = np.dot(rm.homomat_from_posrot(eepos, eerot), np.linalg.inv(pipettemat4))
        self.outside.set_rgba([0., 0.4, 0.8, 0.6])
        self.outside.set_pos(tcpmat4[:3, 3])
        self.outside.set_rotmat(tcpmat4[:3, :3])
        self.rbt_s.hold(self.armname, self.outside)
        self.pipette = mcm.CollisionModel("./meshes/model_tip.stl", cdprimit_type="box", expand_radius=0.01)
        pipettemat4 = rm.homomat_from_posrot(self.pipette_pos, np.eye(3))
        eepos, eerot = self.rbt_s.get_gl_tcp(manipulator_name=self.armname)
        tcpmat4 = np.dot(rm.homomat_from_posrot(eepos, eerot), np.linalg.inv(pipettemat4))
        self.pipette.set_rgba([0, 0.4, 0.8, 0.8])
        self.pipette.set_pos(tcpmat4[:3, 3])
        self.pipette.set_rotmat(tcpmat4[:3, :3])
        self.rbt_s.hold(self.armname, self.pipette)

    def env_build(self):
        self.table_plate = mcm.gen_box(xyz_lengths=[.6, .8, .03])
        self.table_plate.pos = np.array([0.2, 0, self.init_height - 0.015])
        self.table_plate.rgba = np.array([.83, .74, .44, 1])
        self.table_plate.attach_to(base)

        def _balance_combined_cdnp(ex_radius):
            collision_node = CollisionNode("balance")
            collision_primitive_c0 = CollisionBox(Point3(0.214, 0, 0.025),
                                                  x=.214 + ex_radius, y=.1 + ex_radius, z=.025 + ex_radius)
            collision_node.addSolid(collision_primitive_c0)
            collision_primitive_c1 = CollisionBox(Point3(0.332, 0, 0.152),
                                                  x=.096 + ex_radius, y=.1 + ex_radius, z=.152 + ex_radius)
            collision_node.addSolid(collision_primitive_c1)
            collision_primitive_c2 = CollisionBox(Point3(0.072, 0, 0.152),
                                                  x=.001 + ex_radius, y=.1 + ex_radius, z=.152 + ex_radius)
            collision_node.addSolid(collision_primitive_c2)
            collision_primitive_c3 = CollisionBox(Point3(0.17, 0, 0.30),
                                                  x=.1 + ex_radius, y=.1 + ex_radius, z=.005 + ex_radius)
            collision_node.addSolid(collision_primitive_c3)
            return collision_node

        self.balance = mcm.CollisionModel(initor="./meshes/balance_left.stl",
                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                         ex_radius=.005,
                                         userdef_cdprim_fn=_balance_combined_cdnp)
        self.balance.rgba=([200 / 255, 180 / 255, 140 / 255, 1])
        self.balance.pos=(np.array([0.12, -0.255, self.init_height]))
        self.balance.attach_to(base)

        ## 200ul rack
        self.tip_rack = utils.Base("./meshes/rack_mbp.stl")
        rack_pos = np.array([0.2671, -0.0257, self.init_height])
        # rack_pos = np.array([.0495, .0315, 0])
        self.tip_rack.rgba=([140 / 255, 110 / 255, 170 / 255, 1])
        self.tip_rack.pos=(rack_pos)
        self.tip_rack.rotmat=(rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2))
        # self.tip_rack.attach_to(base)

        self.dispose_box = mcm.CollisionModel("./meshes/tip_rack_cover.stl", ex_radius=.007)
        self.dispose_box.rgba=([140 / 255, 110 / 255, 170 / 255, 1])
        self.dispose_box.pos=np.array([.17, -0.095, .003 + self.init_height])
        # dispose_box.set_rotmat(rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi / 2))
        self.dispose_box.attach_to(base)

        self.microplate = utils.Microplate24("./meshes/microplate_24.stl")
        self.microplate.set_homomat(conf.microplate_homo)
        # for tip_pos in self.microplate._hole_pos_list:
        #     print(tip_pos)
        #     mgm.gen_sphere(pos=tip_pos, radius=0.005,rgb=[0,0,1]).attach_to(base)
        # mgm.gen_frame(conf.microplate_pos, conf.microplate_rot).attach_to(base)
        self.microplate.attach_to(base)

        plate_balance_homo = rm.homomat_from_posrot(np.array([0.27, -0.24, 0.05 + self.init_height]))
        self.plate_balance = self.microplate.copy()
        self.plate_balance.homomat=(plate_balance_homo)
        self.plate_balance.attach_to(base)

    def update_env(self, rack_pos=None, rack_rotmat=None, microplate_pos=None, microplate_rotmat=None):
        self.tip_rack.detach()
        if rack_pos is not None:
            self.tip_rack.pos=(rack_pos)
        if rack_rotmat is not None:
            self.tip_rack.rotmat=(rack_rotmat)
        if microplate_pos is not None:
            self.microplate.pos=(microplate_pos)
        if microplate_rotmat is not None:
            self.microplate.rotmat=(microplate_rotmat)


if __name__ == '__main__':
    base = wd.World(cam_pos=[-0.331782, 1.2348, 0.634336], lookat_pos=[-0.0605939, 0.649106, 0.311471])
    # mgm.gen_frame().attach_to(base)
    component_name = 'arm'
    robot_s = cbtr.CobottaRIPPS()
    env = Env(robot_s)

    # env.tip_rack.attach_to(base)
    # tip = mcm.CollisionModel("./meshes/tip.stl")
    # tip.rgba=([200 / 255, 180 / 255, 140 / 255, 1])

    # env.update_env(microplate_pos=conf.microplate_pos, microplate_rotmat=conf.microplate_rot)
    # env.microplate.homomat=(conf.microplate_homo)
    # env.microplate.attach_to(base)

    # box0 = env.tip_rack

    for tid in range(24):
        t_pos = env.microplate._hole_pos_list[tid % 24]
        print(t_pos)
        mgm.gen_sphere(pos=t_pos, radius=0.001 + 0.001 * tid).attach_to(base)
    # current_jnts = np.array([0.08685238, 0.72893128, 1.2966003, 1.90433666, 1.02620525, -0.51833472])
    eject_jnt_values = np.array([-0.92587354, 0.29804923, 2.05512946, 0.8513571, -0.90180492,
                                 -0.05923206])
    robot_s.fk(jnt_values=eject_jnt_values)
    robot_s.gen_meshmodel().attach_to(base)

    # chemical_pos = copy.deepcopy(env.microplate._hole_pos_list[0])
    # chemical_pos[:3] += conf.adjust_pos
    # chemical_pos[2] = conf.chemical_height - 0.05
    # mgm.gen_sphere(pos=chemical_pos,radius=0.1 ,rgb=[0,1,0]).attach_to(base)


    base.run()
