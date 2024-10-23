import numpy as np

if __name__ == "__main__":
    import cobotta2 as cbtx
    from numpy import array
    import basis.robot_math as rm
    import file_sys as fs

    rbtx = cbtx.CobottaX()

    # p = array([2.90139458e-01, 2.37984829e-02, .203, 1.57136404e+00,
    #        -7.59717102e-04, 1.67314437e+00, 5.00000000e+00])

    rbt_rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T

    data = fs.load_json("resources/collect_data.json")
    rot_deg = data['deg']
    if f'data_{rot_deg}' not in data or 'rot' not in data[f'data_{rot_deg}']:
        rbt_rot_r = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(rot_deg)).dot(rbt_rot)
        data[f'data_{rot_deg}'] = data.get(f'data_{rot_deg}', {})
        data[f'data_{rot_deg}']['rot'] = rbt_rot_r.tolist()
    else:
        rbt_rot_r = np.asarray(data[f'data_{rot_deg}']['rot'])

    if f'data_{rot_deg}' not in data or 'pos' not in data[f'data_{rot_deg}']:
        p = array([0.29244215, 0.02199824, .203])
        data[f'data_{rot_deg}'] = data.get(f'data_{rot_deg}', {})
        data[f'data_{rot_deg}']['pos'] = p.tolist()
    else:
        p = np.asarray(data[f'data_{rot_deg}']['pos'])
    # p1 = np.array([.29, .024, .203])

    rbtx.move_p(p, rbt_rot_r)
    jnt = rbtx.get_jnt_values()

    # if f'data_{rot_deg}' not in data or 'joint_val' not in data[f'data_{rot_deg}']:
    #     data[f'data_{rot_deg}']['joint_val'] = jnt.tolist()

    fs.dump_json(data, "collect_data.json", reminder=False)

    from base_boost import boost_base
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.collision_model as cm
    import modeling.geometric_model as gm
    import basis.robot_math as rm

    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0], lens_type="perspective")
    base = boost_base(base)


    def rbt_move_func_factory(dir):
        def rbt_func():
            pos, rot = rbtx.get_pose()
            pos += dir
            rbtx.move_p(pos, rot)
            print(rbtx.get_pose())

        return rbt_func


    move_res = .0001
    base.boost.bind_task_2_key("w", rbt_move_func_factory(dir=np.array([1, 0, 0]) * move_res))
    base.boost.bind_task_2_key("s", rbt_move_func_factory(dir=np.array([-1, 0, 0]) * move_res))
    base.boost.bind_task_2_key("d", rbt_move_func_factory(dir=np.array([0, -1, 0]) * move_res))
    base.boost.bind_task_2_key("a", rbt_move_func_factory(dir=np.array([0, 1, 0]) * move_res))
    base.boost.bind_task_2_key("q", rbt_move_func_factory(dir=np.array([0, 0, -1]) * move_res))
    base.boost.bind_task_2_key("e", rbt_move_func_factory(dir=np.array([0, 0, 1]) * move_res))

    # from huri.components.utils.panda3d_utils import ImgOnscreen
    # from huri.work2.dual_d405_manager import RealSenseD405DualCrop

    # rs_pipe = RealSenseD405DualCrop()
    # img_f, img_1, img_2 = rs_pipe.get_learning_feature()
    # img_on_win = ImgOnscreen(size=(320, 160), parent_np=base)

    # def update_img(task):
    #     img_on_win.update_img(rs_pipe.get_learning_feature()[0])
    #     return task.again

    obj_p_old = [rot_deg]


    def update_obj_p(task):
        data = fs.load_json("collect_data.json")
        if obj_p_old[0] != data['deg']:
            obj_p_old[0] = data['deg']
            rot_deg = data['deg']
            if f'data_{rot_deg}' not in data or 'rot' not in data[f'data_{rot_deg}']:
                rbt_rot_r = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(rot_deg)).dot(rbt_rot)
                data[f'data_{rot_deg}'] = data.get(f'data_{rot_deg}', {})
                data[f'data_{rot_deg}']['rot'] = rbt_rot_r.tolist()
            else:
                rbt_rot_r = np.asarray(data[f'data_{rot_deg}']['rot'])

            if f'data_{rot_deg}' not in data or 'pos' not in data[f'data_{rot_deg}']:
                p = array([0.29244215, 0.02199824, .203])
                data[f'data_{rot_deg}'] = data.get(f'data_{rot_deg}', {})
                data[f'data_{rot_deg}']['pos'] = p.tolist()
            else:
                p = np.asarray(data[f'data_{rot_deg}']['pos'])
            # p1 = np.array([.29, .024, .203])

            rbtx.move_p(p, rbt_rot_r)
            jnt = rbtx.get_jnt_values()

            # if f'data_{rot_deg}' not in data or 'joint_val' not in data[f'data_{rot_deg}']:
            #     data[f'data_{rot_deg}']['joint_val'] = jnt.tolist()

            fs.dump_json(data, "collect_data.json", reminder=False)

        return task.again


    # base.boost.add_task(update_img, timestep=.1)

    base.boost.add_task(update_obj_p, timestep=.5)

    base.run()
