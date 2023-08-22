from guided_tamp_benchmark.tasks.demonstration import Demonstration
import matplotlib.pyplot as plt
import numpy as np
import argparse
from preprocessing.preproc_utils import (
    get_obj_poses_from_file,
    get_contact_from_file,
    smooth_contacts,
    smooth_obj_poses,
    get_cosypose_bb_from_file,
    get_handobj_bb_from_file,
    assign_contact_to_obj,
    get_all_contacts,
    get_compact_contact,
    data_folder
)

if __name__ == "__main__":
    furniture_poses = {
        'table': np.array([[1., 0., 0., 0.],
                           [0., 1., 0., -0.1],
                           [0., 0., 1., 0.75],
                           [0., 0., 0., 1.]]),
        'shelf': np.array([[1., 0., 0., 0.95],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 1.28],
                           [0., 0., 0., 1.]]),
    }
    furniture_params = {
        'table': {'desk_size': [1.5, 1.1, 0.75], 'leg_display': True},
        'shelf': {'display_inside_shelf': False}
    }
    assert furniture_poses.keys() == furniture_params.keys()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-task_names", type=str, nargs="+", default=None, help="String name of the task"
    )
    parser.add_argument(
        "-task_ids", type=int, nargs="+", default=None, help='String task id (f.e. "01"'
    )
    parser.add_argument(
        "-tray_matrix", type=str, default=None, help="String path to tray matrix"
    )
    parser.add_argument("--visualize", dest="visualize", action="store_true")
    parser.add_argument("--ensure_static", dest="ensure_static", action="store_true")
    parser.add_argument(
        "-smooth_threshold",
        type=float,
        default=0.7,
        dest="smooth_threshold",
        help="Smoothing threshold parameter",
    )
    parser.add_argument(
        "-smooth_window",
        type=int,
        default=5,
        dest="smooth_window",
        help="Smoothing window size",
    )
    parser.add_argument(
        "--replace",
        dest="replace",
        action="store_true",
        help="Replace existing robot pose files",
    )
    args = parser.parse_args()
    for task_name in args.task_names:
        for task_id in args.task_ids:
            cosypose_folder = data_folder.joinpath("cosypose")
            contacts_folder = data_folder.joinpath("handobj")

            cosypose_path = cosypose_folder.joinpath(f"{task_name}_{task_id}.pkl")
            contacts_path = contacts_folder.joinpath(f"{task_name}_{task_id}.pkl")

            cosypose_bb_path = cosypose_folder.joinpath(
                f"{task_name}_{task_id}_bb_list.pkl"
            )
            contacts_obj_path = contacts_folder.joinpath(
                f"{task_name}_{task_id}_obj_dets.pkl"
            )

            cosypose_bb = get_cosypose_bb_from_file(cosypose_bb_path)
            handobj_bb = get_handobj_bb_from_file(contacts_obj_path)

            transform_m_path = data_folder.joinpath(
                f"calibration_{task_name}_{task_id}.npy"
            )
            transform_cosypose = np.load(transform_m_path)
            tray_poses = (
                np.load(args.tray_matrix) if args.tray_matrix is not None else None
            )
            if tray_poses is not None:
                tray_poses = np.array(
                    [transform_cosypose.dot(mat) for mat in tray_poses]
                )
            obj_keys, full_obj_poses = get_obj_poses_from_file(
                cosypose_path, transform=transform_cosypose
            )
            full_contacts = get_contact_from_file(contacts_path)

            assert len(full_contacts) == full_obj_poses.shape[1]
            if args.visualize:
                subplot_amount = len(full_obj_poses)
                if tray_poses is not None:
                    subplot_amount += 1
                fig, axes = plt.subplots(subplot_amount, 1)
                if subplot_amount == 1:
                    axes.plot(full_obj_poses[0][:, :3, 3])
                else:
                    for j in range(len(full_obj_poses)):
                        axes[j].plot(full_obj_poses[j][:, :3, 3])
                if tray_poses is not None:
                    axes[len(full_obj_poses)].plot(tray_poses[:, :3, 3])
                plt.title("Object poses in time")
                plt.show()
            if args.visualize:
                plt.plot(full_contacts)
                plt.title("Initial contacts")
                plt.show()
            # contacts = smooth_contacts(full_contacts, window=args.smooth_window,
            #                            threshold=args.smooth_threshold)
            full_obj_poses = smooth_obj_poses(full_obj_poses)
            if tray_poses is not None:
                tray_poses = smooth_obj_poses(tray_poses[np.newaxis, :])[0]

            # if args.visualize:
            #     plt.plot(contacts)
            #     plt.title("Smoothed contacts")
            #     plt.show()

            assigned_contacts = assign_contact_to_obj(cosypose_bb, handobj_bb)
            full_contacts = get_all_contacts(assigned_contacts)

            if args.visualize:
                for k, v in assigned_contacts.items():
                    assigned_contacts[k] = smooth_contacts(
                        assigned_contacts[k],
                        window=args.smooth_window,
                        threshold=args.smooth_threshold,
                    )

                    plt.plot(assigned_contacts[k], label=k)
                plt.legend()
                plt.title("Assigned contacts")
                plt.show()

            # objects_poses, final_contacts, grasp_idx, release_idx =
            # get_obj_contacts(contacts, full_obj_poses)

            final_contacts, objects_poses, tray_poses_at_contact = get_compact_contact(
                assigned_contacts, full_obj_poses, tray_poses, visualize=args.visualize
            )

            if args.visualize:
                # cmap = plt.get_cmap("tab10")
                # plt.plot(contacts, color='gray')
                # for i in range(len(grasp_idx)):
                #     plt.plot(range(grasp_idx[i], release_idx[i]),
                #     contacts[grasp_idx[i]:release_idx[i]], c=cmap(i))
                # plt.show()

                fig, ax = plt.subplots()
                cmap = plt.get_cmap("tab10")
                for i, obj_id_poses in enumerate(objects_poses):
                    for j, obj_pose in enumerate(obj_id_poses):
                        ax.scatter(*obj_pose[:2, 3], c=cmap(i))
                        ax.annotate(
                            str(j), obj_pose[:2, 3] + np.array([0.007 * j, -0.005])
                        )
                plt.title("2D components of subgoal poses")
                plt.show()

            if args.replace:
                # TODO: check that tray is included correctly

                for robot in ['panda', 'ur5', 'kmr_iiwa']:
                    demo = Demonstration()
                    demo.task_name = task_name
                    demo.demo_id = task_id
                    demo.robot_name = robot
                    demo.pose_id = 0
                    demo.object_ids = list(obj_keys)
                    demo.objects_poses = full_obj_poses
                    demo.contacts = np.stack(
                        [assigned_contacts[k] for k in list(assigned_contacts.keys())]
                    )
                    demo.robot_pose = np.eye(4)
                    demo.furniture_ids = list(furniture_poses.keys())
                    demo.furniture_poses = np.stack(
                        [furniture_poses[k] for k in demo.furniture_ids]
                    )
                    demo.furniture_params = [
                        furniture_params[k] for k in demo.furniture_ids
                    ]
                    demo.save(overwrite=True)
